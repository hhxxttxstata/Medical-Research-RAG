"use client"

import { useCallback, useRef, useState } from "react"
import { api, type ChatResponse, type Source, type ProcessLogEntry, type AgentInfo, type FeedbackRequest } from "@/lib/api"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
  sources?: Source[]
  agentInfo?: AgentInfo | null
  processLog?: ProcessLogEntry[]
  elapsed?: number
  mode?: string
  isStreaming?: boolean
  verboseContent?: string
  expanded?: boolean
}

export interface DiagnoseResult {
  probability: number
  prediction: number
  riskLevel: string
  filename: string
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const sessionIdRef = useRef<string>("")
  const conversationIdRef = useRef(0)

  const sendMessage = useCallback(async (question: string, mode?: "auto" | "rag" | "agent") => {
    setIsLoading(true)
    const userMsgId = `msg-${++conversationIdRef.current}`
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: question, timestamp: Date.now() },
    ])

    const assistantMsgId = `msg-${++conversationIdRef.current}`
    setMessages((prev) => [
      ...prev,
      { id: assistantMsgId, role: "assistant", content: "", timestamp: Date.now(), isStreaming: true },
    ])

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, mode: mode ?? "auto" }),
      })
      if (!res.ok) throw new Error(`SSE ${res.status}`)

      const reader = res.body?.getReader()
      if (!reader) throw new Error("No reader")
      const decoder = new TextDecoder()
      let verboseAnswer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const lines = decoder.decode(value, { stream: true }).split("\n")
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          try {
            const { event, data } = JSON.parse(line.slice(6))

            if (event === "quick_answer") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, content: data, isStreaming: false } : m,
                ),
              )
            } else if (event === "verbose_answer") {
              verboseAnswer = data
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId && m.expanded
                    ? { ...m, content: data, verboseContent: data }
                    : m.id === assistantMsgId
                    ? { ...m, verboseContent: data }
                    : m,
                ),
              )
            } else if (event === "answer") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, content: data, isStreaming: false } : m,
                ),
              )
            } else if (event === "error") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, content: `❌ ${data}`, isStreaming: false } : m,
                ),
              )
            }
          } catch {
            /* partial line skip */
          }
        }
      }
      // If we got verbose_answer but never quick_answer (weird), fallback
      if (verboseAnswer) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId && !m.content
              ? { ...m, content: verboseAnswer, verboseContent: verboseAnswer, isStreaming: false }
              : m,
          ),
        )
      }
    } catch {
      // Fallback to blocking POST
      try {
        const res: ChatResponse = await api.chat({ question, mode: mode ?? "auto" })
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: res.answer, sources: res.sources, elapsed: res.elapsed, mode: res.mode, isStreaming: false }
              : m,
          ),
        )
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "请求失败"
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, content: `❌ ${msg}`, isStreaming: false } : m,
          ),
        )
      }
    } finally {
      setIsLoading(false)
    }
  }, [])

  const expandAnswer = useCallback((messageId: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === messageId && m.verboseContent
          ? { ...m, content: m.verboseContent, expanded: true }
          : m,
      ),
    )
  }, [])

  const diagnose = useCallback(async (file: File) => {
    setIsLoading(true)
    const id = `msg-${++conversationIdRef.current}`
    setMessages((prev) => [
      ...prev,
      { id, role: "assistant", content: "", timestamp: Date.now(), isStreaming: true },
    ])
    try {
      const res = await api.diagnose(file)
      const riskEmoji = res.risk_level === "高风险" ? "🔴" : res.risk_level === "中风险" ? "🟡" : res.risk_level === "低风险" ? "🟢" : "✅"
      const report = [
        `## 🩺 肺栓塞诊断报告`,
        ``,
        `| 项目 | 结果 |`,
        `|------|------|`,
        `| 📂 影像文件 | \`${res.filename}\` |`,
        `| ${riskEmoji} 诊断结果 | **${res.risk_level}** (${res.prediction ? "阳性" : "阴性"}) |`,
        `| 📊 肺栓塞概率 | **${(res.probability * 100).toFixed(2)}%** |`,
        `| ⏱️ 推理耗时 | ${res.inference_time.toFixed(2)}s |`,
        ``,
        `> ⚠️ **免责声明:** 本结果为 AI 辅助诊断建议，仅供参考。`,
      ].join("\n")
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, content: report, isStreaming: false, elapsed: res.total_time } : m)))
    } catch {
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, content: "❌ 诊断失败", isStreaming: false } : m)))
    } finally {
      setIsLoading(false)
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    sessionIdRef.current = ""
  }, [])

  const submitFeedback = useCallback(async (message: Message, rating: 0 | 1) => {
    // 找到前一条用户消息作为 question
    const userMsg = messages.find((m) => m.role === "user" && m.timestamp < message.timestamp)
    try {
      await api.feedback({
        question: userMsg?.content || "",
        answer: message.content,
        rating,
        reason: "",
        message_id: message.id,
      })
    } catch {
      // silent fail
    }
  }, [messages])

  return { messages, isLoading, sendMessage, expandAnswer, diagnose, clearMessages, submitFeedback }
}
