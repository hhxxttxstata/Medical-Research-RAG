"use client"

import { useCallback, useRef, useState } from "react"
import { api, type ChatResponse, type Source, type ProcessLogEntry, type AgentInfo } from "@/lib/api"

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
    const userMessage: Message = {
      id: userMsgId,
      role: "user",
      content: question,
      timestamp: Date.now(),
    }
    setMessages((prev) => [...prev, userMessage])

    const assistantMsgId = `msg-${++conversationIdRef.current}`
    const placeholder: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      isStreaming: true,
    }
    setMessages((prev) => [...prev, placeholder])

    try {
      const res: ChatResponse = await api.chat(
        { question, mode: mode ?? "auto" },
        sessionIdRef.current || undefined,
      )
      sessionIdRef.current = res.session_id

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? {
                ...m,
                content: res.answer,
                sources: res.sources,
                agentInfo: res.agent_info,
                processLog: res.process_log,
                elapsed: res.elapsed,
                mode: res.mode,
                isStreaming: false,
              }
            : m,
        ),
      )
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "请求失败"
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, content: `❌ **错误**: ${errorMsg}`, isStreaming: false }
            : m,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }, [])

  const diagnose = useCallback(async (file: File) => {
    setIsLoading(true)
    const resultMsgId = `msg-${++conversationIdRef.current}`
    const placeholder: Message = {
      id: resultMsgId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      isStreaming: true,
    }
    setMessages((prev) => [...prev, placeholder])

    try {
      const res = await api.diagnose(file)

      const riskEmoji =
        res.risk_level === "高风险" ? "🔴" :
        res.risk_level === "中风险" ? "🟡" :
        res.risk_level === "低风险" ? "🟢" : "✅"

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

      setMessages((prev) =>
        prev.map((m) =>
          m.id === resultMsgId
            ? { ...m, content: report, isStreaming: false, elapsed: res.total_time }
            : m,
        ),
      )
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "诊断失败"
      setMessages((prev) =>
        prev.map((m) =>
          m.id === resultMsgId
            ? { ...m, content: `❌ **诊断失败**: ${errorMsg}`, isStreaming: false }
            : m,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    sessionIdRef.current = ""
  }, [])

  return {
    messages,
    isLoading,
    sendMessage,
    diagnose,
    clearMessages,
  }
}
