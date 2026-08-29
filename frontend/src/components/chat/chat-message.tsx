"use client"

import { useState } from "react"
import { type Message } from "@/hooks/use-chat"
import { cn } from "@/lib/utils"
import { Bot, User, RefreshCw, Clock, CheckCircle2, AlertTriangle, ChevronDown, ChevronRight, ThumbsUp, ThumbsDown } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import { ProcessTimeline } from "./process-timeline"
import { SourceList } from "./source-list"

interface ChatMessageProps {
  message: Message
  onFeedback?: (messageId: string, rating: 0 | 1) => void
}

interface AnswerSections {
  conclusion: string
  evidence: string
  sources: string
  uncertain: string
  nextSteps: string
}

function parseAnswerSections(text: string): AnswerSections {
  const sections: AnswerSections = {
    conclusion: "",
    evidence: "",
    sources: "",
    uncertain: "",
    nextSteps: "",
  }

  // Split by markdown headers like **结论：** or **结论：**
  const patterns: [keyof AnswerSections, RegExp][] = [
    ["conclusion", /\*\*结论[：:]\*\*/],
    ["evidence", /\*\*依据[：:]\*\*/],
    ["sources", /\*\*引用来源[：:]\*\*/],
    ["uncertain", /\*\*不确定信息[：:]\*\*/],
    ["nextSteps", /\*\*建议下一步[：:]\*\*/],
  ]

  for (let i = 0; i < patterns.length; i++) {
    const [key, pattern] = patterns[i]
    const match = pattern.exec(text)
    if (!match) continue

    const startIdx = match.index + match[0].length
    // Find the next section header (or end of string)
    let endIdx = text.length
    for (let j = i + 1; j < patterns.length; j++) {
      const nextMatch = patterns[j][1].exec(text)
      if (nextMatch && nextMatch.index > match.index) {
        endIdx = nextMatch.index
        break
      }
    }

    sections[key] = text.slice(startIdx, endIdx).trim()
  }

  return sections
}

function isRefusal(text: string): boolean {
  return /知识库中未找到/.test(text) || /拒答/.test(text)
}

function SectionBlock({ title, content, defaultOpen = false }: { title: string; content: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)

  if (!content) return null

  return (
    <div className="mb-2">
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex w-full items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer py-0.5">
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          {title}
          {content.length > 80 && (
            <span className="text-[10px] text-muted-foreground/60 ml-auto">
              {open ? "收起" : `${content.length} 字`}
            </span>
          )}
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-1 text-sm leading-relaxed prose-chat">
          <div dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }} />
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}

export function ChatMessage({ message, onFeedback }: ChatMessageProps) {
  const isUser = message.role === "user"
  const isStreaming = message.isStreaming

  if (isUser) {
    return (
      <div className="flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
          <User size={15} className="text-primary" />
        </div>
        <div className="flex-1 pt-1">
          <p className="text-sm font-medium text-foreground/80 mb-1">你</p>
          <p className="text-sm text-foreground whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    )
  }

  const sections = parseAnswerSections(message.content)
  const hasSections = Object.values(sections).some((s) => s.length > 0)

  return (
    <div className="flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/15">
        <Bot size={15} className="text-primary" />
      </div>
      <div className="flex-1 pt-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          <p className="text-sm font-medium text-foreground/80">AI 助手</p>
          {message.mode && (
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 font-mono">
              {message.mode}
            </Badge>
          )}
          {message.agentInfo?.status && (
            <Badge
              variant="secondary"
              className={
                message.agentInfo.status === "ABSTAIN"
                  ? "text-[10px] px-1.5 py-0 h-4 bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                  : "text-[10px] px-1.5 py-0 h-4"
              }
            >
              {message.agentInfo.status}
            </Badge>
          )}
          {message.elapsed !== undefined && (
            <span className="flex items-center gap-1 text-[11px] text-muted-foreground ml-auto">
              <Clock size={11} />
              {message.elapsed.toFixed(1)}s
            </span>
          )}
        </div>

        {isStreaming ? (
          <div className="flex items-center gap-2 py-2">
            <RefreshCw size={14} className="animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">正在生成回答...</span>
          </div>
        ) : (
          <>
            {message.agentInfo?.route && message.agentInfo.route.length > 0 && (
              <div className="mb-2">
                <Badge
                  variant="outline"
                  className="text-[11px] bg-accent/30 border-accent text-accent-foreground"
                >
                  路由: {message.agentInfo.route.join(" → ")}
                </Badge>
              </div>
            )}

            {hasSections || isRefusal(message.content) ? (
              <div className="space-y-2">
                {sections.conclusion && (
                  <div className="rounded-lg border border-border bg-card/50 px-3 py-2">
                    <div className="text-xs font-semibold text-foreground/70 mb-1">📌 结论</div>
                    <div
                      className="text-sm leading-relaxed prose-chat"
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(sections.conclusion) }}
                    />
                  </div>
                )}
                {sections.nextSteps && (
                  <div className="rounded-lg border border-border bg-card/50 px-3 py-2">
                    <div className="text-xs font-semibold text-foreground/70 mb-1">👉 建议下一步</div>
                    <div
                      className="text-sm leading-relaxed prose-chat"
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(sections.nextSteps) }}
                    />
                  </div>
                )}
                <div className="space-y-0.5">
                  <SectionBlock title="📋 依据" content={sections.evidence} />
                  <SectionBlock title="📚 引用来源" content={sections.sources} />
                </div>
                {isRefusal(message.content) && !sections.conclusion && (
                  <div
                    className="prose-chat text-sm"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
                  />
                )}
              </div>
            ) : (
              <>
                <div
                  className="prose-chat text-sm"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
                />

                {/* 两段式：快答后有 verboseContent 且未展开 → 显示"展开详细"按钮 */}
                {message.verboseContent && !message.expanded && (
                  <button
                    onClick={() => {
                      // Call back to parent to expand
                      const ev = new CustomEvent("expand-answer", { detail: message.id })
                      window.dispatchEvent(ev)
                    }}
                    className="mt-2 inline-flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors cursor-pointer"
                  >
                    📖 展开详细回答
                  </button>
                )}
              </>
            )}

            {/* Process Log */}
            {message.processLog && message.processLog.length > 0 && (
              <div className="mt-4">
                <ProcessTimeline logs={message.processLog} />
              </div>
            )}

            {/* Sources */}
            {message.sources && message.sources.length > 0 && (
              <div className="mt-3">
                <SourceList sources={message.sources} />
              </div>
            )}

            {/* Feedback buttons */}
            {!message.isStreaming && message.content && (
              <div className="mt-3 flex items-center gap-2">
                <span className="text-[11px] text-muted-foreground/50">这个回答对你有帮助吗？</span>
                <button
                  onClick={() => onFeedback?.(message.id, 1)}
                  className="inline-flex items-center gap-1 text-[11px] text-muted-foreground/60 hover:text-green-600 transition-colors"
                  title="有帮助"
                >
                  <ThumbsUp size={13} />
                </button>
                <button
                  onClick={() => onFeedback?.(message.id, 0)}
                  className="inline-flex items-center gap-1 text-[11px] text-muted-foreground/60 hover:text-red-500 transition-colors"
                  title="没帮助"
                >
                  <ThumbsDown size={13} />
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function renderMarkdown(text: string): string {
  // Simple but safe markdown-to-HTML rendering
  let html = text
    // Headers (decrease heading level for nested display)
    .replace(/^### (.+)$/gm, '<h4 class="text-sm font-semibold my-1.5">$1</h4>')
    .replace(/^## (.+)$/gm, '<h3 class="text-sm font-semibold my-1.5">$1</h3>')
    .replace(/^# (.+)$/gm, '<h2 class="text-sm font-semibold my-1.5">$1</h2>')
    // Bold & italic
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code class="bg-muted px-1 rounded text-xs">$1</code>')
    // Blockquotes
    .replace(/^> (.+)$/gm, '<blockquote class="border-l-2 border-muted-foreground/20 pl-3 my-1 text-muted-foreground text-xs"><p>$1</p></blockquote>')
    // Horizontal rules
    .replace(/^---+$/gm, '<hr class="my-2" />')
    // Unordered list
    .replace(/^- (.+)$/gm, '<li class="ml-4 list-disc text-sm">$1</li>')
    // Ordered list
    .replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal text-sm">$1</li>')
    // Paragraphs (double newlines)
    .replace(/\n\n/g, '</p><p class="text-sm leading-relaxed">')
    // Line breaks
    .replace(/\n/g, '<br />')

  return `<p class="text-sm leading-relaxed">${html}</p>`
}
