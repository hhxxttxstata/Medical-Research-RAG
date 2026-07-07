"use client"

import { type Message } from "@/hooks/use-chat"
import { cn } from "@/lib/utils"
import { Bot, User, RefreshCw, Clock, CheckCircle2, AlertTriangle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { ProcessTimeline } from "./process-timeline"
import { SourceList } from "./source-list"

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
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
          {message.agentInfo?.tool && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4">
              {message.agentInfo.tool}
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
            {message.agentInfo?.intent && (
              <div className="mb-2">
                <Badge
                  variant="outline"
                  className="text-[11px] bg-accent/30 border-accent text-accent-foreground"
                >
                  意图: {message.agentInfo.intent}
                </Badge>
              </div>
            )}

            <div
              className="prose-chat text-sm"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
            />

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
          </>
        )}
      </div>
    </div>
  )
}

function renderMarkdown(text: string): string {
  // Simple but safe markdown-to-HTML rendering
  // For production, use a proper markdown library like react-markdown
  let html = text
    // Headers
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold & italic
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Blockquotes
    .replace(/^> (.+)$/gm, '<blockquote><p>$1</p></blockquote>')
    // Horizontal rules
    .replace(/^---+$/gm, '<hr />')
    // Paragraphs (double newlines)
    .replace(/\n\n/g, '</p><p>')
    // Line breaks
    .replace(/\n/g, '<br />')

  return `<p>${html}</p>`
}
