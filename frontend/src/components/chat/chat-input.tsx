"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Send } from "lucide-react"

interface ChatInputProps {
  onSend: (text: string) => void
  isLoading: boolean
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [text, setText] = useState("")
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = () => {
    const trimmed = text.trim()
    if (!trimmed) return
    onSend(trimmed)
    setText("")
    if (inputRef.current) {
      inputRef.current.style.height = "auto"
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleInput = () => {
    const el = inputRef.current
    if (el) {
      el.style.height = "auto"
      el.style.height = Math.min(el.scrollHeight, 200) + "px"
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div
        className={cn(
          "flex items-end gap-2 rounded-xl border bg-card px-3 py-2",
          "transition-all",
          "border-border",
          "focus-within:border-ring focus-within:ring-1 focus-within:ring-ring/30",
        )}
      >
        <textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="输入肺栓塞科研文献问题，如：急性与慢性肺栓塞在 CTPA 影像上如何鉴别？"
          rows={1}
          className={cn(
            "flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground/60",
            "border-0 outline-none ring-0 focus:ring-0",
            "min-h-[20px] max-h-[200px]",
          )}
          disabled={isLoading}
        />
        <Button
          size="icon-sm"
          onClick={handleSubmit}
          disabled={!text.trim() || isLoading}
          className="shrink-0"
        >
          {isLoading ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-background border-t-transparent" />
          ) : (
            <Send size={15} />
          )}
        </Button>
      </div>
      <p className="px-1 text-[11px] text-muted-foreground/50">
        仅科研辅助：回答均附文献引用编号 [N]；域外或诊断类问题将明确拒答
      </p>
    </div>
  )
}
