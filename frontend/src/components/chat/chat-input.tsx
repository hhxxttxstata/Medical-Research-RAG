"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Send, Paperclip, X } from "lucide-react"

interface ChatInputProps {
  onSend: (text: string, file?: File) => void
  isLoading: boolean
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [text, setText] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = () => {
    const trimmed = text.trim()
    if (!trimmed && !file) return
    onSend(trimmed, file ?? undefined)
    setText("")
    setFile(null)
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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) setFile(f)
  }

  const handleInput = () => {
    const el = inputRef.current
    if (el) {
      el.style.height = "auto"
      el.style.height = Math.min(el.scrollHeight, 200) + "px"
    }
  }

  // ── drag & drop support ────────────────────────────
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) setFile(f)
  }

  return (
    <div className="flex flex-col gap-2">
      {/* File preview */}
      {file && (
        <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3 py-1.5 text-sm">
          <Paperclip size={14} className="text-muted-foreground shrink-0" />
          <span className="truncate text-muted-foreground">{file.name}</span>
          <span className="text-xs text-muted-foreground/60 shrink-0">
            ({(file.size / 1024 / 1024).toFixed(1)} MB)
          </span>
          <Button
            variant="ghost"
            size="icon-xs"
            className="ml-auto shrink-0"
            onClick={() => { setFile(null); if (fileInputRef.current) fileInputRef.current.value = "" }}
          >
            <X size={14} />
          </Button>
        </div>
      )}

      <div
        className={cn(
          "flex items-end gap-2 rounded-xl border bg-card px-3 py-2",
          "transition-all",
          isDragging
            ? "border-primary border-dashed ring-2 ring-primary/20 bg-primary/5"
            : "border-border",
          "focus-within:border-ring focus-within:ring-1 focus-within:ring-ring/30",
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".nii,.nii.gz,.pdf,.md,.txt"
          className="hidden"
          onChange={handleFileSelect}
        />
        <Button
          variant="ghost"
          size="icon-xs"
          className="shrink-0 text-muted-foreground hover:text-foreground"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
        >
          <Paperclip size={16} />
        </Button>
        <textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="输入问题，或上传 CTPA 影像自动诊断..."
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
          disabled={(!text.trim() && !file) || isLoading}
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
        支持文字提问、上传 CTPA 影像诊断、拖拽 PDF/MD/TXT 文档
      </p>
    </div>
  )
}
