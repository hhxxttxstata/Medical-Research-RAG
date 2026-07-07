"use client"

import { type Source } from "@/lib/api"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { FileText, ExternalLink } from "lucide-react"

interface SourceListProps {
  sources: Source[]
}

export function SourceList({ sources }: SourceListProps) {
  return (
    <div>
      <p className="mb-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wider">
        引用来源 ({sources.length})
      </p>
      <div className="space-y-1">
        {sources.slice(0, 5).map((s, i) => (
          <div
            key={s.id || i}
            className="flex items-start gap-2 rounded-md border border-border bg-card p-2 text-xs"
          >
            <FileText size={12} className="mt-0.5 shrink-0 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium text-foreground/80">
                {s.filename || "未知来源"}
              </p>
              <p className="mt-0.5 text-muted-foreground line-clamp-2">{s.text}</p>
            </div>
            <span
              className={cn(
                "shrink-0 rounded px-1 py-0.5 text-[10px] font-medium",
                s.score > 0.7
                  ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                  : s.score > 0.4
                    ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                    : "bg-muted text-muted-foreground",
              )}
            >
              {(s.score * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
