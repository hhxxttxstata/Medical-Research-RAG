"use client"

import { type ProcessLogEntry } from "@/lib/api"
import { cn } from "@/lib/utils"
import { CheckCircle2, Loader2, XCircle, Ban } from "lucide-react"

interface ProcessTimelineProps {
  logs: ProcessLogEntry[]
}

const statusConfig = {
  ok: { icon: CheckCircle2, color: "text-green-600 dark:text-green-400", bg: "bg-green-100 dark:bg-green-900/20" },
  running: { icon: Loader2, color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-100 dark:bg-blue-900/20" },
  error: { icon: XCircle, color: "text-red-600 dark:text-red-400", bg: "bg-red-100 dark:bg-red-900/20" },
  blocked: { icon: Ban, color: "text-orange-600 dark:text-orange-400", bg: "bg-orange-100 dark:bg-orange-900/20" },
}

export function ProcessTimeline({ logs }: ProcessTimelineProps) {
  return (
    <div className="rounded-lg border border-border bg-card/50 p-3">
      <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">处理过程</p>
      <div className="space-y-1.5">
        {logs.map((log, i) => {
          const cfg = statusConfig[log.status] || statusConfig.ok
          const Icon = cfg.icon
          const isLast = i === logs.length - 1

          return (
            <div key={i} className="flex gap-2.5">
              <div className="flex flex-col items-center">
                <div className={cn("flex h-5 w-5 items-center justify-center rounded-full", cfg.bg)}>
                  <Icon size={11} className={cn(cfg.color, log.status === "running" && "animate-spin")} />
                </div>
                {!isLast && <div className="mt-0.5 w-px flex-1 bg-border" />}
              </div>
              <div className={cn("pb-1.5", isLast && "pb-0")}>
                <p className="text-xs font-medium text-foreground/80">{log.step}</p>
                <p className="text-[11px] text-muted-foreground">{log.detail}</p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
