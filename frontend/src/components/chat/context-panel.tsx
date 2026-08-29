"use client"

import { type Message } from "@/hooks/use-chat"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import {
  X,
  Clock,
  FileText,
  Cpu,
  Brain,
  AlertTriangle,
  Lightbulb,
  Target,
  Route,
} from "lucide-react"
import { ProcessTimeline } from "./process-timeline"
import { SourceList } from "./source-list"

interface ContextPanelProps {
  message: Message
  onClose: () => void
}

export function ContextPanel({ message, onClose }: ContextPanelProps) {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-border px-4">
        <h3 className="text-sm font-semibold text-foreground/80">上下文详情</h3>
        <Button variant="ghost" size="icon-xs" onClick={onClose}>
          <X size={15} />
        </Button>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-5">
          {/* Metadata summary */}
          <Section title="基本信息">
            <InfoRow icon={<Clock size={13} />} label="耗时" value={`${message.elapsed?.toFixed(2) ?? "?"}s`} />
            <InfoRow icon={<Cpu size={13} />} label="模式" value={message.mode ?? "?"} badge />
            <InfoRow icon={<Brain size={13} />} label="决策" value={message.agentInfo?.status ?? "RAG 直答"} />
            <InfoRow icon={<Target size={13} />} label="路由" value={message.agentInfo?.route?.join(" → ") ?? "—"} />
            {message.agentInfo?.iterations !== undefined && (
              <InfoRow icon={<Route size={13} />} label="决策迭代" value={`${message.agentInfo.iterations}`} />
            )}
          </Section>

          {/* Agent info */}
          {message.agentInfo && (
            <Section title="Agent 决策">
              <div className="flex items-center gap-2 rounded-md bg-accent/20 p-2 text-xs">
                <Lightbulb size={13} className="text-primary shrink-0" />
                <span>
                  决策状态: <strong>{message.agentInfo.status ?? "—"}</strong>
                  {message.agentInfo.grader_called !== undefined && (
                    <span className="text-muted-foreground">
                      {" "}(grader: {message.agentInfo.grader_called ? "已调用" : "未调用"})
                    </span>
                  )}
                </span>
              </div>
              {message.agentInfo.status === "ABSTAIN" && message.agentInfo.abstain_reason && (
                <div className="flex items-center gap-2 rounded-md bg-yellow-50 dark:bg-yellow-900/10 p-2 text-xs text-yellow-700 dark:text-yellow-400">
                  <AlertTriangle size={13} className="shrink-0" />
                  <span>拒答原因: {message.agentInfo.abstain_reason}</span>
                </div>
              )}
            </Section>
          )}

          {/* Process Timeline */}
          {message.processLog && message.processLog.length > 0 && (
            <Section title="处理过程">
              <ProcessTimeline logs={message.processLog} />
            </Section>
          )}

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <Section title={`引用来源 (${message.sources.length})`}>
              <SourceList sources={message.sources} />
            </Section>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        {title}
      </h4>
      <div className="space-y-1.5">{children}</div>
    </div>
  )
}

function InfoRow({
  icon,
  label,
  value,
  badge,
}: {
  icon?: React.ReactNode
  label: string
  value: string
  badge?: boolean
}) {
  return (
    <div className="flex items-center gap-2 text-xs text-foreground/70">
      {icon && <span className="shrink-0 text-muted-foreground">{icon}</span>}
      <span className="min-w-[48px] text-muted-foreground">{label}</span>
      {badge ? (
        <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 font-mono">
          {value}
        </Badge>
      ) : (
        <span className="font-medium text-foreground/80 truncate">{value}</span>
      )}
    </div>
  )
}
