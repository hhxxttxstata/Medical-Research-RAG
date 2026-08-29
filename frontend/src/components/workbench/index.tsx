"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import { api, type StatsResponse, type HealthResponse } from "@/lib/api"
import { useChat, type Message } from "@/hooks/use-chat"
import { ChatMessage } from "@/components/chat/chat-message"
import { ChatInput } from "@/components/chat/chat-input"
import { ContextPanel } from "@/components/chat/context-panel"
import {
  PanelLeft,
  Plus,
  MessageSquare,
  BookOpen,
  Settings,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Upload,
  Activity,
  Database,
  Cpu,
  Server,
  HardDrive,
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
} from "lucide-react"

// ─── Types ────────────────────────────────────────

type NavPanel = "chat" | "knowledge" | "stats" | "settings"

interface NavItemProps {
  icon: React.ReactNode
  label: string
  isActive?: boolean
  onClick?: () => void
  collapsed?: boolean
}

function NavItem({ icon, label, isActive, onClick, collapsed }: NavItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
        "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
        isActive
          ? "bg-sidebar-accent text-sidebar-accent-foreground"
          : "text-sidebar-foreground/70",
        collapsed && "justify-center px-2",
      )}
    >
      <span className="shrink-0">{icon}</span>
      {!collapsed && <span className="truncate">{label}</span>}
    </button>
  )
}

// ─── Panel: 知识库 ───────────────────────────────

function KnowledgePanel() {
  const [collections, setCollections] = useState<string[]>([])
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const loadCollections = useCallback(async () => {
    setLoading(true)
    try {
      const health = await api.health()
      if (health.knowledge_base) {
        setCollections([health.knowledge_base.embedding || "rag_docs_c300_500"])
      }
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  useEffect(() => { loadCollections() }, [loadCollections])

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setUploadMsg(null)
    try {
      const res = await api.uploadDocument(file)
      const data = await res.json()
      setUploadMsg(data.message || `上传成功`)
      setFile(null)
      loadCollections()
    } catch (err: unknown) {
      setUploadMsg(err instanceof Error ? err.message : "上传失败")
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-6 p-6">
      <div>
        <h3 className="text-sm font-semibold text-foreground/80 flex items-center gap-2">
          <BookOpen size={16} /> 知识库管理
        </h3>
        <p className="text-xs text-muted-foreground mt-1">
          上传 PDF/MD/TXT 文档并自动索引到知识库
        </p>
      </div>

      {/* Upload */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-3">
          <input
            type="file"
            accept=".pdf,.md,.txt"
            id="kb-file-input"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <Button variant="outline" size="sm" onClick={() => document.getElementById("kb-file-input")?.click()}>
            <Upload size={14} className="mr-1" /> 选择文档
          </Button>
          {file && <span className="text-xs text-muted-foreground truncate flex-1">{file.name}</span>}
          {file && (
            <Button size="sm" onClick={handleUpload} disabled={uploading}>
              {uploading ? "上传中..." : "上传并索引"}
            </Button>
          )}
        </div>
        {uploadMsg && (
          <p className="mt-2 text-xs text-muted-foreground">{uploadMsg}</p>
        )}
      </div>

      {/* Collections */}
      <div>
        <h4 className="text-xs font-medium text-foreground/70 mb-2">集合列表</h4>
        {loading ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <RefreshCw size={12} className="animate-spin" /> 加载中...
          </div>
        ) : collections.length === 0 ? (
          <p className="text-xs text-muted-foreground">暂无集合</p>
        ) : (
          <div className="space-y-2">
            {collections.map((c) => (
              <div key={c} className="flex items-center gap-2 rounded-lg border border-border bg-card/50 px-3 py-2">
                <Database size={14} className="text-muted-foreground shrink-0" />
                <span className="text-xs font-mono">{c}</span>
                <Badge variant="outline" className="text-[10px] ml-auto">活跃</Badge>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Panel: 统计 ─────────────────────────────────

function StatsPanel() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.stats().catch(() => null),
      api.health().catch(() => null),
    ]).then(([s, h]) => {
      setStats(s)
      setHealth(h)
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
        <RefreshCw size={14} className="animate-spin" /> 加载统计...
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      <div>
        <h3 className="text-sm font-semibold text-foreground/80 flex items-center gap-2">
          <BarChart3 size={16} /> 运行统计
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatCard icon={<Activity size={16} />} label="总查询" value={String(stats?.total_queries ?? "-")} />
        <StatCard icon={<CheckCircle2 size={16} />} label="成功" value={String(stats?.success_count ?? "-")} color="text-green-600" />
        <StatCard icon={<XCircle size={16} />} label="失败" value={String(stats?.error_count ?? "-")} color="text-red-500" />
        <StatCard icon={<AlertTriangle size={16} />} label="拒答" value={String(stats?.refusal_count ?? "-")} color="text-amber-500" />
        <StatCard icon={<Activity size={16} />} label="拒答率" value={stats ? `${(stats.refusal_rate * 100).toFixed(1)}%` : "-"} />
        <StatCard icon={<Activity size={16} />} label="平均响应" value={stats ? `${stats.avg_response_time.toFixed(2)}s` : "-"} />
      </div>

      <Separator />

      <div>
        <h4 className="text-xs font-medium text-foreground/70 mb-3">系统信息</h4>
        <div className="space-y-2 text-xs">
          <SysInfoRow icon={<Server size={14} />} label="服务状态" value={health?.status ?? "未知"} />
          <SysInfoRow icon={<HardDrive size={14} />} label="版本" value={health?.version ?? "-"} />
          {health?.knowledge_base && (
            <>
              <SysInfoRow icon={<Database size={14} />} label="知识库 Chunks" value={String(health.knowledge_base.chunk_count ?? "-")} />
              <SysInfoRow icon={<Cpu size={14} />} label="Embedding" value={health.knowledge_base.embedding ?? "-"} />
              <SysInfoRow icon={<Activity size={14} />} label="Top-K" value={String(health.knowledge_base.top_k ?? "-")} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground mb-1">
        {icon} {label}
      </div>
      <div className={cn("text-lg font-semibold", color || "text-foreground")}>{value}</div>
    </div>
  )
}

function SysInfoRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground shrink-0">{icon}</span>
      <span className="text-muted-foreground/70">{label}:</span>
      <span className="ml-auto font-mono text-foreground/80">{value}</span>
    </div>
  )
}

// ─── Panel: 设置 ─────────────────────────────────

function SettingsPanel() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelStatus, setModelStatus] = useState<string | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => {})
  }, [])

  return (
    <div className="space-y-6 p-6">
      <div>
        <h3 className="text-sm font-semibold text-foreground/80 flex items-center gap-2">
          <Settings size={16} /> 设置
        </h3>
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <h4 className="text-xs font-semibold text-foreground/70 mb-3">系统状态</h4>
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground/70">后端状态</span>
            <Badge variant={health ? "default" : "destructive"} className="ml-auto text-[10px]">
              {health ? "正常运行" : "未连接"}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground/70">知识库</span>
            <span className="ml-auto text-foreground/80">
              {health?.knowledge_base?.chunk_count ?? "-"} Chunks
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground/70">Embedding 模型</span>
            <span className="ml-auto text-foreground/80 font-mono text-[10px]">
              {health?.knowledge_base?.embedding ?? "-"}
            </span>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <h4 className="text-xs font-semibold text-foreground/70 mb-3">API 端点</h4>
        <div className="space-y-1.5 text-xs">
          <code className="block rounded bg-muted px-2 py-1 text-[11px]">POST /chat — RAG 问答</code>
          <code className="block rounded bg-muted px-2 py-1 text-[11px]">POST /query — Agentic 问答（LangGraph）</code>
          <code className="block rounded bg-muted px-2 py-1 text-[11px]">GET /health — 健康检查</code>
          <code className="block rounded bg-muted px-2 py-1 text-[11px]">GET /stats — 运行统计</code>
          <code className="block rounded bg-muted px-2 py-1 text-[11px]">POST /documents/upload — 上传文档</code>
        </div>
      </div>
    </div>
  )
}

// ─── Main Workbench ──────────────────────────────

export function Workbench() {
  const { messages, isLoading, sendMessage, clearMessages, submitFeedback } = useChat()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [activeNav, setActiveNav] = useState<NavPanel>("chat")
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null)

  const handleSend = (text: string) => {
    sendMessage(text)
  }

  const handleMessageClick = (msg: Message) => {
    if (msg.role === "assistant") {
      setSelectedMessage(msg)
    }
  }

  const getPanelTitle = () => {
    switch (activeNav) {
      case "knowledge": return "知识库"
      case "stats": return "统计"
      case "settings": return "设置"
      default: return "对话"
    }
  }

  return (
    <div className="flex h-screen w-full bg-background">
      {/* ── 左侧导航栏 ── */}
      <div
        className={cn(
          "flex flex-col border-r border-border bg-sidebar transition-all duration-200",
          sidebarCollapsed ? "w-16" : "w-56",
        )}
      >
        <div className="flex h-14 items-center gap-2 border-b border-border px-4">
          {!sidebarCollapsed && (
            <span className="text-sm font-semibold text-sidebar-foreground tracking-tight">
              PE RAG System
            </span>
          )}
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className={cn("ml-auto shrink-0", sidebarCollapsed && "mx-auto")}
          >
            {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </Button>
        </div>

        <div className="flex flex-1 flex-col gap-1 p-2">
          <NavItem
            icon={<Plus size={16} />}
            label="新建对话"
            onClick={() => { clearMessages(); setActiveNav("chat") }}
            collapsed={sidebarCollapsed}
          />
          <Separator className="my-2" />
          <NavItem
            icon={<MessageSquare size={16} />}
            label="对话"
            isActive={activeNav === "chat"}
            onClick={() => setActiveNav("chat")}
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<BookOpen size={16} />}
            label="知识库"
            isActive={activeNav === "knowledge"}
            onClick={() => setActiveNav("knowledge")}
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<BarChart3 size={16} />}
            label="统计"
            isActive={activeNav === "stats"}
            onClick={() => setActiveNav("stats")}
            collapsed={sidebarCollapsed}
          />
          <div className="mt-auto">
            <Separator className="my-2" />
            <NavItem
              icon={<Settings size={16} />}
              label="设置"
              isActive={activeNav === "settings"}
              onClick={() => setActiveNav("settings")}
              collapsed={sidebarCollapsed}
            />
          </div>
        </div>
      </div>

      {/* ── 中间区域 ── */}
      <div className="flex flex-1 flex-col min-w-0">
        <div className="flex h-14 items-center gap-3 border-b border-border px-6">
          <Button
            variant="ghost"
            size="icon-sm"
            className="md:hidden"
            onClick={() => setSidebarCollapsed(false)}
          >
            <PanelLeft size={16} />
          </Button>
          <h2 className="text-sm font-semibold text-foreground/80">{getPanelTitle()}</h2>
          {activeNav === "chat" && messages.length > 0 && (
            <Badge variant="secondary" className="ml-auto text-xs">
              {messages.filter((m) => m.role === "assistant" && !m.isStreaming).length} 条回答
            </Badge>
          )}
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-auto">
          {activeNav === "chat" && (
            <>
              <div className="px-4 py-4">
                <div className="mx-auto max-w-4xl space-y-4">
                  {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-24 text-center">
                      <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                        <BookOpen className="h-8 w-8 text-primary" />
                      </div>
                      <h3 className="mb-2 text-lg font-semibold text-foreground/80">
                        肺栓塞科研文献问答助手
                      </h3>
                      <p className="mb-8 max-w-md text-sm text-muted-foreground">
                        面向肺栓塞中英文文献与论文写作规范的知识问答（仅科研辅助，不提供诊断建议）。
                      </p>
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                        <button
                          onClick={() => handleSend("急性肺栓塞和慢性肺栓塞在病理生理上有什么区别？")}
                          className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                        >
                          <span className="font-medium text-foreground/80">❓ 急慢性 PE 病理区别</span>
                          <p className="mt-1 text-xs text-muted-foreground">跨文档多跳检索示例</p>
                        </button>
                        <button
                          onClick={() => handleSend("sPESI 评分如何计算？预测 30 天死亡率的灵敏度是多少？")}
                          className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                        >
                          <span className="font-medium text-foreground/80">📋 sPESI 评分</span>
                          <p className="mt-1 text-xs text-muted-foreground">文献中的数值问答</p>
                        </button>
                        <button
                          onClick={() => handleSend("深度学习方法在肺栓塞 CTPA 检测中的研究进展有哪些？")}
                          className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                        >
                          <span className="font-medium text-foreground/80">📄 研究进展综述</span>
                          <p className="mt-1 text-xs text-muted-foreground">聚合多篇文献的回答</p>
                        </button>
                        <button
                          onClick={() => handleSend("如何撰写一篇结构规范的医学综述论文？")}
                          className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                        >
                          <span className="font-medium text-foreground/80">✍️ 论文写作规范</span>
                          <p className="mt-1 text-xs text-muted-foreground">写作规范知识域问答</p>
                        </button>
                      </div>
                    </div>
                  )}
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      onClick={() => handleMessageClick(msg)}
                      className={cn(
                        "transition-colors rounded-lg -mx-2 px-2 py-1",
                        selectedMessage?.id === msg.id && "bg-accent/30",
                        msg.role === "assistant" && "cursor-pointer",
                      )}
                    >
                      <ChatMessage message={msg} onFeedback={(mid, r) => submitFeedback(msg, r)} />
                    </div>
                  ))}
                  {messages.length > 0 && messages[messages.length - 1].role === "user" && isLoading && (
                    <div className="flex items-center gap-2 py-2 text-sm text-muted-foreground">
                      <span className="flex gap-1">
                        <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" />
                        <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" style={{ animationDelay: "0.3s" }} />
                        <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" style={{ animationDelay: "0.6s" }} />
                      </span>
                      <span>思考中...</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="border-t border-border px-4 py-3">
                <div className="mx-auto max-w-4xl">
                  <ChatInput onSend={handleSend} isLoading={isLoading} />
                </div>
              </div>
            </>
          )}

          {activeNav === "knowledge" && <KnowledgePanel />}
          {activeNav === "stats" && <StatsPanel />}
          {activeNav === "settings" && <SettingsPanel />}
        </div>
      </div>

      {/* ── 右侧上下文面板 ── */}
      {activeNav === "chat" && selectedMessage && (
        <div className="hidden w-80 border-l border-border bg-card xl:block">
          <ContextPanel
            message={selectedMessage}
            onClose={() => setSelectedMessage(null)}
          />
        </div>
      )}
    </div>
  )
}