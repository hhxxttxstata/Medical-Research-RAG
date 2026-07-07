"use client"

import { useState } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import { useChat, type Message } from "@/hooks/use-chat"
import { ChatMessage } from "@/components/chat/chat-message"
import { ChatInput } from "@/components/chat/chat-input"
import { ContextPanel } from "@/components/chat/context-panel"
import {
  PanelLeft,
  Plus,
  MessageSquare,
  Stethoscope,
  BookOpen,
  Settings,
  BarChart3,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"

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

export function Workbench() {
  const { messages, isLoading, sendMessage, diagnose, clearMessages } = useChat()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [activeNav, setActiveNav] = useState("chat")
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null)

  const handleSend = (text: string, file?: File) => {
    if (file) {
      diagnose(file)
    } else {
      sendMessage(text)
    }
  }

  const handleMessageClick = (msg: Message) => {
    if (msg.role === "assistant") {
      setSelectedMessage(msg)
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
        {/* Logo / Top */}
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

        {/* Navigation */}
        <div className="flex flex-1 flex-col gap-1 p-2">
          <NavItem
            icon={<Plus size={16} />}
            label="新建对话"
            onClick={clearMessages}
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
            icon={<Stethoscope size={16} />}
            label="诊断工具"
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<BookOpen size={16} />}
            label="知识库"
            collapsed={sidebarCollapsed}
          />
          <NavItem
            icon={<BarChart3 size={16} />}
            label="统计"
            collapsed={sidebarCollapsed}
          />
          <div className="mt-auto">
            <Separator className="my-2" />
            <NavItem
              icon={<Settings size={16} />}
              label="设置"
              collapsed={sidebarCollapsed}
            />
          </div>
        </div>
      </div>

      {/* ── 中间聊天区 ── */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <div className="flex h-14 items-center gap-3 border-b border-border px-6">
          <Button
            variant="ghost"
            size="icon-sm"
            className="md:hidden"
            onClick={() => setSidebarCollapsed(false)}
          >
            <PanelLeft size={16} />
          </Button>
          <h2 className="text-sm font-semibold text-foreground/80">对话</h2>
          {messages.length > 0 && (
            <Badge variant="secondary" className="ml-auto text-xs">
              {messages.filter((m) => m.role === "assistant" && !m.isStreaming).length} 条回答
            </Badge>
          )}
        </div>

        {/* Messages area */}
        <ScrollArea className="flex-1 px-4 py-4">
          <div className="mx-auto max-w-4xl space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center py-24 text-center">
                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                  <Stethoscope className="h-8 w-8 text-primary" />
                </div>
                <h3 className="mb-2 text-lg font-semibold text-foreground/80">
                  肺栓塞智能问诊系统
                </h3>
                <p className="mb-8 max-w-md text-sm text-muted-foreground">
                  基于 RAG 检索增强生成与 Agent 智能路由，支持知识问答、报告生成与 CTPA 影像诊断。
                </p>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <button
                    onClick={() => handleSend("什么是肺栓塞？有哪些临床症状？")}
                    className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                  >
                    <span className="font-medium text-foreground/80">❓ 什么是肺栓塞？</span>
                    <p className="mt-1 text-xs text-muted-foreground">了解疾病基础知识</p>
                  </button>
                  <button
                    onClick={() => handleSend("肺栓塞的 Wells 评分是什么？如何计算？")}
                    className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                  >
                    <span className="font-medium text-foreground/80">📋 Wells 评分</span>
                    <p className="mt-1 text-xs text-muted-foreground">了解临床评估工具</p>
                  </button>
                  <button
                    onClick={() => handleSend("帮我生成一份关于肺栓塞诊断与治疗的综述报告")}
                    className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                  >
                    <span className="font-medium text-foreground/80">📄 生成报告</span>
                    <p className="mt-1 text-xs text-muted-foreground">Agent 自动撰写综述</p>
                  </button>
                  <button
                    onClick={() => handleSend("CTPA 影像如何诊断肺栓塞？")}
                    className="rounded-xl border border-border bg-card p-4 text-left text-sm hover:bg-accent transition-colors"
                  >
                    <span className="font-medium text-foreground/80">🩻 影像诊断</span>
                    <p className="mt-1 text-xs text-muted-foreground">了解诊断流程</p>
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
                <ChatMessage message={msg} />
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
        </ScrollArea>

        {/* Input */}
        <div className="border-t border-border px-4 py-3">
          <div className="mx-auto max-w-4xl">
            <ChatInput onSend={handleSend} isLoading={isLoading} />
          </div>
        </div>
      </div>

      {/* ── 右侧上下文面板 ── */}
      {selectedMessage && (
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
