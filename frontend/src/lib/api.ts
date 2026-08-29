const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"

export interface ChatRequest {
  question: string
  top_k?: number | null
  mode?: "auto" | "rag" | "agent"
  session_id?: string | null
}

export interface Source {
  id: string
  filename: string
  page?: string
  score: number
  text: string
}

export interface AgentInfo {
  status?: string
  route?: string[]
  iterations?: number
  grader_called?: boolean
  latency_ms?: number
  abstain_reason?: string | null
}

export interface ProcessLogEntry {
  step: string
  detail: string
  status: "ok" | "running" | "error" | "blocked"
}

export interface ChatResponse {
  success: boolean
  answer: string
  mode: string
  sources: Source[]
  elapsed: number
  is_refusal: boolean
  agent_info: AgentInfo | null
  process_log: ProcessLogEntry[]
  session_id: string
}

export interface HealthResponse {
  status: string
  version: string
  knowledge_base: {
    chunk_count?: number
    data_dir?: string
    embedding?: string
    top_k?: number
    chunk_range?: string
  } | null
  timestamp: string
}

export interface FeedbackRequest {
  question: string
  answer: string
  rating: 0 | 1
  reason: string
  message_id: string
}

export interface StatsResponse {
  date: string
  total_queries: number
  success_count: number
  error_count: number
  refusal_count: number
  refusal_rate: number
  avg_response_time: number
}

class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
  }
}

async function postJSON<T>(url: string, data: unknown, sessionId?: string): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(sessionId ? { "X-Session-ID": sessionId } : {}),
    },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => "Unknown error")
    throw new ApiError(res.status, detail)
  }
  return res.json()
}

async function postFormData<T>(url: string, formData: FormData): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => "Unknown error")
    throw new ApiError(res.status, detail)
  }
  return res.json()
}

export const api = {
  health(): Promise<HealthResponse> {
    return fetch(`${API_BASE}/health`).then((r) => r.json())
  },

  chat(req: ChatRequest, sessionId?: string): Promise<ChatResponse> {
    return postJSON<ChatResponse>(`${API_BASE}/chat`, req, sessionId)
  },

  stats(): Promise<StatsResponse> {
    return fetch(`${API_BASE}/stats`).then((r) => r.json())
  },

  uploadDocument(file: File): Promise<Response> {
    const fd = new FormData()
    fd.append("file", file)
    fd.append("auto_index", "true")
    return fetch(`${API_BASE}/documents/upload`, { method: "POST", body: fd })
  },

  logs(n = 10): Promise<{ records: Array<Record<string, unknown>> }> {
    return fetch(`${API_BASE}/logs?n=${n}`).then((r) => r.json())
  },

  feedback(req: FeedbackRequest): Promise<{ success: boolean }> {
    return postJSON(`${API_BASE}/feedback`, req)
  },
}
