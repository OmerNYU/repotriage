import type {
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  InferRequest,
  InferenceResponse,
} from './types'

const base = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') {
      return body.detail
    }
    if (Array.isArray(body.detail)) {
      return JSON.stringify(body.detail)
    }
  } catch {
    // fall through
  }
  return res.statusText || `HTTP ${res.status}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}${path}`, init)
  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export async function postInfer(body: InferRequest): Promise<InferenceResponse> {
  return request<InferenceResponse>('/api/v1/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function postFeedback(body: FeedbackRequest): Promise<FeedbackResponse> {
  return request<FeedbackResponse>('/api/v1/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}
