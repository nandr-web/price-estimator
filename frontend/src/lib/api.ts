import type {
  QuoteRequest,
  QuoteResponse,
  QuoteDetail,
  QuoteSummary,
  OverrideRequest,
  OverrideResponse,
  OutcomeRequest,
} from "./types"

// ---------------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------------

const BASE = "/v1"

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    // Backend uses error envelope: { detail: { error: { message } } }
    // or validation errors: { error: { message, details } }
    const detail = body.detail
    const message =
      (typeof detail === "object" && detail?.error?.message) ||
      body.error?.message ||
      (typeof detail === "string" ? detail : null) ||
      `Request failed: ${res.status}`
    throw new Error(message)
  }
  return res.json()
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export function createQuote(req: QuoteRequest): Promise<QuoteResponse> {
  return request<QuoteResponse>("/quote", {
    method: "POST",
    body: JSON.stringify(req),
  })
}

export function getQuote(id: string): Promise<QuoteDetail> {
  return request<QuoteDetail>(`/quote/${id}`)
}

export function listQuotes(): Promise<QuoteSummary[]> {
  return request<QuoteSummary[]>("/quotes")
}

export function overrideQuote(id: string, req: OverrideRequest): Promise<OverrideResponse> {
  return request<OverrideResponse>(`/quote/${id}/override`, {
    method: "POST",
    body: JSON.stringify(req),
  })
}

export async function sendQuote(id: string): Promise<void> {
  await request<{ ok: true }>(`/quote/${id}/send`, {
    method: "POST",
  })
}

export async function recordOutcome(id: string, req: OutcomeRequest): Promise<void> {
  await request<{ ok: true }>(`/quote/${id}/outcome`, {
    method: "POST",
    body: JSON.stringify(req),
  })
}
