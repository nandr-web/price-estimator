// ---------------------------------------------------------------------------
// Types mirroring the FastAPI backend for the price-estimator API
// ---------------------------------------------------------------------------

// Quote lifecycle states
export type QuoteStatus = "draft" | "review" | "sent" | "won" | "lost" | "expired"

// Override reason categories (from the Python OverrideReasonCategory enum)
export type OverrideReasonCategory =
  | "material_hardness"
  | "geometry_complexity"
  | "surface_finish"
  | "tooling_difficulty"
  | "customer_relationship"
  | "scrap_risk"
  | "certification_requirements"
  | "other"

// Loss reason categories
export type LossReasonCategory =
  | "price_too_high"
  | "lead_time"
  | "went_with_competitor"
  | "scope_change"
  | "other"

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

export interface QuoteRequest {
  part_description: string
  material: string | null
  process: string | null
  quantity: number
  rush_job: boolean
  lead_time_weeks: number
  estimator: string | null
}

export interface QuoteResponse {
  quote_id: string
  estimate: number
  aggressive_estimate: number | null
  conservative_estimate: number | null
  typical_range: { low: number; high: number; coverage: number } | null
  warnings: string[]
  shap_explanation: { feature: string; contribution: number }[] | null
}

export interface OverrideRequest {
  human_price: number
  reason_category: OverrideReasonCategory | null
  reason_text: string | null
  estimator_id: string | null
}

export interface OverrideResponse {
  stored: boolean
  override_id: string
  delta_from_model: number
  delta_pct: number
}

export interface OutcomeRequest {
  outcome: "won" | "lost" | "expired"
  reason?: LossReasonCategory
  reason_text?: string
  final_negotiated_price?: number
  po_number?: string
}

// Full quote detail with lifecycle events
export interface QuoteDetail {
  quote_id: string
  status: QuoteStatus
  features: QuoteRequest
  original_estimate: number
  aggressive_estimate: number | null
  conservative_estimate: number | null
  typical_range: { low: number; high: number; coverage: number } | null
  warnings: string[]
  shap_explanation: { feature: string; contribution: number }[] | null
  override: {
    human_price: number
    reason_category: OverrideReasonCategory | null
    reason_text: string | null
    estimator_id: string | null
    delta_from_model: number
    overridden_at: string
  } | null
  outcome: {
    result: "won" | "lost" | "expired"
    reason: string | null
    reason_text: string | null
    final_negotiated_price: number | null
    po_number: string | null
    recorded_at: string
  } | null
  final_price: number
  created_at: string
  sent_at: string | null
  timeline: { event: string; timestamp: string; detail: string | null }[]
}

// Summary for list view
export interface QuoteSummary {
  quote_id: string
  status: QuoteStatus
  part_description: string
  material: string | null
  process: string | null
  quantity: number
  original_estimate: number
  final_price: number
  created_at: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const OVERRIDE_REASON_LABELS: Record<OverrideReasonCategory, string> = {
  material_hardness: "Material Hardness",
  geometry_complexity: "Geometry Complexity",
  surface_finish: "Surface Finish",
  tooling_difficulty: "Tooling Difficulty",
  customer_relationship: "Customer Relationship",
  scrap_risk: "Scrap Risk",
  certification_requirements: "Certification Requirements",
  other: "Other",
}

export const LOSS_REASON_LABELS: Record<LossReasonCategory, string> = {
  price_too_high: "Price Too High",
  lead_time: "Lead Time",
  went_with_competitor: "Went with Competitor",
  scope_change: "Scope Change",
  other: "Other",
}

export const MATERIALS: string[] = [
  "Aluminum 6061",
  "Aluminum 7075",
  "Inconel 718",
  "Stainless Steel 17-4 PH",
  "Titanium Grade 5",
]

export const PROCESSES: string[] = [
  "3-Axis Milling",
  "5-Axis Milling",
  "CNC Turning",
  "Surface Grinding",
  "Wire EDM",
]

export const VALID_QUANTITIES: number[] = [1, 5, 10, 20, 50, 100]

export const ESTIMATORS: string[] = ["Sato-san", "Suzuki-san", "Tanaka-san"]
