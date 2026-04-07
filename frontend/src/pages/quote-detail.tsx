import { useState } from "react"
import { Link, useParams } from "react-router-dom"
import {
  useQuote,
  useOverrideQuote,
  useSendQuote,
  useRecordOutcome,
} from "@/hooks/use-quotes"
import { formatCurrency, formatPercent } from "@/lib/utils"
import {
  OVERRIDE_REASON_LABELS,
  LOSS_REASON_LABELS,
  type OverrideReasonCategory,
  type LossReasonCategory,
  type QuoteStatus,
} from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"
import { ShapChart } from "@/components/shap-chart"
import { EstimateRange } from "@/components/estimate-range"

const STATUS_VARIANT: Record<
  QuoteStatus,
  "default" | "secondary" | "outline" | "success" | "warning" | "destructive"
> = {
  draft: "secondary",
  review: "default",
  sent: "outline",
  won: "success",
  lost: "secondary",
  expired: "secondary",
}

const STATUS_LABEL: Record<QuoteStatus, string> = {
  draft: "Draft",
  review: "In Review",
  sent: "Sent",
  won: "Won",
  lost: "Lost",
  expired: "Expired",
}

export function QuoteDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { data: quote, isLoading, error } = useQuote(id!)

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading quote...</p>
  }
  if (error) {
    return (
      <p className="text-sm text-destructive">
        Failed to load quote: {(error as Error).message}
      </p>
    )
  }
  if (!quote) return null

  const canOverride = quote.status === "draft" || quote.status === "review"
  const canRecordOutcome = quote.status === "sent"

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            Quotes
          </Link>
          <span className="text-muted-foreground">/</span>
          <span className="font-mono text-sm">{quote.quote_id}</span>
          <Badge variant={STATUS_VARIANT[quote.status]}>
            {STATUS_LABEL[quote.status]}
          </Badge>
        </div>
      </div>

      {/* Job Details */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Job Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm sm:grid-cols-3">
            <div>
              <span className="text-muted-foreground">Part</span>
              <p className="font-medium">{quote.features.part_description}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Material</span>
              <p className="font-medium">
                {quote.features.material ?? "Not specified"}
              </p>
            </div>
            <div>
              <span className="text-muted-foreground">Process</span>
              <p className="font-medium">
                {quote.features.process ?? "Not specified"}
              </p>
            </div>
            <div>
              <span className="text-muted-foreground">Quantity</span>
              <p className="font-medium">{quote.features.quantity}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Lead Time</span>
              <p className="font-medium">
                {quote.features.lead_time_weeks} weeks
              </p>
            </div>
            <div>
              <span className="text-muted-foreground">Rush</span>
              <p className="font-medium">
                {quote.features.rush_job ? "Yes" : "No"}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Estimate */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">AI Estimate</CardTitle>
          {quote.override && (
            <CardDescription>
              Overridden to {formatCurrency(quote.override.human_price)} (
              {formatPercent(
                ((quote.override.human_price - quote.original_estimate) /
                  quote.original_estimate) *
                  100
              )}
              )
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          <EstimateRange
            estimate={quote.original_estimate}
            aggressive={quote.aggressive_estimate}
            conservative={quote.conservative_estimate}
            range={quote.typical_range}
            override={quote.override?.human_price}
          />

          {quote.warnings.length > 0 && (
            <div className="space-y-1">
              {quote.warnings.map((w, i) => (
                <p
                  key={i}
                  className="text-sm text-amber-600 before:mr-1.5 before:content-['!']"
                >
                  {w}
                </p>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* SHAP Explanation */}
      {quote.shap_explanation && quote.shap_explanation.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Why This Price</CardTitle>
            <CardDescription>
              Top features driving the estimate
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ShapChart features={quote.shap_explanation} />
          </CardContent>
        </Card>
      )}

      {/* Action Panel — state-dependent */}
      {canOverride && <ReviewActionPanel quoteId={quote.quote_id} estimate={quote.original_estimate} />}
      {canRecordOutcome && <OutcomeActionPanel quoteId={quote.quote_id} finalPrice={quote.final_price} />}

      {/* Timeline */}
      {quote.timeline.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {quote.timeline.map((event, i) => (
                <div key={i} className="flex gap-4 text-sm">
                  <span className="w-32 shrink-0 text-muted-foreground">
                    {new Date(event.timestamp).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                  <div>
                    <span className="font-medium">{event.event}</span>
                    {event.detail && (
                      <span className="text-muted-foreground">
                        {" "}
                        — {event.detail}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function ReviewActionPanel({
  quoteId,
  estimate,
}: {
  quoteId: string
  estimate: number
}) {
  const overrideMutation = useOverrideQuote(quoteId)
  const sendMutation = useSendQuote(quoteId)
  const [overridePrice, setOverridePrice] = useState("")
  const [reasonCategory, setReasonCategory] = useState("")
  const [reasonText, setReasonText] = useState("")

  const isOverriding = overridePrice !== "" && parseFloat(overridePrice) !== estimate
  const isPending = overrideMutation.isPending || sendMutation.isPending

  function handleApproveAndSend() {
    sendMutation.mutate()
  }

  function handleOverrideAndSend() {
    overrideMutation.mutate(
      {
        human_price: parseFloat(overridePrice),
        reason_category: (reasonCategory as OverrideReasonCategory) || null,
        reason_text: reasonText || null,
        estimator_id: null,
      },
      {
        onSuccess: () => {
          sendMutation.mutate()
        },
      }
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Review & Send</CardTitle>
        <CardDescription>
          Approve the AI estimate or adjust the price before sending.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="override_price">Final Price</Label>
            <Input
              id="override_price"
              type="number"
              step="0.01"
              placeholder={estimate.toFixed(2)}
              value={overridePrice}
              onChange={(e) => setOverridePrice(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Leave empty to approve the AI estimate.
            </p>
          </div>

          {isOverriding && (
            <div className="space-y-2">
              <Label>Override Reason</Label>
              <Select
                value={reasonCategory}
                onValueChange={setReasonCategory}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select reason" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(OVERRIDE_REASON_LABELS).map(([k, v]) => (
                    <SelectItem key={k} value={k}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {isOverriding && (
          <div className="space-y-2">
            <Label htmlFor="reason_text">Note (optional)</Label>
            <Textarea
              id="reason_text"
              placeholder="Additional context for this override..."
              value={reasonText}
              onChange={(e) => setReasonText(e.target.value)}
              rows={2}
            />
          </div>
        )}

        <Separator />

        <div className="flex justify-end gap-3">
          {isOverriding ? (
            <Button onClick={handleOverrideAndSend} disabled={isPending}>
              {isPending ? "Sending..." : `Override & Send (${formatCurrency(parseFloat(overridePrice))})`}
            </Button>
          ) : (
            <Button onClick={handleApproveAndSend} disabled={isPending}>
              {isPending ? "Sending..." : `Approve & Send (${formatCurrency(estimate)})`}
            </Button>
          )}
        </div>

        {(overrideMutation.isError || sendMutation.isError) && (
          <p className="text-sm text-destructive">
            {((overrideMutation.error || sendMutation.error) as Error).message}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

function OutcomeActionPanel({
  quoteId,
  finalPrice,
}: {
  quoteId: string
  finalPrice: number
}) {
  const outcomeMutation = useRecordOutcome(quoteId)
  const [lossReason, setLossReason] = useState("")
  const [lossReasonText, setLossReasonText] = useState("")
  const [showLostForm, setShowLostForm] = useState(false)
  const [poNumber, setPoNumber] = useState("")

  function handleWon() {
    outcomeMutation.mutate({
      outcome: "won",
      po_number: poNumber || undefined,
    })
  }

  function handleLost() {
    outcomeMutation.mutate({
      outcome: "lost",
      reason: (lossReason as LossReasonCategory) || undefined,
      reason_text: lossReasonText || undefined,
    })
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Record Outcome</CardTitle>
        <CardDescription>
          Sent at {formatCurrency(finalPrice)} — how did it go?
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!showLostForm ? (
          <div className="flex items-end gap-4">
            <div className="flex-1 space-y-2">
              <Label htmlFor="po_number">PO Number (optional)</Label>
              <Input
                id="po_number"
                placeholder="e.g. PO-2026-0412"
                value={poNumber}
                onChange={(e) => setPoNumber(e.target.value)}
              />
            </div>
            <div className="flex gap-3">
              <Button
                onClick={handleWon}
                disabled={outcomeMutation.isPending}
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                {outcomeMutation.isPending ? "Saving..." : "Mark Won"}
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowLostForm(true)}
                disabled={outcomeMutation.isPending}
              >
                Mark Lost
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Reason for Loss</Label>
                <Select value={lossReason} onValueChange={setLossReason}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select reason" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(LOSS_REASON_LABELS).map(([k, v]) => (
                      <SelectItem key={k} value={k}>
                        {v}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="loss_note">Note (optional)</Label>
                <Textarea
                  id="loss_note"
                  placeholder="Additional context..."
                  value={lossReasonText}
                  onChange={(e) => setLossReasonText(e.target.value)}
                  rows={2}
                />
              </div>
            </div>
            <div className="flex justify-end gap-3">
              <Button
                variant="outline"
                onClick={() => setShowLostForm(false)}
              >
                Back
              </Button>
              <Button
                variant="destructive"
                onClick={handleLost}
                disabled={outcomeMutation.isPending}
              >
                {outcomeMutation.isPending ? "Saving..." : "Confirm Lost"}
              </Button>
            </div>
          </div>
        )}

        {outcomeMutation.isError && (
          <p className="text-sm text-destructive">
            {(outcomeMutation.error as Error).message}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
