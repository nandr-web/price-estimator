import { formatCurrency } from "@/lib/utils"

interface EstimateRangeProps {
  estimate: number
  aggressive: number | null
  conservative: number | null
  range: { low: number; high: number; coverage: number } | null
  override?: number
}

export function EstimateRange({
  estimate,
  aggressive,
  conservative,
  range,
  override,
}: EstimateRangeProps) {
  const low = aggressive ?? range?.low ?? estimate * 0.9
  const high = conservative ?? range?.high ?? estimate * 1.1
  const span = high - low
  const estimatePos = ((estimate - low) / span) * 100

  const overridePos =
    override != null ? ((override - low) / span) * 100 : null

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="text-3xl font-semibold tabular-nums">
          {formatCurrency(override ?? estimate)}
        </span>
        {conservative && (
          <span className="text-sm text-muted-foreground">
            conservative: {formatCurrency(conservative)}
          </span>
        )}
      </div>

      {/* Range bar */}
      <div className="relative h-3 rounded-full bg-muted">
        {/* Typical range band */}
        {range && (
          <div
            className="absolute inset-y-0 rounded-full bg-primary/15"
            style={{
              left: `${((range.low - low) / span) * 100}%`,
              right: `${100 - ((range.high - low) / span) * 100}%`,
            }}
          />
        )}

        {/* Estimate marker */}
        <div
          className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-background"
          style={{ left: `${estimatePos}%` }}
        />

        {/* Override marker */}
        {overridePos != null && (
          <div
            className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-amber-500 bg-background"
            style={{ left: `${Math.max(0, Math.min(100, overridePos))}%` }}
          />
        )}
      </div>

      {/* Labels below the bar */}
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{formatCurrency(low)} {aggressive ? "(win bid)" : ""}</span>
        <span>{formatCurrency(estimate)} (estimate)</span>
        <span>{formatCurrency(high)} {range ? `(${(range.coverage * 100).toFixed(0)}% range)` : ""}</span>
      </div>
    </div>
  )
}
