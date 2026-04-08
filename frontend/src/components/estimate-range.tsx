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
  const span = high - low || 1

  const pct = (v: number) => Math.max(0, Math.min(100, ((v - low) / span) * 100))
  const estimatePos = pct(estimate)
  const overridePos = override != null ? pct(override) : null

  return (
    <div className="space-y-1">
      {/* Large price display */}
      <span className="text-3xl font-semibold tabular-nums">
        {formatCurrency(override ?? estimate)}
      </span>

      {/* Number line */}
      <div className="relative mt-4 mb-8 mx-2">
        {/* Track */}
        <div className="h-1 rounded-full bg-muted" />

        {/* Range band */}
        {range && (
          <div
            className="absolute top-0 h-1 rounded-full bg-primary/20"
            style={{
              left: `${pct(range.low)}%`,
              width: `${pct(range.high) - pct(range.low)}%`,
            }}
          />
        )}

        {/* Tick: aggressive / low */}
        <Tick position={0} label={formatCurrency(low)} sublabel={aggressive ? "aggressive" : ""} />

        {/* Tick: estimate */}
        <Tick
          position={estimatePos}
          label={formatCurrency(estimate)}
          sublabel="estimate"
          emphasis
        />

        {/* Tick: override */}
        {overridePos != null && override != null && (
          <Tick
            position={overridePos}
            label={formatCurrency(override)}
            sublabel="override"
            color="amber"
          />
        )}

        {/* Tick: conservative / high */}
        <Tick
          position={100}
          label={formatCurrency(high)}
          sublabel={conservative ? "conservative" : ""}
          align="right"
        />
      </div>
    </div>
  )
}

function Tick({
  position,
  label,
  sublabel,
  emphasis,
  color,
  align,
}: {
  position: number
  label: string
  sublabel?: string
  emphasis?: boolean
  color?: "amber"
  align?: "right"
}) {
  const dotColor = color === "amber"
    ? "bg-amber-500"
    : emphasis
      ? "bg-primary"
      : "bg-muted-foreground/50"

  const textColor = color === "amber"
    ? "text-amber-600"
    : emphasis
      ? "text-foreground"
      : "text-muted-foreground"

  // Determine horizontal alignment of the label relative to the tick
  const alignClass = align === "right"
    ? "right-0 text-right"
    : position < 15
      ? "left-0 text-left"
      : "-translate-x-1/2 text-center"

  return (
    <div className="absolute top-0 -translate-y-1/2" style={{ left: `${position}%` }}>
      {/* Dot on the line */}
      <div
        className={`h-3 w-3 -translate-x-1/2 rounded-full border-2 border-background ${dotColor}`}
      />
      {/* Label below */}
      <div className={`absolute top-5 whitespace-nowrap ${alignClass}`}>
        <p className={`text-xs font-medium tabular-nums ${textColor}`}>{label}</p>
        {sublabel && (
          <p className="text-[10px] text-muted-foreground">{sublabel}</p>
        )}
      </div>
    </div>
  )
}
