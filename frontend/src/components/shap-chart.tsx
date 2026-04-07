import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts"
import { formatCurrency } from "@/lib/utils"

interface ShapFeature {
  feature: string
  contribution: number
}

interface ShapChartProps {
  features: ShapFeature[]
}

export function ShapChart({ features }: ShapChartProps) {
  const sorted = [...features]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8)

  const data = sorted.map((f) => ({
    name: f.feature.replace(/_/g, " "),
    value: f.contribution,
  }))

  return (
    <ResponsiveContainer width="100%" height={sorted.length * 36 + 20}>
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 40 }}>
        <XAxis
          type="number"
          tickFormatter={(v) => formatCurrency(v)}
          tick={{ fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={140}
          tick={{ fontSize: 12 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          formatter={(value) => [formatCurrency(Number(value)), "Contribution"]}
          contentStyle={{
            fontSize: 12,
            borderRadius: 8,
            border: "1px solid var(--color-border)",
          }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={
                entry.value >= 0
                  ? "oklch(0.646 0.222 41.116)"   /* chart-1 warm */
                  : "oklch(0.6 0.118 184.714)"     /* chart-2 cool */
              }
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
