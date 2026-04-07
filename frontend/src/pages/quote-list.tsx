import { Link } from "react-router-dom"
import { useQuotes } from "@/hooks/use-quotes"
import { formatCurrency } from "@/lib/utils"
import type { QuoteStatus } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

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

export function QuoteListPage() {
  const { data: quotes, isLoading, error } = useQuotes()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Quotes</h1>
          <p className="text-sm text-muted-foreground">
            Review, override, and track quote outcomes.
          </p>
        </div>
        <Button asChild>
          <Link to="/quote/new">New Quote</Link>
        </Button>
      </div>

      {isLoading && (
        <p className="text-sm text-muted-foreground">Loading quotes...</p>
      )}

      {error && (
        <p className="text-sm text-destructive">
          Failed to load quotes: {(error as Error).message}
        </p>
      )}

      {quotes && quotes.length === 0 && (
        <div className="rounded-lg border border-dashed p-12 text-center">
          <p className="text-muted-foreground">No quotes yet.</p>
          <Button asChild className="mt-4" variant="outline">
            <Link to="/quote/new">Create your first quote</Link>
          </Button>
        </div>
      )}

      {quotes && quotes.length > 0 && (
        <div className="rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[140px]">Quote ID</TableHead>
                <TableHead>Part</TableHead>
                <TableHead>Material</TableHead>
                <TableHead className="text-right">Qty</TableHead>
                <TableHead className="text-right">Estimate</TableHead>
                <TableHead className="text-right">Final</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Created</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {quotes.map((q) => (
                <TableRow key={q.quote_id}>
                  <TableCell>
                    <Link
                      to={`/quote/${q.quote_id}`}
                      className="font-mono text-sm text-primary hover:underline"
                    >
                      {q.quote_id}
                    </Link>
                  </TableCell>
                  <TableCell className="max-w-[200px] truncate">
                    {q.part_description}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {q.material ?? "—"}
                  </TableCell>
                  <TableCell className="text-right">{q.quantity}</TableCell>
                  <TableCell className="text-right font-mono">
                    {formatCurrency(q.original_estimate)}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {formatCurrency(q.final_price)}
                  </TableCell>
                  <TableCell>
                    <Badge variant={STATUS_VARIANT[q.status]}>
                      {STATUS_LABEL[q.status]}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {new Date(q.created_at).toLocaleDateString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
