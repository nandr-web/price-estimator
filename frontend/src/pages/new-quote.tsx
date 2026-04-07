import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useCreateQuote } from "@/hooks/use-quotes"
import { ESTIMATORS, MATERIALS, PROCESSES, VALID_QUANTITIES } from "@/lib/types"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"

export function NewQuotePage() {
  const navigate = useNavigate()
  const createQuote = useCreateQuote()

  const [form, setForm] = useState({
    part_description: "",
    material: "",
    process: "",
    quantity: 1,
    rush_job: false,
    lead_time_weeks: 4,
    estimator: "",
  })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    createQuote.mutate(
      {
        part_description: form.part_description,
        material: form.material || null,
        process: form.process || null,
        quantity: form.quantity,
        rush_job: form.rush_job,
        lead_time_weeks: form.lead_time_weeks,
        estimator: form.estimator || null,
      },
      {
        onSuccess: (data) => {
          navigate(`/quote/${data.quote_id}`)
        },
      }
    )
  }

  return (
    <div className="mx-auto max-w-2xl">
      <Card>
        <CardHeader>
          <CardTitle>New Quote</CardTitle>
          <CardDescription>
            Enter job details to get an AI price estimate.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="part_description">Part Description</Label>
              <Input
                id="part_description"
                placeholder="e.g. Bracket Assembly, Turbine Blade"
                value={form.part_description}
                onChange={(e) =>
                  setForm({ ...form, part_description: e.target.value })
                }
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Material</Label>
                <Select
                  value={form.material}
                  onValueChange={(v) => setForm({ ...form, material: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select material" />
                  </SelectTrigger>
                  <SelectContent>
                    {MATERIALS.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Process</Label>
                <Select
                  value={form.process}
                  onValueChange={(v) => setForm({ ...form, process: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select process" />
                  </SelectTrigger>
                  <SelectContent>
                    {PROCESSES.map((p) => (
                      <SelectItem key={p} value={p}>
                        {p}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Quantity</Label>
                <Select
                  value={String(form.quantity)}
                  onValueChange={(v: string) =>
                    setForm({ ...form, quantity: parseInt(v) })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VALID_QUANTITIES.map((q) => (
                      <SelectItem key={q} value={String(q)}>
                        {q}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="lead_time">Lead Time (weeks)</Label>
                <Input
                  id="lead_time"
                  type="number"
                  min={1}
                  max={52}
                  value={form.lead_time_weeks}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      lead_time_weeks: parseInt(e.target.value) || 1,
                    })
                  }
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Estimator</Label>
                <Select
                  value={form.estimator}
                  onValueChange={(v) => setForm({ ...form, estimator: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Optional" />
                  </SelectTrigger>
                  <SelectContent>
                    {ESTIMATORS.map((e) => (
                      <SelectItem key={e} value={e}>
                        {e}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-end space-x-3 pb-1">
                <Switch
                  id="rush_job"
                  checked={form.rush_job}
                  onCheckedChange={(v) => setForm({ ...form, rush_job: v })}
                />
                <Label htmlFor="rush_job">Rush Job</Label>
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => navigate("/")}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={createQuote.isPending}>
                {createQuote.isPending ? "Estimating..." : "Get Estimate"}
              </Button>
            </div>

            {createQuote.isError && (
              <p className="text-sm text-destructive">
                {(createQuote.error as Error).message}
              </p>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
