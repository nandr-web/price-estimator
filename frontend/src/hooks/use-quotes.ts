import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  OutcomeRequest,
  OverrideRequest,
  QuoteRequest,
} from "@/lib/types"
import * as api from "@/lib/api"

export function useQuotes() {
  return useQuery({
    queryKey: ["quotes"],
    queryFn: api.listQuotes,
  })
}

export function useQuote(id: string) {
  return useQuery({
    queryKey: ["quote", id],
    queryFn: () => api.getQuote(id),
    enabled: !!id,
  })
}

export function useCreateQuote() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (req: QuoteRequest) => api.createQuote(req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quotes"] })
    },
  })
}

export function useOverrideQuote(quoteId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (req: OverrideRequest) => api.overrideQuote(quoteId, req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quote", quoteId] })
      queryClient.invalidateQueries({ queryKey: ["quotes"] })
    },
  })
}

export function useSendQuote(quoteId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.sendQuote(quoteId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quote", quoteId] })
      queryClient.invalidateQueries({ queryKey: ["quotes"] })
    },
  })
}

export function useRecordOutcome(quoteId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (req: OutcomeRequest) => api.recordOutcome(quoteId, req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quote", quoteId] })
      queryClient.invalidateQueries({ queryKey: ["quotes"] })
    },
  })
}
