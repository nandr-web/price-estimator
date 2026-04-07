import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { BrowserRouter, Route, Routes } from "react-router-dom"
import { Layout } from "@/components/layout"
import { QuoteListPage } from "@/pages/quote-list"
import { QuoteDetailPage } from "@/pages/quote-detail"
import { NewQuotePage } from "@/pages/new-quote"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<QuoteListPage />} />
            <Route path="/quote/new" element={<NewQuotePage />} />
            <Route path="/quote/:id" element={<QuoteDetailPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
