import { useCallback, useState } from 'react'
import { ApiError, postInfer } from '../api/client'
import type { InferRequest, InferenceResponse } from '../api/types'

export interface IssueFormValues {
  title: string
  body: string
  top_k: number
}

export function useInference() {
  const [result, setResult] = useState<InferenceResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = useCallback(async (values: IssueFormValues) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const body: InferRequest = {
        title: values.title.trim(),
        body: values.body,
        top_k: values.top_k,
      }
      const response = await postInfer(body)
      setResult(response)
      return response
    } catch (err) {
      const message =
        err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Inference failed'
      setError(message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
    setLoading(false)
  }, [])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return { result, loading, error, submit, reset, clearError }
}
