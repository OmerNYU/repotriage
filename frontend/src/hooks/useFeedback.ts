import { useCallback, useState } from 'react'
import { ApiError, postFeedback } from '../api/client'
import type { FeedbackRequest, FeedbackResponse } from '../api/types'

export function useFeedback() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<FeedbackResponse | null>(null)

  const submit = useCallback(async (request: FeedbackRequest) => {
    setLoading(true)
    setError(null)
    setSuccess(null)
    try {
      const response = await postFeedback(request)
      setSuccess(response)
      return response
    } catch (err) {
      const message =
        err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Feedback failed'
      setError(message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setLoading(false)
    setError(null)
    setSuccess(null)
  }, [])

  return { loading, error, success, submit, reset }
}
