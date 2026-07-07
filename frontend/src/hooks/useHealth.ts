import { useCallback, useEffect, useState } from 'react'
import { getHealth } from '../api/client'
import type { HealthResponse } from '../api/types'

export interface HealthState {
  data: HealthResponse | null
  loading: boolean
  error: string | null
}

export function useHealth() {
  const [state, setState] = useState<HealthState>({
    data: null,
    loading: true,
    error: null,
  })

  const refresh = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }))
    try {
      const data = await getHealth()
      setState({ data, loading: false, error: null })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'API unreachable'
      setState({ data: null, loading: false, error: message })
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { ...state, refresh }
}
