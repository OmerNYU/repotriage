import type { HealthState } from '../hooks/useHealth'

interface HealthIndicatorProps {
  health: HealthState
}

export function HealthIndicator({ health }: HealthIndicatorProps) {
  if (health.loading) {
    return <span className="health-badge health-loading">Checking API…</span>
  }

  if (health.error || !health.data) {
    return <span className="health-badge health-error">API unreachable</span>
  }

  return (
    <span className="health-badge health-ok" title={health.data.inference_config_path}>
      Connected · {health.data.repository}
    </span>
  )
}
