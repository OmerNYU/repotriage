import type { HealthState } from '../hooks/useHealth'

interface HealthIndicatorProps {
  health: HealthState
}

export function HealthIndicator({ health }: HealthIndicatorProps) {
  if (health.loading) {
    return (
      <span className="health-badge health-loading" role="status" aria-live="polite">
        <span className="health-dot" aria-hidden="true" />
        Checking API…
      </span>
    )
  }

  if (health.error || !health.data) {
    return (
      <span className="health-badge health-error" role="status" aria-live="polite">
        <span className="health-dot" aria-hidden="true" />
        API unreachable
      </span>
    )
  }

  return (
    <span
      className="health-badge health-ok"
      role="status"
      aria-live="polite"
      title={health.data.inference_config_path}
    >
      <span className="health-dot" aria-hidden="true" />
      Connected · {health.data.repository}
    </span>
  )
}
