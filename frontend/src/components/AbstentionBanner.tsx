import type { AbstentionResult } from '../api/types'
import { formatConfidence, humanizeAbstentionReason } from '../lib/format'

interface AbstentionBannerProps {
  abstention: AbstentionResult
}

export function AbstentionBanner({ abstention }: AbstentionBannerProps) {
  const variant = abstention.should_abstain ? 'abstain' : 'confident'
  const title = abstention.should_abstain
    ? 'Recommend abstain — low confidence for auto-triage'
    : 'Confident enough — proceed with predicted labels'

  return (
    <section className={`abstention-banner abstention-${variant}`}>
      <h3>{title}</h3>
      <ul className="abstention-details">
        <li>
          <strong>Confidence:</strong> {formatConfidence(abstention.confidence)}
        </li>
        <li>
          <strong>Threshold:</strong> {abstention.threshold.toFixed(2)} (
          {abstention.threshold_basis_points} bps)
        </li>
        <li>
          <strong>Reason:</strong> {humanizeAbstentionReason(abstention.reason)}
        </li>
      </ul>
    </section>
  )
}
