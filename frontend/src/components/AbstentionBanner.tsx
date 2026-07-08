import type { AbstentionResult } from '../api/types'
import { formatConfidence, humanizeAbstentionReason } from '../lib/format'

interface AbstentionBannerProps {
  abstention: AbstentionResult
}

export function AbstentionBanner({ abstention }: AbstentionBannerProps) {
  const variant = abstention.should_abstain ? 'abstain' : 'confident'
  const badge = abstention.should_abstain ? 'Abstain' : 'Proceed'
  const title = abstention.should_abstain
    ? 'Hold auto-triage — confidence below threshold'
    : 'Proceed — confidence meets threshold'

  return (
    <section
      className={`decision-card decision-${variant} zone`}
      role="status"
      aria-live="polite"
    >
      <div className="decision-card-header">
        <span className="decision-badge">{badge}</span>
        <h3 className="decision-title">{title}</h3>
      </div>
      <dl className="decision-meta">
        <div>
          <dt>Confidence</dt>
          <dd className="mono">{formatConfidence(abstention.confidence)}</dd>
        </div>
        <div>
          <dt>Threshold</dt>
          <dd className="mono">
            {abstention.threshold.toFixed(2)} ({abstention.threshold_basis_points} bps)
          </dd>
        </div>
        <div className="decision-meta-reason">
          <dt>Reason</dt>
          <dd>{humanizeAbstentionReason(abstention.reason)}</dd>
        </div>
      </dl>
    </section>
  )
}
