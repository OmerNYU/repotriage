import type { InferenceResponse } from '../api/types'
import { formatConfidence } from '../lib/format'

interface ResultsSummaryProps {
  result: InferenceResponse
}

export function ResultsSummary({ result }: ResultsSummaryProps) {
  const shouldAbstain = result.abstention.should_abstain
  const topLabel = result.classification.predicted_labels[0]?.label ?? 'No labels'
  const similarCount = result.retrieval.similar_issues.length
  const decision = shouldAbstain ? 'Abstain' : 'Proceed'

  return (
    <div className="results-summary" role="region" aria-label="Inference summary">
      <div className="summary-stat">
        <span className="summary-label">Decision</span>
        <span
          className={`summary-value summary-decision summary-decision-${shouldAbstain ? 'abstain' : 'proceed'}`}
        >
          <span className="summary-decision-dot" aria-hidden="true" />
          {decision}
        </span>
      </div>
      <div className="summary-stat">
        <span className="summary-label">Top label</span>
        <span className="summary-value">{topLabel}</span>
      </div>
      <div className="summary-stat">
        <span className="summary-label">Confidence</span>
        <span className="summary-value mono">{formatConfidence(result.abstention.confidence)}</span>
      </div>
      <div className="summary-stat">
        <span className="summary-label">Similar issues</span>
        <span className="summary-value mono">{similarCount}</span>
      </div>
    </div>
  )
}
