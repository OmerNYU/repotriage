import type { ClassificationResult } from '../api/types'
import { formatScore } from '../lib/format'

interface ClassificationPanelProps {
  classification: ClassificationResult
}

export function ClassificationPanel({ classification }: ClassificationPanelProps) {
  const { predicted_labels, scores, threshold, threshold_basis_points } = classification

  return (
    <section className="subpanel zone classification-panel">
      <div className="zone-header">
        <h3>Classification</h3>
        <p className="meta meta-quiet">
          Threshold {threshold.toFixed(2)} ({threshold_basis_points} bps)
        </p>
      </div>

      <div className="label-chips">
        <span className="label-section-title">Predicted labels</span>
        {predicted_labels.length === 0 ? (
          <span className="empty-callout empty-callout-compact">No labels above threshold</span>
        ) : (
          predicted_labels.map((item) => (
            <span key={item.label} className="chip chip-predicted">
              <span className="chip-label">{item.label}</span>
              <em className="chip-score">{formatScore(item.score)}</em>
            </span>
          ))
        )}
      </div>

      <details className="score-details tech-details">
        <summary>All label scores</summary>
        <div className="tech-details-body">
          <table className="data-table">
            <caption className="sr-only">All label scores for this inference</caption>
            <thead>
              <tr>
                <th scope="col">Label</th>
                <th scope="col">Score</th>
              </tr>
            </thead>
            <tbody>
              {scores.map((item) => (
                <tr key={item.label}>
                  <td>{item.label}</td>
                  <td className="mono">{formatScore(item.score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  )
}
