import type { ClassificationResult } from '../api/types'
import { formatScore } from '../lib/format'

interface ClassificationPanelProps {
  classification: ClassificationResult
}

export function ClassificationPanel({ classification }: ClassificationPanelProps) {
  const { predicted_labels, scores, threshold, threshold_basis_points } = classification

  return (
    <section className="subpanel">
      <h3>Classification</h3>
      <p className="meta">
        Threshold: {threshold.toFixed(2)} ({threshold_basis_points} bps)
      </p>

      <div className="label-chips">
        <span className="label-section-title">Predicted labels</span>
        {predicted_labels.length === 0 ? (
          <span className="muted">No labels predicted</span>
        ) : (
          predicted_labels.map((item) => (
            <span key={item.label} className="chip chip-predicted">
              {item.label} <em>{formatScore(item.score)}</em>
            </span>
          ))
        )}
      </div>

      <details className="score-details">
        <summary>All label scores</summary>
        <table className="data-table">
          <thead>
            <tr>
              <th>Label</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {scores.map((item) => (
              <tr key={item.label}>
                <td>{item.label}</td>
                <td>{formatScore(item.score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  )
}
