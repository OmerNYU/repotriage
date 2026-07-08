import type { SimilarIssueResult } from '../api/types'
import { formatSimilarity, githubIssueUrl } from '../lib/format'

interface SimilarIssuesListProps {
  repository: string
  similarIssues: SimilarIssueResult[]
}

function LabelChips({ labels }: { labels: string[] }) {
  if (labels.length === 0) {
    return <span className="muted">—</span>
  }
  return (
    <span className="mini-chip-row">
      {labels.map((label) => (
        <span key={label} className="mini-chip">
          {label}
        </span>
      ))}
    </span>
  )
}

export function SimilarIssuesList({ repository, similarIssues }: SimilarIssuesListProps) {
  if (similarIssues.length === 0) {
    return (
      <section className="subpanel zone">
        <h3>Similar historical issues</h3>
        <p className="empty-callout empty-callout-compact">
          No similar issues returned. Try a higher count or more specific text.
        </p>
      </section>
    )
  }

  return (
    <section className="subpanel zone">
      <h3>Similar historical issues</h3>

      <div className="similar-table-wrap">
        <table className="data-table similar-table">
          <caption className="sr-only">Similar historical issues ranked by similarity</caption>
          <thead>
            <tr>
              <th scope="col">Rank</th>
              <th scope="col">Issue</th>
              <th scope="col">Similarity</th>
              <th scope="col">Labels</th>
              <th scope="col">Overlap</th>
            </tr>
          </thead>
          <tbody>
            {similarIssues.map((issue) => (
              <tr key={issue.rank}>
                <td className="mono">{issue.rank}</td>
                <td>
                  <a
                    className="issue-link"
                    href={githubIssueUrl(repository, issue.issue_number)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    #{issue.issue_number}
                  </a>
                </td>
                <td className="mono similarity-cell">{formatSimilarity(issue.similarity)}</td>
                <td>
                  <LabelChips labels={issue.neighbor_selected_labels} />
                </td>
                <td>
                  <LabelChips labels={issue.predicted_label_overlap} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="similar-cards">
        {similarIssues.map((issue) => (
          <article key={issue.rank} className="similar-card">
            <div className="similar-card-header">
              <a
                className="issue-link"
                href={githubIssueUrl(repository, issue.issue_number)}
                target="_blank"
                rel="noreferrer"
              >
                #{issue.issue_number}
              </a>
              <span className="mono similarity-cell">{formatSimilarity(issue.similarity)}</span>
            </div>
            <dl className="similar-card-meta">
              <dt>Rank</dt>
              <dd className="mono">{issue.rank}</dd>
              <dt>Labels</dt>
              <dd>
                <LabelChips labels={issue.neighbor_selected_labels} />
              </dd>
              <dt>Overlap</dt>
              <dd>
                <LabelChips labels={issue.predicted_label_overlap} />
              </dd>
            </dl>
          </article>
        ))}
      </div>
    </section>
  )
}
