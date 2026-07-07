import type { SimilarIssueResult } from '../api/types'
import { formatSimilarity, githubIssueUrl } from '../lib/format'

interface SimilarIssuesListProps {
  repository: string
  similarIssues: SimilarIssueResult[]
}

export function SimilarIssuesList({ repository, similarIssues }: SimilarIssuesListProps) {
  if (similarIssues.length === 0) {
    return (
      <section className="subpanel">
        <h3>Similar historical issues</h3>
        <p className="muted">No similar issues returned.</p>
      </section>
    )
  }

  return (
    <section className="subpanel">
      <h3>Similar historical issues</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Issue</th>
            <th>Similarity</th>
            <th>Labels</th>
            <th>Overlap</th>
          </tr>
        </thead>
        <tbody>
          {similarIssues.map((issue) => (
            <tr key={issue.rank}>
              <td>{issue.rank}</td>
              <td>
                <a
                  href={githubIssueUrl(repository, issue.issue_number)}
                  target="_blank"
                  rel="noreferrer"
                >
                  #{issue.issue_number}
                </a>
              </td>
              <td>{formatSimilarity(issue.similarity)}</td>
              <td>
                {issue.neighbor_selected_labels.length > 0
                  ? issue.neighbor_selected_labels.join(', ')
                  : '—'}
              </td>
              <td>
                {issue.predicted_label_overlap.length > 0
                  ? issue.predicted_label_overlap.join(', ')
                  : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
