import type { InferenceResponse } from '../api/types'
import { humanizeWarning } from '../lib/format'
import { AbstentionBanner } from './AbstentionBanner'
import { ClassificationPanel } from './ClassificationPanel'
import { DebugDetails } from './DebugDetails'
import { SimilarIssuesList } from './SimilarIssuesList'

interface InferenceResultsProps {
  result: InferenceResponse
}

export function InferenceResults({ result }: InferenceResultsProps) {
  return (
    <section className="panel">
      <h2>Inference results</h2>
      <p className="meta">
        Repository: <strong>{result.repository}</strong> · Generated at{' '}
        {new Date(result.generated_at).toLocaleString()}
      </p>

      {result.warnings.length > 0 && (
        <div className="alert alert-warning" role="status">
          <strong>Warnings:</strong>{' '}
          {result.warnings.map(humanizeWarning).join(' · ')}
        </div>
      )}

      <AbstentionBanner abstention={result.abstention} />
      <ClassificationPanel classification={result.classification} />
      <SimilarIssuesList
        repository={result.repository}
        similarIssues={result.retrieval.similar_issues}
      />
      <DebugDetails artifacts={result.artifacts} reproducibility={result.reproducibility} />
    </section>
  )
}
