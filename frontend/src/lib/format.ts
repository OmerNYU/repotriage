import type { AbstentionReason, InferenceWarning } from '../api/types'

export const BODY_PREVIEW_MAX_LENGTH = 200

export function truncateBodyPreview(body: string, maxLength = BODY_PREVIEW_MAX_LENGTH): string {
  if (body.length <= maxLength) {
    return body
  }
  return body.slice(0, maxLength)
}

export function formatScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

export function formatSimilarity(similarity: number): string {
  return `${(similarity * 100).toFixed(1)}%`
}

export function humanizeAbstentionReason(reason: AbstentionReason): string {
  switch (reason) {
    case 'no_labels_predicted':
      return 'No labels exceeded the classification threshold'
    case 'confidence_below_threshold':
      return 'Top predicted label score is below the abstention threshold'
    case 'confidence_meets_threshold':
      return 'Top predicted label score meets the abstention threshold'
  }
}

export function humanizeWarning(warning: InferenceWarning): string {
  switch (warning) {
    case 'empty_title':
      return 'Title is empty'
    case 'empty_body':
      return 'Body is empty'
    case 'no_labels_predicted':
      return 'No labels were predicted'
  }
}

export function formatConfidence(confidence: number | null): string {
  if (confidence === null) {
    return 'N/A'
  }
  return confidence.toFixed(3)
}

export function githubIssueUrl(repository: string, issueNumber: number): string {
  return `https://github.com/${repository}/issues/${issueNumber}`
}
