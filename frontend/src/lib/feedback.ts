import type {
  ArtifactReferences,
  FeedbackRequest,
  ReviewAction,
} from '../api/types'
import { truncateBodyPreview } from './format'

export interface FeedbackContext {
  repository: string
  issueNumber: number
  issueTitle: string
  issueBody: string
  predictedLabels: string[]
  labelOrder: string[]
  artifacts: ArtifactReferences
  reviewerNote?: string | null
}

export function labelsInOrder(selected: Set<string>, labelOrder: string[]): string[] {
  return labelOrder.filter((label) => selected.has(label))
}

export function labelsEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) {
    return false
  }
  return a.every((label, index) => label === b[index])
}

export function canSubmitCorrection(
  selectedLabels: string[],
  predictedLabels: string[],
): boolean {
  return !labelsEqual(selectedLabels, predictedLabels)
}

export function buildFeedbackRequest(
  context: FeedbackContext,
  reviewAction: ReviewAction,
  acceptedLabels: string[],
  rejectedLabels: string[] = [],
): FeedbackRequest {
  return {
    feedback_schema_version: '1',
    repository: context.repository,
    issue_number: context.issueNumber,
    issue_title: context.issueTitle,
    issue_body_preview: truncateBodyPreview(context.issueBody),
    predicted_labels: context.predictedLabels,
    accepted_labels: acceptedLabels,
    rejected_labels: rejectedLabels,
    review_action: reviewAction,
    reviewer_note: context.reviewerNote?.trim() || null,
    inference_artifacts: { ...context.artifacts },
  }
}

export function buildAcceptedFeedback(context: FeedbackContext): FeedbackRequest {
  return buildFeedbackRequest(
    context,
    'accepted',
    context.predictedLabels,
    [],
  )
}

export function buildRejectedFeedback(context: FeedbackContext): FeedbackRequest {
  return buildFeedbackRequest(context, 'rejected', [], context.predictedLabels)
}

export function buildCorrectedFeedback(
  context: FeedbackContext,
  selectedLabels: string[],
): FeedbackRequest {
  const acceptedLabels = labelsInOrder(new Set(selectedLabels), context.labelOrder)
  const selectedSet = new Set(selectedLabels)
  const rejectedLabels = context.predictedLabels.filter((label) => !selectedSet.has(label))
  return buildFeedbackRequest(context, 'corrected', acceptedLabels, rejectedLabels)
}

export function isValidIssueNumber(value: number): boolean {
  return Number.isInteger(value) && value > 0
}
