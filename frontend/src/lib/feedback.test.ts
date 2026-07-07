import { describe, expect, it } from 'vitest'
import {
  buildAcceptedFeedback,
  buildCorrectedFeedback,
  buildRejectedFeedback,
  canSubmitCorrection,
  isValidIssueNumber,
  labelsInOrder,
} from './feedback'
import type { FeedbackContext } from './feedback'

const baseContext: FeedbackContext = {
  repository: 'pandas-dev/pandas',
  issueNumber: 12345,
  issueTitle: 'BUG: example',
  issueBody: 'Body text',
  predictedLabels: ['Indexing'],
  labelOrder: ['Bug', 'Indexing', 'Docs'],
  artifacts: {
    model_dataset_id: 'md1',
    baseline_run_id: 'bl1',
    threshold_policy_id: 'tp1',
    abstention_policy_id: 'ap1',
    retrieval_run_id: 'rb1',
  },
}

describe('buildAcceptedFeedback', () => {
  it('sets accepted labels equal to predicted', () => {
    const request = buildAcceptedFeedback(baseContext)
    expect(request.review_action).toBe('accepted')
    expect(request.accepted_labels).toEqual(['Indexing'])
    expect(request.rejected_labels).toEqual([])
  })
})

describe('buildRejectedFeedback', () => {
  it('rejects all predicted labels', () => {
    const request = buildRejectedFeedback(baseContext)
    expect(request.review_action).toBe('rejected')
    expect(request.accepted_labels).toEqual([])
    expect(request.rejected_labels).toEqual(['Indexing'])
  })
})

describe('buildCorrectedFeedback', () => {
  it('orders accepted labels by label_order', () => {
    const request = buildCorrectedFeedback(baseContext, ['Docs', 'Bug'])
    expect(request.review_action).toBe('corrected')
    expect(request.accepted_labels).toEqual(['Bug', 'Docs'])
    expect(request.rejected_labels).toEqual(['Indexing'])
  })
})

describe('canSubmitCorrection', () => {
  it('returns false when selection matches predicted', () => {
    expect(canSubmitCorrection(['Indexing'], ['Indexing'])).toBe(false)
  })

  it('returns true when selection differs', () => {
    expect(canSubmitCorrection(['Bug', 'Indexing'], ['Indexing'])).toBe(true)
  })
})

describe('labelsInOrder', () => {
  it('filters and orders selected labels', () => {
    expect(labelsInOrder(new Set(['Docs', 'Bug']), baseContext.labelOrder)).toEqual([
      'Bug',
      'Docs',
    ])
  })
})

describe('isValidIssueNumber', () => {
  it('accepts positive integers', () => {
    expect(isValidIssueNumber(1)).toBe(true)
    expect(isValidIssueNumber(12345)).toBe(true)
  })

  it('rejects invalid values', () => {
    expect(isValidIssueNumber(0)).toBe(false)
    expect(isValidIssueNumber(-1)).toBe(false)
    expect(isValidIssueNumber(1.5)).toBe(false)
  })
})
