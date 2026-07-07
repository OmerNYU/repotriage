import { describe, expect, it } from 'vitest'
import {
  formatConfidence,
  formatScore,
  formatSimilarity,
  humanizeAbstentionReason,
  humanizeWarning,
  truncateBodyPreview,
} from './format'

describe('truncateBodyPreview', () => {
  it('returns short bodies unchanged', () => {
    expect(truncateBodyPreview('hello')).toBe('hello')
  })

  it('truncates long bodies to 200 chars', () => {
    const body = 'x'.repeat(250)
    expect(truncateBodyPreview(body)).toHaveLength(200)
  })
})

describe('formatScore', () => {
  it('formats as percentage', () => {
    expect(formatScore(0.553)).toBe('55.3%')
  })
})

describe('formatSimilarity', () => {
  it('formats as percentage', () => {
    expect(formatSimilarity(0.0903)).toBe('9.0%')
  })
})

describe('humanizeAbstentionReason', () => {
  it('maps known reasons', () => {
    expect(humanizeAbstentionReason('no_labels_predicted')).toContain('No labels')
    expect(humanizeAbstentionReason('confidence_below_threshold')).toContain('below')
    expect(humanizeAbstentionReason('confidence_meets_threshold')).toContain('meets')
  })
})

describe('humanizeWarning', () => {
  it('maps known warnings', () => {
    expect(humanizeWarning('empty_title')).toBe('Title is empty')
  })
})

describe('formatConfidence', () => {
  it('returns N/A for null', () => {
    expect(formatConfidence(null)).toBe('N/A')
  })

  it('formats numbers', () => {
    expect(formatConfidence(0.5526)).toBe('0.553')
  })
})
