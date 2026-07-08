import { describe, expect, it } from 'vitest'
import { SAMPLE_ISSUES, getSampleById, toFormValues } from './samples'

describe('SAMPLE_ISSUES', () => {
  it('includes the canonical docs sample', () => {
    const canonical = getSampleById('loc-indexing')
    expect(canonical).toBeDefined()
    expect(canonical?.values.title).toBe('BUG: loc indexing returns unexpected result')
    expect(canonical?.values.body).toBe(
      'When using .loc with a list indexer, result dtype is wrong.',
    )
    expect(canonical?.values.top_k).toBe(5)
  })

  it('includes a secondary documentation sample', () => {
    const secondary = getSampleById('docs-enhancement')
    expect(secondary).toBeDefined()
    expect(secondary?.label).toBe('Docs sample')
    expect(secondary?.values.title.toLowerCase()).toContain('doc')
    expect(secondary?.values.top_k).toBeGreaterThanOrEqual(1)
  })

  it('exposes exactly two presets with unique ids', () => {
    expect(SAMPLE_ISSUES).toHaveLength(2)
    const ids = SAMPLE_ISSUES.map((sample) => sample.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const sample of SAMPLE_ISSUES) {
      expect(sample.label.length).toBeGreaterThan(0)
      expect(sample.values.title.trim().length).toBeGreaterThan(0)
      expect(sample.values.body.trim().length).toBeGreaterThan(0)
    }
  })

  it('toFormValues copies title, body, and top_k explicitly', () => {
    const sample = getSampleById('loc-indexing')!
    const values = toFormValues(sample)
    expect(values).toEqual({
      title: sample.values.title,
      body: sample.values.body,
      top_k: 5,
    })
    expect(values).not.toBe(sample.values)
  })
})
