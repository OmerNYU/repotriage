import type { IssueFormValues } from '../hooks/useInference'

export interface SampleIssue {
  id: string
  label: string
  description: string
  values: IssueFormValues
}

/** Hardcoded presets for demos — fill form fields only; do not auto-submit. */
export const SAMPLE_ISSUES: SampleIssue[] = [
  {
    id: 'loc-indexing',
    label: 'Bug · indexing',
    description: 'Canonical demo sample from docs.',
    values: {
      title: 'BUG: loc indexing returns unexpected result',
      body: 'When using .loc with a list indexer, result dtype is wrong.',
      top_k: 5,
    },
  },
  {
    id: 'docs-enhancement',
    label: 'Docs sample',
    description: 'Secondary sample with documentation-style wording.',
    values: {
      title: 'DOC: clarify MultiIndex droplevel examples',
      body:
        'The MultiIndex droplevel examples in the user guide leave out the common case of dropping by name rather than level number. A short example would help.',
      top_k: 5,
    },
  },
]

export function getSampleById(id: string): SampleIssue | undefined {
  return SAMPLE_ISSUES.find((sample) => sample.id === id)
}

/** Explicit field copy so controlled inputs always receive title, body, and top_k. */
export function toFormValues(sample: SampleIssue): IssueFormValues {
  return {
    title: sample.values.title,
    body: sample.values.body,
    top_k: sample.values.top_k,
  }
}
