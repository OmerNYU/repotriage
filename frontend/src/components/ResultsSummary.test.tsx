import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { InferenceResponse } from '../api/types'
import { ResultsSummary } from './ResultsSummary'

function stubResult(overrides: Partial<InferenceResponse> = {}): InferenceResponse {
  return {
    schema_version: '1',
    repository: 'pandas-dev/pandas',
    generated_at: '2026-01-01T00:00:00Z',
    input: {
      title: 't',
      body_preview: 'b',
      feature_text_sha256: 'a'.repeat(64),
      text_representation_version: '1',
    },
    classification: {
      label_order: ['Bug', 'Indexing'],
      scores: [
        { label: 'Bug', score: 0.2 },
        { label: 'Indexing', score: 0.81 },
      ],
      threshold: 0.5,
      threshold_basis_points: 5000,
      predicted_labels: [{ label: 'Indexing', score: 0.81 }],
    },
    abstention: {
      confidence_method: 'max_predicted_label_score',
      confidence: 0.81,
      threshold: 0.7,
      threshold_basis_points: 7000,
      should_abstain: false,
      reason: 'confidence_meets_threshold',
    },
    retrieval: {
      method: 'tfidf_cosine',
      top_k: 5,
      similar_issues: [
        {
          rank: 1,
          issue_id: 1,
          issue_number: 10,
          similarity: 0.9,
          neighbor_selected_labels: ['Indexing'],
          predicted_label_overlap: ['Indexing'],
        },
      ],
    },
    artifacts: {
      model_dataset_id: 'm',
      baseline_run_id: 'b',
      threshold_policy_id: 't',
      abstention_policy_id: 'a',
      retrieval_run_id: 'r',
    },
    reproducibility: {
      inference_config_path: 'c',
      model_semantic_sha256: 'a'.repeat(64),
      index_semantic_sha256: 'b'.repeat(64),
      baseline_experiment_sha256: 'c'.repeat(64),
      numerical_environment_sha256: 'd'.repeat(64),
      serialization_security_warning: null,
    },
    warnings: [],
    ...overrides,
  }
}

describe('ResultsSummary', () => {
  it('renders real response fields only', () => {
    render(<ResultsSummary result={stubResult()} />)

    expect(screen.getByText('Proceed')).toBeInTheDocument()
    expect(screen.getByText('Indexing')).toBeInTheDocument()
    expect(screen.getByText('0.810')).toBeInTheDocument()
    expect(screen.getByText('1')).toBeInTheDocument()
  })

  it('shows Abstain and No labels when appropriate', () => {
    render(
      <ResultsSummary
        result={stubResult({
          classification: {
            label_order: ['Bug'],
            scores: [{ label: 'Bug', score: 0.1 }],
            threshold: 0.5,
            threshold_basis_points: 5000,
            predicted_labels: [],
          },
          abstention: {
            confidence_method: 'max_predicted_label_score',
            confidence: null,
            threshold: 0.7,
            threshold_basis_points: 7000,
            should_abstain: true,
            reason: 'no_labels_predicted',
          },
          retrieval: { method: 'tfidf_cosine', top_k: 5, similar_issues: [] },
        })}
      />,
    )

    expect(screen.getByText('Abstain')).toBeInTheDocument()
    expect(screen.getByText('No labels')).toBeInTheDocument()
    expect(screen.getByText('N/A')).toBeInTheDocument()
    expect(screen.getByText('0')).toBeInTheDocument()
  })
})
