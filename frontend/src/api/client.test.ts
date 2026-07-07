import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getHealth, postFeedback, postInfer } from './client'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('api client', () => {
  it('getHealth returns parsed JSON', async () => {
    const payload = {
      status: 'ok',
      schema_version: '1',
      repository: 'pandas-dev/pandas',
      inference_config_path: 'configs/test.json',
    }
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => payload,
      }),
    )

    await expect(getHealth()).resolves.toEqual(payload)
  })

  it('throws ApiError with detail from 422', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        json: async () => ({ detail: 'title is required' }),
      }),
    )

    await expect(postInfer({ title: '' })).rejects.toEqual(
      new ApiError(422, 'title is required'),
    )
  })

  it('postFeedback posts JSON body', async () => {
    const response = {
      feedback_id: 'abc',
      created_at: '2026-01-01T00:00:00Z',
      status: 'stored',
    }
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => response,
    })
    vi.stubGlobal('fetch', fetchMock)

    const body = {
      repository: 'pandas-dev/pandas',
      issue_number: 1,
      issue_title: 't',
      predicted_labels: ['Bug'],
      accepted_labels: ['Bug'],
      review_action: 'accepted' as const,
      inference_artifacts: {
        model_dataset_id: 'md1',
        baseline_run_id: 'bl1',
        threshold_policy_id: 'tp1',
        abstention_policy_id: 'ap1',
        retrieval_run_id: 'rb1',
      },
    }

    await expect(postFeedback(body)).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/feedback',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }),
    )
  })
})
