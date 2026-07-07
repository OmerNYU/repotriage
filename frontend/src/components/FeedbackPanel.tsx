import { useEffect, useState } from 'react'
import type { InferenceResponse } from '../api/types'
import type { FeedbackContext } from '../lib/feedback'
import {
  buildAcceptedFeedback,
  buildCorrectedFeedback,
  buildRejectedFeedback,
  canSubmitCorrection,
  isValidIssueNumber,
} from '../lib/feedback'
import { LabelCheckboxGroup } from './LabelCheckboxGroup'

interface FeedbackPanelProps {
  inferResult: InferenceResponse
  issueTitle: string
  issueBody: string
  issueNumber: number | ''
  reviewerNote: string
  loading: boolean
  onIssueNumberChange: (value: number | '') => void
  onReviewerNoteChange: (value: string) => void
  onAccept: (context: FeedbackContext) => void
  onReject: (context: FeedbackContext) => void
  onCorrect: (context: FeedbackContext, selectedLabels: string[]) => void
}

export function FeedbackPanel({
  inferResult,
  issueTitle,
  issueBody,
  issueNumber,
  reviewerNote,
  loading,
  onIssueNumberChange,
  onReviewerNoteChange,
  onAccept,
  onReject,
  onCorrect,
}: FeedbackPanelProps) {
  const predictedLabels = inferResult.classification.predicted_labels.map((item) => item.label)
  const labelOrder = inferResult.classification.label_order
  const [correcting, setCorrecting] = useState(false)
  const [selectedLabels, setSelectedLabels] = useState<string[]>(predictedLabels)

  useEffect(() => {
    setCorrecting(false)
    setSelectedLabels(predictedLabels)
  }, [inferResult.generated_at, predictedLabels.join('|')])

  const issueNumberValid = typeof issueNumber === 'number' && isValidIssueNumber(issueNumber)
  const actionsDisabled = loading || !issueNumberValid

  const baseContext = (): FeedbackContext | null => {
    if (!issueNumberValid) {
      return null
    }
    return {
      repository: inferResult.repository,
      issueNumber,
      issueTitle,
      issueBody,
      predictedLabels,
      labelOrder,
      artifacts: inferResult.artifacts,
      reviewerNote,
    }
  }

  const handleAccept = () => {
    const context = baseContext()
    if (context) {
      onAccept(context)
    }
  }

  const handleReject = () => {
    const context = baseContext()
    if (context) {
      onReject(context)
    }
  }

  const handleCorrectSubmit = () => {
    const context = baseContext()
    if (context && canSubmitCorrection(selectedLabels, predictedLabels)) {
      onCorrect(context, selectedLabels)
      setCorrecting(false)
    }
  }

  const rejectDisabled = actionsDisabled || predictedLabels.length === 0
  const correctionReady =
    canSubmitCorrection(
      labelOrder.filter((label) => selectedLabels.includes(label)),
      predictedLabels,
    )

  return (
    <section className="panel">
      <h2>Maintainer review</h2>

      <label className="field">
        <span>Issue number</span>
        <input
          type="number"
          min={1}
          value={issueNumber}
          disabled={loading}
          onChange={(e) => {
            const raw = e.target.value
            onIssueNumberChange(raw === '' ? '' : Number(raw))
          }}
          placeholder="12345"
        />
        <span className="field-hint">Demo tracking ID — not validated against GitHub.</span>
      </label>

      <label className="field">
        <span>Reviewer note (optional)</span>
        <textarea
          rows={3}
          value={reviewerNote}
          disabled={loading}
          maxLength={4000}
          onChange={(e) => onReviewerNoteChange(e.target.value)}
          placeholder="Optional note for this review…"
        />
      </label>

      <div className="action-row">
        <button
          type="button"
          className="btn btn-success"
          disabled={actionsDisabled}
          onClick={handleAccept}
        >
          Accept prediction
        </button>
        <button
          type="button"
          className="btn btn-danger"
          disabled={rejectDisabled}
          title={
            predictedLabels.length === 0
              ? 'Nothing to reject — no labels were predicted.'
              : undefined
          }
          onClick={handleReject}
        >
          Reject prediction
        </button>
        <button
          type="button"
          className="btn"
          disabled={actionsDisabled}
          onClick={() => {
            setSelectedLabels(predictedLabels)
            setCorrecting(true)
          }}
        >
          Correct labels
        </button>
      </div>

      {correcting && (
        <div className="correction-panel">
          <LabelCheckboxGroup
            labelOrder={labelOrder}
            selectedLabels={selectedLabels}
            disabled={loading}
            onChange={setSelectedLabels}
          />
          <div className="action-row">
            <button
              type="button"
              className="btn btn-primary"
              disabled={actionsDisabled || !correctionReady}
              onClick={handleCorrectSubmit}
            >
              Submit correction
            </button>
            <button
              type="button"
              className="btn"
              disabled={loading}
              onClick={() => setCorrecting(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </section>
  )
}

export {
  buildAcceptedFeedback,
  buildCorrectedFeedback,
  buildRejectedFeedback,
}
