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
  settled?: boolean
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
  settled = false,
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
  const actionsDisabled = loading || settled || !issueNumberValid

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
  const correctionReady = canSubmitCorrection(
    labelOrder.filter((label) => selectedLabels.includes(label)),
    predictedLabels,
  )

  return (
    <section
      className={`panel panel-feedback${settled ? ' panel-settled' : ''}`}
      aria-disabled={settled || undefined}
    >
      <p className="panel-eyebrow">Maintainer review</p>
      <h2>Record feedback</h2>
      <p className="panel-lead panel-lead-compact">
        Accept, reject, or correct labels for this run. Issue numbers are demo tracking IDs only.
      </p>

      <div className="feedback-fields">
        <label className="field field-issue-number">
          <span>Issue number</span>
          <input
            type="number"
            min={1}
            value={issueNumber}
            disabled={loading || settled}
            onChange={(e) => {
              const raw = e.target.value
              onIssueNumberChange(raw === '' ? '' : Number(raw))
            }}
            placeholder="12345"
          />
          <span className="field-hint">Not validated against GitHub.</span>
        </label>

        <label className="field">
          <span>Reviewer note (optional)</span>
          <textarea
            rows={2}
            value={reviewerNote}
            disabled={loading || settled}
            maxLength={4000}
            onChange={(e) => onReviewerNoteChange(e.target.value)}
            placeholder="Optional note for this review…"
          />
        </label>
      </div>

      <div className="action-row action-row-feedback">
        <button
          type="button"
          className="btn btn-success btn-compact"
          disabled={actionsDisabled}
          onClick={handleAccept}
        >
          Accept
        </button>
        <button
          type="button"
          className="btn btn-danger btn-compact"
          disabled={rejectDisabled}
          title={
            predictedLabels.length === 0
              ? 'Nothing to reject — no labels were predicted.'
              : undefined
          }
          onClick={handleReject}
        >
          Reject
        </button>
        <button
          type="button"
          className="btn btn-neutral btn-compact"
          disabled={actionsDisabled}
          onClick={() => {
            setSelectedLabels(predictedLabels)
            setCorrecting(true)
          }}
        >
          Correct labels
        </button>
      </div>

      {correcting && !settled && (
        <div className="correction-panel">
          <p className="correction-title">Correct predicted labels</p>
          <div className="correction-box">
            <LabelCheckboxGroup
              labelOrder={labelOrder}
              selectedLabels={selectedLabels}
              disabled={loading}
              onChange={setSelectedLabels}
            />
          </div>
          <div className="action-row action-row-feedback">
            <button
              type="button"
              className="btn btn-primary btn-compact"
              disabled={actionsDisabled || !correctionReady}
              onClick={handleCorrectSubmit}
            >
              Submit correction
            </button>
            <button
              type="button"
              className="btn btn-neutral btn-compact"
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
