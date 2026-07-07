import { useState } from 'react'
import { HealthIndicator } from './components/HealthIndicator'
import {
  FeedbackPanel,
  buildAcceptedFeedback,
  buildCorrectedFeedback,
  buildRejectedFeedback,
} from './components/FeedbackPanel'
import { InferenceResults } from './components/InferenceResults'
import { IssueInputForm } from './components/IssueInputForm'
import { StatusAlert } from './components/StatusAlert'
import { useFeedback } from './hooks/useFeedback'
import { useHealth } from './hooks/useHealth'
import type { IssueFormValues } from './hooks/useInference'
import { useInference } from './hooks/useInference'
import './App.css'

const DEFAULT_ISSUE: IssueFormValues = {
  title: '',
  body: '',
  top_k: 5,
}

function App() {
  const health = useHealth()
  const inference = useInference()
  const feedback = useFeedback()

  const [issueForm, setIssueForm] = useState<IssueFormValues>(DEFAULT_ISSUE)
  const [issueNumber, setIssueNumber] = useState<number | ''>('')
  const [reviewerNote, setReviewerNote] = useState('')

  const handleInfer = async (values: IssueFormValues) => {
    feedback.reset()
    await inference.submit(values)
  }

  const submitFeedback = async (request: ReturnType<typeof buildAcceptedFeedback>) => {
    await feedback.submit(request)
  }

  const handleReviewAnother = () => {
    inference.reset()
    feedback.reset()
    setIssueForm(DEFAULT_ISSUE)
    setIssueNumber('')
    setReviewerNote('')
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>RepoTriage Maintainer Review</h1>
          <p className="subtitle">Local demo for issue scoring and maintainer feedback.</p>
        </div>
        <HealthIndicator health={health} />
      </header>

      <main className="app-main">
        <IssueInputForm
          values={issueForm}
          loading={inference.loading}
          onChange={setIssueForm}
          onSubmit={handleInfer}
        />

        {inference.error && (
          <StatusAlert variant="error" title="Inference failed" message={inference.error} />
        )}

        {inference.result && (
          <>
            <InferenceResults result={inference.result} />
            <FeedbackPanel
              inferResult={inference.result}
              issueTitle={issueForm.title.trim()}
              issueBody={issueForm.body}
              issueNumber={issueNumber}
              reviewerNote={reviewerNote}
              loading={feedback.loading}
              onIssueNumberChange={setIssueNumber}
              onReviewerNoteChange={setReviewerNote}
              onAccept={(context) => void submitFeedback(buildAcceptedFeedback(context))}
              onReject={(context) => void submitFeedback(buildRejectedFeedback(context))}
              onCorrect={(context, selectedLabels) =>
                void submitFeedback(buildCorrectedFeedback(context, selectedLabels))
              }
            />
          </>
        )}

        {feedback.loading && (
          <StatusAlert variant="info" message="Submitting feedback…" />
        )}

        {feedback.error && (
          <StatusAlert
            variant="error"
            title="Feedback failed"
            message={feedback.error}
            onDismiss={feedback.reset}
          />
        )}

        {feedback.success && (
          <div className="success-panel">
            <StatusAlert
              variant="success"
              title="Feedback stored"
              message={`ID ${feedback.success.feedback_id} · ${new Date(feedback.success.created_at).toLocaleString()}`}
            />
            <button type="button" className="btn" onClick={handleReviewAnother}>
              Review another issue
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
