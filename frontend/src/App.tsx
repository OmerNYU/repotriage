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
import { OutcomePreview } from './components/OutcomePreview'
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

const CAPABILITIES = [
  'Predict labels',
  'Recommend abstention',
  'Retrieve similar issues',
  'Capture feedback',
] as const

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

  const showOutcomePreview = !inference.loading && !inference.result

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1 className="brand-mark">RepoTriage</h1>
          <p className="brand-tagline">Local issue intelligence for open-source maintainers.</p>
          <ul className="capability-strip" aria-label="Product capabilities">
            {CAPABILITIES.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="header-meta">
          <HealthIndicator health={health} />
        </div>
      </header>

      <main className="app-main" aria-busy={inference.loading}>
        <div
          className={
            showOutcomePreview ? 'workbench-layout workbench-layout-split' : 'workbench-layout'
          }
        >
          <IssueInputForm
            values={issueForm}
            loading={inference.loading}
            onChange={setIssueForm}
            onSubmit={handleInfer}
          />
          {showOutcomePreview && <OutcomePreview />}
        </div>

        {inference.loading && (
          <section className="panel loading-skeleton" aria-live="polite" role="status">
            <span className="sr-only">Scoring issue…</span>
            <div className="skeleton-line narrow" />
            <div className="skeleton-line medium" />
            <div className="skeleton-block" />
            <div className="skeleton-line wide" />
            <div className="skeleton-line medium" />
          </section>
        )}

        {inference.error && (
          <StatusAlert
            variant="error"
            title="Inference failed"
            message={inference.error}
            onDismiss={inference.clearError}
          />
        )}

        {inference.result && !inference.loading && (
          <>
            <InferenceResults result={inference.result} />
            <FeedbackPanel
              inferResult={inference.result}
              issueTitle={issueForm.title.trim()}
              issueBody={issueForm.body}
              issueNumber={issueNumber}
              reviewerNote={reviewerNote}
              loading={feedback.loading}
              settled={Boolean(feedback.success)}
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
