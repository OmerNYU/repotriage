import type { FormEvent } from 'react'
import type { IssueFormValues } from '../hooks/useInference'
import { SAMPLE_ISSUES, toFormValues } from '../lib/samples'

interface IssueInputFormProps {
  values: IssueFormValues
  loading: boolean
  onChange: (values: IssueFormValues) => void
  onSubmit: (values: IssueFormValues) => void
}

export function IssueInputForm({
  values,
  loading,
  onChange,
  onSubmit,
}: IssueInputFormProps) {
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    onSubmit(values)
  }

  const canScore = values.title.trim().length > 0

  return (
    <section className="panel panel-workbench">
      <div className="workbench-header">
        <p className="panel-eyebrow">Maintainer workflow</p>
        <h2>Issue workbench</h2>
        <p className="panel-lead">
          Paste an issue or load a sample, then score. Nothing is written to GitHub.
        </p>
      </div>

      <div className="sample-row" role="group" aria-label="Sample issues">
        <span className="sample-row-label">Try a sample</span>
        {SAMPLE_ISSUES.map((sample) => (
          <button
            key={sample.id}
            type="button"
            className="sample-btn"
            disabled={loading}
            title={sample.description}
            onClick={() => onChange(toFormValues(sample))}
          >
            {sample.label}
          </button>
        ))}
      </div>

      <form className="issue-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Title</span>
          <input
            type="text"
            required
            value={values.title}
            disabled={loading}
            onChange={(e) => onChange({ ...values, title: e.target.value })}
            placeholder="Short issue title"
          />
        </label>
        <label className="field">
          <span>
            Body <span className="field-hint-inline">(optional)</span>
          </span>
          <textarea
            rows={3}
            value={values.body}
            disabled={loading}
            onChange={(e) => onChange({ ...values, body: e.target.value })}
            placeholder="Describe the issue…"
          />
        </label>
        <label className="field field-inline">
          <span>Number of similar issues</span>
          <input
            type="number"
            min={1}
            value={values.top_k}
            disabled={loading}
            onChange={(e) =>
              onChange({ ...values, top_k: Math.max(1, Number(e.target.value) || 1) })
            }
          />
        </label>
        <div className="workbench-actions">
          <button
            type="submit"
            className={`btn btn-primary btn-score${canScore && !loading ? ' btn-score-ready' : ''}`}
            disabled={loading || !canScore}
          >
            {loading ? 'Scoring…' : 'Score issue'}
          </button>
        </div>
      </form>
    </section>
  )
}
