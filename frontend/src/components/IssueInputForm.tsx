import type { FormEvent } from 'react'
import type { IssueFormValues } from '../hooks/useInference'

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

  return (
    <section className="panel">
      <h2>Issue input</h2>
      <form className="issue-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Title</span>
          <input
            type="text"
            required
            value={values.title}
            disabled={loading}
            onChange={(e) => onChange({ ...values, title: e.target.value })}
            placeholder="BUG: loc indexing returns unexpected result"
          />
        </label>
        <label className="field">
          <span>Body</span>
          <textarea
            rows={6}
            value={values.body}
            disabled={loading}
            onChange={(e) => onChange({ ...values, body: e.target.value })}
            placeholder="Describe the issue…"
          />
        </label>
        <label className="field field-inline">
          <span>Similar issues (top_k)</span>
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
        <button type="submit" className="btn btn-primary" disabled={loading || !values.title.trim()}>
          {loading ? 'Scoring…' : 'Score issue'}
        </button>
      </form>
    </section>
  )
}
