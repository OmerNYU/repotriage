const OUTCOMES = [
  {
    label: 'Predicted labels',
    detail: 'Scores above the policy threshold.',
  },
  {
    label: 'Abstention',
    detail: 'Act or hold based on confidence.',
  },
  {
    label: 'Similar issues',
    detail: 'Nearest historical neighbors.',
  },
  {
    label: 'Feedback',
    detail: 'Accept, reject, or correct.',
  },
] as const

export function OutcomePreview() {
  return (
    <aside className="panel outcome-preview" aria-label="What scoring produces">
      <p className="panel-eyebrow">What you get</p>
      <h2 className="outcome-preview-title">After you score</h2>
      <ul className="outcome-tiles">
        {OUTCOMES.map((item) => (
          <li key={item.label} className="outcome-tile">
            <strong className="outcome-tile-label">{item.label}</strong>
            <span className="outcome-tile-detail">{item.detail}</span>
          </li>
        ))}
      </ul>
    </aside>
  )
}
