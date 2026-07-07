interface LabelCheckboxGroupProps {
  labelOrder: string[]
  selectedLabels: string[]
  onChange: (labels: string[]) => void
  disabled?: boolean
}

export function LabelCheckboxGroup({
  labelOrder,
  selectedLabels,
  onChange,
  disabled = false,
}: LabelCheckboxGroupProps) {
  const selectedSet = new Set(selectedLabels)

  const toggle = (label: string) => {
    const next = new Set(selectedSet)
    if (next.has(label)) {
      next.delete(label)
    } else {
      next.add(label)
    }
    onChange(labelOrder.filter((item) => next.has(item)))
  }

  return (
    <fieldset className="label-checkbox-group" disabled={disabled}>
      <legend>Select correct labels</legend>
      <div className="checkbox-grid">
        {labelOrder.map((label) => (
          <label key={label} className="checkbox-item">
            <input
              type="checkbox"
              checked={selectedSet.has(label)}
              disabled={disabled}
              onChange={() => toggle(label)}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}
