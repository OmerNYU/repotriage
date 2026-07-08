import { fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { IssueInputForm } from '../components/IssueInputForm'
import type { IssueFormValues } from '../hooks/useInference'
import { getSampleById, toFormValues } from '../lib/samples'

const emptyValues: IssueFormValues = {
  title: '',
  body: '',
  top_k: 5,
}

function ControlledForm({
  onSubmit = vi.fn(),
}: {
  onSubmit?: (values: IssueFormValues) => void
}) {
  const [values, setValues] = useState<IssueFormValues>(emptyValues)
  return (
    <IssueInputForm
      values={values}
      loading={false}
      onChange={setValues}
      onSubmit={onSubmit}
    />
  )
}

describe('IssueInputForm samples', () => {
  it('fills title, body, and top_k without auto-submitting', () => {
    const onChange = vi.fn()
    const onSubmit = vi.fn()

    render(
      <IssueInputForm
        values={emptyValues}
        loading={false}
        onChange={onChange}
        onSubmit={onSubmit}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /bug · indexing/i }))

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith({
      title: 'BUG: loc indexing returns unexpected result',
      body: 'When using .loc with a list indexer, result dtype is wrong.',
      top_k: 5,
    })
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('updates controlled inputs with title, body, and similar-issues count', () => {
    const onSubmit = vi.fn()
    render(<ControlledForm onSubmit={onSubmit} />)

    const scoreButton = screen.getByRole('button', { name: /score issue/i })
    expect(scoreButton).toBeDisabled()
    expect(scoreButton).not.toHaveClass('btn-score-ready')

    fireEvent.click(screen.getByRole('button', { name: /bug · indexing/i }))

    expect(screen.getByLabelText(/^title$/i)).toHaveValue(
      'BUG: loc indexing returns unexpected result',
    )
    expect(screen.getByLabelText(/body/i)).toHaveValue(
      'When using .loc with a list indexer, result dtype is wrong.',
    )
    expect(screen.getByLabelText(/number of similar issues/i)).toHaveValue(5)
    expect(scoreButton).toBeEnabled()
    expect(scoreButton).toHaveClass('btn-score-ready')
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('does not treat placeholder text as a filled title', () => {
    render(
      <IssueInputForm
        values={emptyValues}
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    const title = screen.getByLabelText(/^title$/i)
    expect(title).toHaveValue('')
    expect(title).toHaveAttribute('placeholder', 'Short issue title')
    expect(screen.getByRole('button', { name: /score issue/i })).toBeDisabled()
  })

  it('enables Score issue once title is filled even if body is empty', () => {
    render(
      <IssueInputForm
        values={{ title: 'BUG: only title', body: '', top_k: 5 }}
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: /score issue/i })).toBeEnabled()
  })

  it('keeps Score issue disabled when title is empty', () => {
    render(
      <IssueInputForm
        values={emptyValues}
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: /score issue/i })).toBeDisabled()
  })

  it('fills the docs sample fields explicitly', () => {
    const onChange = vi.fn()

    render(
      <IssueInputForm
        values={emptyValues}
        loading={false}
        onChange={onChange}
        onSubmit={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /docs sample/i }))

    expect(onChange).toHaveBeenCalledWith(toFormValues(getSampleById('docs-enhancement')!))
  })

  it('uses a human label for the similar-issues field', () => {
    render(
      <IssueInputForm
        values={emptyValues}
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(screen.getByLabelText(/number of similar issues/i)).toBeInTheDocument()
    expect(screen.queryByText(/top_k/i)).not.toBeInTheDocument()
  })
})
