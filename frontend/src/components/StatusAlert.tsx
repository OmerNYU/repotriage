interface StatusAlertProps {
  variant: 'error' | 'success' | 'info'
  title?: string
  message: string
  onDismiss?: () => void
}

export function StatusAlert({ variant, title, message, onDismiss }: StatusAlertProps) {
  return (
    <div className={`alert alert-${variant}`} role="alert">
      {title && <strong>{title}</strong>}
      <span>{message}</span>
      {onDismiss && (
        <button type="button" className="alert-dismiss" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      )}
    </div>
  )
}
