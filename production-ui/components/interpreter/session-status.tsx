'use client'

type Props = {
  status: string
  error: string | null
  connecting: boolean
  finalizing: boolean
  finalized: boolean
  onReset: () => void
  canReset: boolean
}

export function SessionStatus({
  status,
  error,
  connecting,
  finalizing,
  finalized,
  onReset,
  canReset,
}: Props) {
  const message = error ?? status

  if (!message && !canReset) {
    return null
  }

  return (
    <div className="flex flex-col items-center gap-3 text-xs text-hint">
      {message && (
        <p
          className={`max-w-md text-center tracking-wide ${error ? 'text-destructive' : ''}`}
          aria-live="polite"
        >
          {connecting ? 'Connecting…' : finalizing ? 'Finalizing…' : message}
        </p>
      )}
      {canReset && !connecting && !finalizing && (
        <button
          type="button"
          onClick={onReset}
          className="tracking-wide underline decoration-transparent underline-offset-4 transition-colors hover:text-ink hover:decoration-current focus-visible:outline-none focus-visible:text-ink"
        >
          {finalized ? 'New session' : 'Clear'}
        </button>
      )}
    </div>
  )
}
