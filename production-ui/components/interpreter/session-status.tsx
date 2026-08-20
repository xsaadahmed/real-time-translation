'use client'

type Props = {
  status: string
  error: string | null
  connecting: boolean
  reconnecting: boolean
  finalizing: boolean
  finalized: boolean
  phase: string
  durationSec: number
  onReset: () => void
  onRetry: () => void
  canReset: boolean
  serverReady: boolean | null
}

export function SessionStatus({
  status,
  error,
  connecting,
  reconnecting,
  finalizing,
  finalized,
  phase,
  durationSec,
  onReset,
  onRetry,
  canReset,
  serverReady,
}: Props) {
  const message =
    error ??
    (connecting
      ? 'Connecting…'
      : reconnecting
        ? status || 'Reconnecting…'
        : finalizing
          ? status || 'Finalizing…'
          : status)

  const showEmptyHint = phase === 'empty' || (finalized && !error && !status)
  const showIdleHint =
    !message && !canReset && serverReady === false
      ? 'Waiting for translation server…'
      : !message && !canReset && serverReady === null
        ? ''
        : null

  if (!message && !canReset && !showIdleHint && !showEmptyHint) {
    return null
  }

  return (
    <div className="flex flex-col items-center gap-3 text-xs text-hint">
      {serverReady === false && !error && (
        <p className="max-w-md text-center tracking-wide" aria-live="polite">
          Server is loading models — Start will wait until ready.
        </p>
      )}
      {message && (
        <p
          className={`max-w-md text-center tracking-wide ${error ? 'text-destructive' : ''}`}
          aria-live="polite"
        >
          {message}
          {finalizing && durationSec > 0 ? ` · ${durationSec.toFixed(1)}s audio` : ''}
        </p>
      )}
      {finalizing && (
        <div
          className="h-0.5 w-32 overflow-hidden rounded-full bg-border"
          role="progressbar"
          aria-label="Finalizing"
        >
          <div className="h-full w-1/2 animate-pulse rounded-full bg-hint" />
        </div>
      )}
      {error && (
        <button
          type="button"
          onClick={onRetry}
          className="tracking-wide underline decoration-transparent underline-offset-4 transition-colors hover:text-ink hover:decoration-current focus-visible:outline-none focus-visible:text-ink"
        >
          Try again
        </button>
      )}
      {canReset && !connecting && !reconnecting && !finalizing && !error && (
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
