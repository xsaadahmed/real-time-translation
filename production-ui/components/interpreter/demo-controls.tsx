'use client'

/**
 * Minimal, near-invisible demo affordance. Restart is only offered once
 * the stream has been started so the resting screen stays almost empty.
 */
export function DemoControls({
  running,
  onRestart,
}: {
  running: boolean
  onRestart: () => void
}) {
  return (
    <div className="flex items-center gap-4 text-xs text-prediction">
      <span className="uppercase tracking-[0.2em]">Demo</span>
      <button
        type="button"
        onClick={onRestart}
        className="tracking-wide underline decoration-transparent underline-offset-4 transition-colors hover:text-hint hover:decoration-current focus-visible:outline-none focus-visible:text-hint"
      >
        Restart
      </button>
    </div>
  )
}
