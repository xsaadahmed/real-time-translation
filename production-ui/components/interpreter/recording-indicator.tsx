'use client'

import { motion } from 'motion/react'

/**
 * A tiny click-to-toggle mic control that lives in the type, not as a big
 * floating SaaS button. A soft pulsing dot signals the listening state.
 */
export function RecordingIndicator({
  active,
  onToggle,
}: {
  active: boolean
  onToggle: () => void
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={active}
      className="group flex items-center gap-2 text-sm text-hint transition-colors hover:text-ink focus-visible:outline-none focus-visible:text-ink"
    >
      <span className="relative flex h-2 w-2 items-center justify-center">
        {active && (
          <motion.span
            className="absolute inline-flex h-2 w-2 rounded-full bg-ink"
            initial={{ opacity: 0.5, scale: 1 }}
            animate={{ opacity: 0, scale: 2.6 }}
            transition={{ duration: 1.4, repeat: Infinity, ease: 'easeOut' }}
          />
        )}
        <span
          className={
            active
              ? 'inline-flex h-2 w-2 rounded-full bg-ink'
              : 'inline-flex h-2 w-2 rounded-full border border-current'
          }
        />
      </span>
      <span className="tabular-nums">{active ? 'Listening' : 'Start'}</span>
    </button>
  )
}
