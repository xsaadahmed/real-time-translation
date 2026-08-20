'use client'

import { motion } from 'motion/react'

type Props = {
  mode: 'idle' | 'connecting' | 'listening' | 'finalizing'
  disabled?: boolean
  onToggle: () => void
}

const LABELS: Record<Props['mode'], string> = {
  idle: 'Start',
  connecting: 'Connecting',
  listening: 'Stop',
  finalizing: 'Processing',
}

export function RecordingIndicator({ mode, disabled, onToggle }: Props) {
  const active = mode === 'listening'
  const label = LABELS[mode]

  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      aria-pressed={active}
      className="group flex items-center gap-2 text-sm text-hint transition-colors hover:text-ink focus-visible:outline-none focus-visible:text-ink disabled:opacity-50 disabled:pointer-events-none"
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
      <span className="tabular-nums">{label}</span>
    </button>
  )
}
