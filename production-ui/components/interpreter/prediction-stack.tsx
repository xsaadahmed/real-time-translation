'use client'

import { motion } from 'motion/react'
import type { Candidate } from '@/hooks/use-interpreter-stream'

/**
 * Inline candidate stack. The committed candidate (`best`) sits in the normal
 * text flow, directly after the confirmed sentence, so the sentence keeps
 * flowing and the predictions trail its end. The two alternates are absolutely
 * positioned one line above and one line below, dimmer, so they never push the
 * baseline around.
 */
export function PredictionStack({ candidates }: { candidates: Candidate[] }) {
  const above = candidates.find((_, i) => i === 0)
  const best = candidates.find((c) => c.best) ?? candidates[1]
  const below = candidates.find((_, i) => i === 2)

  return (
    <span aria-hidden className="relative inline-block whitespace-nowrap align-baseline">
      {above && (
        <motion.span
          key={`above-${above.text}`}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 0.32, y: 0 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
          className="absolute bottom-full left-0 mb-[0.35em] block whitespace-nowrap text-prediction"
        >
          {above.text}
        </motion.span>
      )}

      <motion.span
        key={`best-${best?.text}`}
        initial={{ opacity: 0, x: -4 }}
        animate={{ opacity: 0.7, x: 0 }}
        transition={{ duration: 0.24, ease: 'easeOut' }}
        className="text-prediction"
      >
        {best?.text}
      </motion.span>

      {below && (
        <motion.span
          key={`below-${below.text}`}
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 0.32, y: 0 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
          className="absolute top-full left-0 mt-[0.35em] block whitespace-nowrap text-prediction"
        >
          {below.text}
        </motion.span>
      )}
    </span>
  )
}
