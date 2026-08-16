'use client'

import { AnimatePresence, motion } from 'motion/react'
import { useTypewriter } from '@/hooks/use-typewriter'
import type { Pending, Phase } from '@/hooks/use-interpreter-stream'

/** Blinking typewriter caret. Always visible while the stream is live. */
function Caret({ className = '' }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={`inline-block h-[0.95em] w-[3px] translate-y-[0.14em] animate-caret rounded-[1px] bg-ink align-baseline ${className}`}
    />
  )
}

/** For the "miss" case: the real word types out character-by-character. */
function MissWord({ target }: { target: string }) {
  const { displayed } = useTypewriter(target, 34)
  return <span className="text-ink">{displayed}</span>
}

type Props = {
  pending: Pending | null
  phase: Phase
  /** remount key so animations restart per word */
  wordKey: string | number
}

/**
 * Renders the single word currently being predicted, plus the two floating
 * alternates, and the caret.
 *
 *  - predicting: caret sits BEFORE the grey top-guess; the two alternates
 *    float above / below as opaque chips (so they never smear into wrapped
 *    lines behind them).
 *  - resolving:  the correct word commits in black and the caret jumps to the
 *    END of it (autocomplete). If the truth was an alternate, it slides in
 *    from above / below; if it was a "miss", the real word types itself out.
 */
export function ActiveWord({ pending, phase, wordKey }: Props) {
  // holding between sentences — just the blinking caret
  if (!pending) return <Caret />

  const resolving = phase === 'resolving'
  const { winner } = pending

  // during resolving, the committed word is whichever slot held the truth
  const committedWord =
    winner === 'center'
      ? pending.center
      : winner === 'above'
        ? pending.above
        : winner === 'below'
          ? pending.below
          : pending.target

  // alternates slide toward the baseline from their original position
  const slideFrom = winner === 'above' ? -40 : winner === 'below' ? 40 : 0

  return (
    <span className="relative inline-block whitespace-nowrap align-baseline">
      {!resolving && <Caret className="mr-[2px]" />}

      {/* floating alternates — only while predicting */}
      <AnimatePresence>
        {!resolving && (
          <>
            <motion.span
              key="above"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              className="absolute bottom-full left-0 mb-[0.28em] block rounded-[3px] bg-paper px-[0.12em] text-prediction"
            >
              {pending.above}
            </motion.span>
            <motion.span
              key="below"
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              className="absolute top-full left-0 mt-[0.28em] block rounded-[3px] bg-paper px-[0.12em] text-prediction"
            >
              {pending.below}
            </motion.span>
          </>
        )}
      </AnimatePresence>

      {/* the inline slot: grey top-guess, or the committed black word */}
      {resolving && winner === 'miss' ? (
        <MissWord key={`miss-${wordKey}`} target={pending.target} />
      ) : (
        <motion.span
          key={`slot-${wordKey}-${resolving ? 'r' : 'p'}`}
          initial={resolving ? { y: slideFrom, opacity: winner === 'center' ? 1 : 0.35 } : false}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.24, ease: 'easeOut' }}
          className={resolving ? 'text-ink' : 'text-prediction opacity-75'}
        >
          {resolving ? committedWord : pending.center}
        </motion.span>
      )}

      {resolving && <Caret className="ml-[2px]" />}
    </span>
  )
}
