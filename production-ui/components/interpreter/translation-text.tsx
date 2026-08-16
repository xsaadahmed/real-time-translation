'use client'

import { useEffect, useRef } from 'react'
import type { Pending, Phase } from '@/hooks/use-interpreter-stream'
import { ActiveWord } from './active-word'

type Props = {
  history: string[]
  committed: string
  pending: Pending | null
  phase: Phase
  running: boolean
  done: boolean
}

/**
 * A running transcript. Completed sentences stack as history; the active line
 * flows left-to-right, committing one word at a time. The word currently being
 * predicted trails inline at the end with two floating alternates, and a
 * blinking caret marks the write head.
 */
export function TranslationText({ history, committed, pending, phase, running }: Props) {
  const endRef = useRef<HTMLDivElement>(null)
  const wordKey = `${history.length}-${committed.length}`

  // keep the newest line in view as the transcript grows
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [history.length, committed])

  const textClass =
    'text-balance text-left text-[1.9rem] font-semibold leading-[1.35] tracking-[-0.01em] text-ink md:text-[2.4rem] lg:text-[2.75rem]'

  const showActive = running || committed || pending

  return (
    <div aria-live="polite" className="flex w-full max-w-4xl flex-col gap-y-[1.6em]">
      {history.map((line, i) => (
        <p key={i} className={textClass}>
          {line}
        </p>
      ))}

      {showActive && (
        // generous top/bottom room so the floating alternates always clear neighbors
        <p className={`${textClass} my-[1.5em]`}>
          {committed}
          {committed && (pending || running) ? ' ' : ''}
          <ActiveWord pending={pending} phase={phase} wordKey={wordKey} />
        </p>
      )}

      <div ref={endRef} />
    </div>
  )
}
