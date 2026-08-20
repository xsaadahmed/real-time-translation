'use client'

import { useEffect, useRef } from 'react'

type Props = {
  verified: string
  provisional: string
  showCaret: boolean
}

function Caret() {
  return (
    <span
      aria-hidden
      className="inline-block h-[0.95em] w-[3px] translate-y-[0.14em] animate-caret rounded-[1px] bg-ink align-baseline ml-[2px]"
    />
  )
}

/**
 * English translation: black = verified (committed), grey = provisional preview.
 */
export function TranslationText({ verified, provisional, showCaret }: Props) {
  const endRef = useRef<HTMLDivElement>(null)
  const hasContent = verified || provisional || showCaret

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [verified, provisional])

  const textClass =
    'text-balance text-left text-[1.9rem] font-semibold leading-[1.35] tracking-[-0.01em] md:text-[2.4rem] lg:text-[2.75rem]'

  if (!hasContent) {
    return (
      <div className="flex w-full max-w-4xl flex-col">
        <p className={`${textClass} text-prediction opacity-40`}>
          English translation appears here.
        </p>
        <div ref={endRef} />
      </div>
    )
  }

  return (
    <div aria-live="polite" className="flex w-full max-w-4xl flex-col">
      <p className={`${textClass} my-[0.5em]`}>
        {verified && <span className="text-ink">{verified}</span>}
        {verified && provisional ? ' ' : ''}
        {provisional && <span className="text-prediction">{provisional}</span>}
        {showCaret && <Caret />}
      </p>
      <div ref={endRef} />
    </div>
  )
}
