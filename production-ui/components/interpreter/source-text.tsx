'use client'

type Props = {
  verified: string
  provisional: string
}

export function SourceText({ verified, provisional }: Props) {
  const hasContent = verified || provisional

  return (
    <div
      dir="rtl"
      lang="ar"
      aria-label="Arabic source speech"
      className="font-arabic min-h-[1.6em] text-center text-lg leading-relaxed md:text-xl"
    >
      {hasContent ? (
        <p>
          {verified && <span className="text-hint">{verified}</span>}
          {verified && provisional ? ' ' : ''}
          {provisional && <span className="text-prediction">{provisional}</span>}
        </p>
      ) : (
        <p className="text-hint opacity-40">&nbsp;</p>
      )}
    </div>
  )
}
