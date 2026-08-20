'use client'

import { useInterpreterStream } from '@/hooks/use-interpreter-stream'
import { RecordingIndicator } from './recording-indicator'
import { SessionStatus } from './session-status'
import { SourceText } from './source-text'
import { TranslationText } from './translation-text'

export function TranslationCanvas() {
  const {
    running,
    start,
    stop,
    reset,
    arabicVerified,
    arabicProvisional,
    englishVerified,
    englishProvisional,
    status,
    connecting,
    finalizing,
    finalized,
    error,
  } = useInterpreterStream()

  const indicatorMode = finalizing
    ? 'finalizing'
    : connecting
      ? 'connecting'
      : running
        ? 'listening'
        : 'idle'

  const toggle = () => {
    if (finalizing || connecting) return
    if (running) stop()
    else start()
  }

  const showCaret = running && !finalizing
  const hasSession = Boolean(
    arabicVerified ||
      arabicProvisional ||
      englishVerified ||
      englishProvisional ||
      finalized ||
      error,
  )

  return (
    <div className="flex min-h-svh flex-col bg-paper text-ink">
      <header className="flex items-center justify-end px-6 py-5 md:px-10 md:py-7">
        <RecordingIndicator
          mode={indicatorMode}
          disabled={finalizing || connecting}
          onToggle={toggle}
        />
      </header>

      <main className="flex-1 overflow-y-auto px-6 md:px-10">
        <div className="mx-auto flex min-h-full max-w-4xl flex-col justify-center gap-10 py-[14svh] md:gap-14">
          <SourceText verified={arabicVerified} provisional={arabicProvisional} />
          <TranslationText
            verified={englishVerified}
            provisional={englishProvisional}
            showCaret={showCaret}
          />
        </div>
      </main>

      <footer className="flex items-center justify-center px-6 pb-8">
        <SessionStatus
          status={status}
          error={error}
          connecting={connecting}
          finalizing={finalizing}
          finalized={finalized}
          onReset={reset}
          canReset={hasSession && !running && !finalizing}
        />
      </footer>
    </div>
  )
}
