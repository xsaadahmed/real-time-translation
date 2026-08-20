'use client'

import { useInterpreterStream } from '@/hooks/use-interpreter-stream'
import { RecordingIndicator } from './recording-indicator'
import { SessionControls } from './session-controls'
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
    phase,
    connecting,
    reconnecting,
    finalizing,
    finalized,
    error,
    durationSec,
    config,
    configBusy,
    configError,
    serverReady,
    updateConfig,
  } = useInterpreterStream()

  const indicatorMode = finalizing
    ? 'finalizing'
    : reconnecting
      ? 'reconnecting'
      : connecting
        ? 'connecting'
        : running
          ? 'listening'
          : 'idle'

  const toggle = () => {
    if (finalizing || connecting || reconnecting) return
    if (running) stop()
    else start()
  }

  const showCaret = running && !finalizing && !reconnecting
  const hasSession = Boolean(
    arabicVerified ||
      arabicProvisional ||
      englishVerified ||
      englishProvisional ||
      finalized ||
      error,
  )

  const idleArabic =
    connecting || reconnecting
      ? '…'
      : running
        ? 'Listening…'
        : error
          ? '—'
          : 'Arabic speech appears here.'

  const idleEnglish = error
    ? 'Start a new session when you are ready.'
    : 'English translation appears here.'

  return (
    <div className="flex min-h-svh flex-col bg-paper text-ink">
      <header className="flex items-start justify-between gap-6 px-6 py-5 md:px-10 md:py-7">
        <SessionControls
          config={config}
          busy={configBusy}
          error={configError}
          disabled={running || finalizing || connecting || reconnecting}
          onChange={updateConfig}
        />
        <RecordingIndicator
          mode={indicatorMode}
          disabled={finalizing || connecting || reconnecting}
          onToggle={toggle}
        />
      </header>

      <main className="flex-1 overflow-y-auto px-6 md:px-10">
        <div className="mx-auto flex min-h-full max-w-4xl flex-col justify-center gap-10 py-[14svh] md:gap-14">
          <SourceText
            verified={arabicVerified}
            provisional={arabicProvisional}
            idleHint={idleArabic}
          />
          <TranslationText
            verified={englishVerified}
            provisional={englishProvisional}
            showCaret={showCaret}
            idleHint={idleEnglish}
          />
        </div>
      </main>

      <footer className="flex items-center justify-center px-6 pb-8">
        <SessionStatus
          status={status}
          error={error}
          connecting={connecting}
          reconnecting={reconnecting}
          finalizing={finalizing}
          finalized={finalized}
          phase={phase}
          durationSec={durationSec}
          onReset={reset}
          onRetry={() => {
            reset()
            void start()
          }}
          canReset={hasSession && !running && !finalizing}
          serverReady={serverReady}
        />
      </footer>
    </div>
  )
}
