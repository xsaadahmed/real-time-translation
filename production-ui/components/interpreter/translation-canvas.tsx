'use client'

import { useInterpreterStream } from '@/hooks/use-interpreter-stream'
import { DemoControls } from './demo-controls'
import { RecordingIndicator } from './recording-indicator'
import { SourceText } from './source-text'
import { TranslationText } from './translation-text'

export function TranslationCanvas() {
  const {
    running,
    start,
    stop,
    restart,
    history,
    committed,
    pending,
    phase,
    revealedSource,
    done,
  } = useInterpreterStream()

  const toggle = () => (running ? stop() : start())

  return (
    <div className="flex min-h-svh flex-col bg-paper text-ink">
      <header className="flex items-center justify-end px-6 py-5 md:px-10 md:py-7">
        <RecordingIndicator active={running} onToggle={toggle} />
      </header>

      <main className="flex-1 overflow-y-auto px-6 md:px-10">
        <div className="mx-auto flex min-h-full max-w-4xl flex-col justify-center gap-10 py-[14svh] md:gap-14">
          <SourceText text={revealedSource} />
          <TranslationText
            history={history}
            committed={committed}
            pending={pending}
            phase={phase}
            running={running}
            done={done}
          />
        </div>
      </main>

      <footer className="flex items-center justify-center px-6 pb-8">
        <DemoControls running={running} onRestart={restart} />
      </footer>
    </div>
  )
}
