'use client'

import type { RuntimeConfig } from '@/hooks/use-interpreter-stream'

type Props = {
  config: RuntimeConfig | null
  busy: boolean
  error: string | null
  disabled: boolean
  onChange: (patch: { live_asr?: string; final_asr?: string; live_mt?: string }) => void
}

const LABELS: Record<string, string> = {
  tiny: 'Tiny', base: 'Base', small: 'Small · recommended', medium: 'Medium',
  'large-v3-turbo': 'Large turbo', 'large-v3': 'Large v3',
}

export function SessionControls({ config, busy, error, disabled, onChange }: Props) {
  const selected = config?.loading_model || config?.live_asr || 'small'
  const larger = ['medium', 'large-v3-turbo', 'large-v3'].includes(selected)
  return (
    <div className="flex max-w-xs flex-col items-end gap-1.5">
      <label className="flex items-center gap-2 text-xs text-hint">
        <span>Speech model</span>
        <select aria-label="Speech model size" value={selected}
          disabled={!config || disabled || busy || Boolean(config.loading_model)}
          onChange={(event) => onChange({ live_asr: event.target.value })}
          className="cursor-pointer rounded-md border border-border bg-paper px-2 py-1.5 text-xs text-ink outline-none focus-visible:ring-1 focus-visible:ring-ink disabled:cursor-wait disabled:opacity-50">
          {(config?.options.live_asr ?? ['small']).map((size) => (
            <option key={size} value={size}>{LABELS[size] ?? size}</option>
          ))}
        </select>
      </label>
      {busy && <p role="status" className="text-right text-xs text-hint">Loading model… first use may download weights.</p>}
      {larger && !busy && <p className="text-right text-xs text-hint">More RAM and slower updates on CPU. First use downloads weights.</p>}
      {error && <p role="alert" className="text-right text-xs text-destructive">{error}</p>}
    </div>
  )
}
