'use client'

import type { RuntimeConfig } from '@/hooks/use-interpreter-stream'

type Props = {
  config: RuntimeConfig | null
  busy: boolean
  error: string | null
  disabled: boolean
  onChange: (patch: { live_asr?: string; final_asr?: string; live_mt?: string }) => void
}

function Select({
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  label: string
  value: string
  options: string[]
  disabled: boolean
  onChange: (value: string) => void
}) {
  return (
    <label className="flex flex-col gap-1 text-[11px] tracking-wide text-hint">
      <span>{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="max-w-[9.5rem] cursor-pointer border-0 border-b border-border bg-transparent py-1 text-xs text-ink outline-none transition-colors hover:border-hint focus:border-ink disabled:cursor-not-allowed disabled:opacity-50"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  )
}

/**
 * Optional live/final model selectors. Language pair is fixed Arabic → English.
 */
export function SessionControls({ config, busy, error, disabled, onChange }: Props) {
  if (!config) return null

  const locked = disabled || busy

  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex flex-wrap items-end justify-end gap-4 md:gap-6">
        <p className="pb-1 text-[11px] tracking-wide text-hint">
          {config.source_lang.toUpperCase()} → {config.target_lang.toUpperCase()}
        </p>
        <Select
          label="Live ASR"
          value={config.live_asr}
          options={config.options.live_asr}
          disabled={locked}
          onChange={(live_asr) => onChange({ live_asr })}
        />
        <Select
          label="Live MT"
          value={config.live_mt}
          options={config.options.live_mt}
          disabled={locked}
          onChange={(live_mt) => onChange({ live_mt })}
        />
        <Select
          label="Final ASR"
          value={config.final_asr}
          options={config.options.final_asr}
          disabled={locked}
          onChange={(final_asr) => onChange({ final_asr })}
        />
      </div>
      {busy && <p className="text-[11px] text-hint">Reloading models…</p>}
      {error && <p className="max-w-xs text-right text-[11px] text-destructive">{error}</p>}
    </div>
  )
}
