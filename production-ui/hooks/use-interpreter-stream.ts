'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://127.0.0.1:8765/ws'
const MIC_CHUNK_MS = 500
const MAX_RECONNECT_ATTEMPTS = 3
const RECONNECT_BASE_MS = 800

function sameOriginWsUrl(): string | null {
  if (typeof window === 'undefined') return null
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws`
}

function apiBaseFromWs(wsUrl: string): string {
  try {
    const u = new URL(wsUrl)
    u.protocol = u.protocol === 'wss:' ? 'https:' : 'http:'
    u.pathname = ''
    u.search = ''
    u.hash = ''
    return u.toString().replace(/\/$/, '')
  } catch {
    return ''
  }
}

async function resolveWsUrl(): Promise<string> {
  try {
    const res = await fetch('/runtime-config.json', { cache: 'no-store' })
    if (res.ok) {
      const data = (await res.json()) as { wsUrl?: string }
      if (data.wsUrl) return data.wsUrl
    }
  } catch {
    // fall through
  }

  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL
  }

  return sameOriginWsUrl() ?? DEFAULT_WS_URL
}

export type ServerPayload = {
  type?: string
  arabic?: string
  english?: string
  arabic_verified?: string
  arabic_provisional?: string
  english_verified?: string
  english_provisional?: string
  status?: string
  duration_sec?: number
  finalized?: boolean
  finalizing?: boolean
  phase?: string
  error?: string
}

export type SessionPhase =
  | 'idle'
  | 'connecting'
  | 'listening'
  | 'reconnecting'
  | 'finalizing'
  | 'final'
  | 'empty'
  | 'error'

export type InterpreterState = {
  arabicVerified: string
  arabicProvisional: string
  englishVerified: string
  englishProvisional: string
  status: string
  phase: SessionPhase
  connecting: boolean
  reconnecting: boolean
  finalizing: boolean
  finalized: boolean
  error: string | null
  durationSec: number
}

export type RuntimeConfig = {
  ready: boolean
  source_lang: string
  target_lang: string
  live_asr: string
  live_mt: string
  final_asr: string
  final_mt: string
  options: {
    live_asr: string[]
    final_asr: string[]
    live_mt: string[]
  }
}

const EMPTY_STATE: InterpreterState = {
  arabicVerified: '',
  arabicProvisional: '',
  englishVerified: '',
  englishProvisional: '',
  status: '',
  phase: 'idle',
  connecting: false,
  reconnecting: false,
  finalizing: false,
  finalized: false,
  error: null,
  durationSec: 0,
}

function applyPayload(state: InterpreterState, msg: ServerPayload): InterpreterState {
  if (msg.type === 'progress') {
    return {
      ...state,
      status: msg.status ?? state.status,
      phase: (msg.phase as SessionPhase) || 'finalizing',
      connecting: false,
      reconnecting: false,
      finalizing: true,
      finalized: false,
      error: null,
    }
  }

  if (msg.type === 'error') {
    return {
      ...state,
      status: msg.status ?? state.status,
      phase: 'error',
      connecting: false,
      reconnecting: false,
      finalizing: false,
      error: msg.error ?? msg.status ?? 'Something went wrong.',
    }
  }

  const finalized = msg.type === 'final' || msg.finalized === true
  const phase = (msg.phase as SessionPhase) || (finalized ? 'final' : 'listening')

  // After reconnect, ignore empty wipe from a brand-new server session.
  const keepLocal =
    state.reconnecting &&
    !msg.arabic_verified &&
    !msg.arabic_provisional &&
    !msg.english_verified &&
    !msg.english_provisional &&
    Boolean(state.arabicVerified || state.englishVerified)

  if (keepLocal) {
    return {
      ...state,
      status: msg.status ?? 'Listening… (reconnected)',
      phase: 'listening',
      connecting: false,
      reconnecting: false,
      finalizing: false,
      error: null,
    }
  }

  return {
    arabicVerified: msg.arabic_verified ?? state.arabicVerified,
    arabicProvisional: finalized ? '' : (msg.arabic_provisional ?? state.arabicProvisional),
    englishVerified: msg.english_verified ?? state.englishVerified,
    englishProvisional: finalized ? '' : (msg.english_provisional ?? state.englishProvisional),
    status: msg.status ?? state.status,
    phase,
    connecting: false,
    reconnecting: false,
    finalizing: finalized ? false : state.finalizing,
    finalized,
    error: null,
    durationSec: msg.duration_sec ?? state.durationSec,
  }
}

export function useInterpreterStream() {
  const [running, setRunning] = useState(false)
  const [state, setState] = useState<InterpreterState>(EMPTY_STATE)
  const [config, setConfig] = useState<RuntimeConfig | null>(null)
  const [configBusy, setConfigBusy] = useState(false)
  const [configError, setConfigError] = useState<string | null>(null)
  const [serverReady, setServerReady] = useState<boolean | null>(null)

  const runningRef = useRef(false)
  const intentionalStopRef = useRef(false)
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wsUrlRef = useRef<string>('')
  const apiBaseRef = useRef<string>('')
  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const pcmBufferRef = useRef<Float32Array[]>([])
  const pcmSamplesRef = useRef(0)
  const micStartedRef = useRef(false)

  const stopMic = useCallback(() => {
    processorRef.current?.disconnect()
    processorRef.current = null
    audioCtxRef.current?.close()
    audioCtxRef.current = null
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
    pcmBufferRef.current = []
    pcmSamplesRef.current = 0
    micStartedRef.current = false
  }, [])

  const sendAudioChunk = useCallback((floatSamples: Float32Array, sampleRate: number) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    const int16 = new Int16Array(floatSamples.length)
    for (let i = 0; i < floatSamples.length; i++) {
      const sample = Math.max(-1, Math.min(1, floatSamples[i]))
      int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
    }

    const bytes = new Uint8Array(int16.buffer)
    let binary = ''
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])

    ws.send(JSON.stringify({ type: 'audio', rate: sampleRate, data: btoa(binary) }))
  }, [])

  const startMic = useCallback(async () => {
    if (micStartedRef.current) return

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    })
    mediaStreamRef.current = stream

    const ctx = new AudioContext()
    audioCtxRef.current = ctx
    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    processorRef.current = processor

    const sampleRate = ctx.sampleRate
    const chunkSamples = Math.max(4096, Math.floor(sampleRate * (MIC_CHUNK_MS / 1000)))

    processor.onaudioprocess = (event) => {
      if (!runningRef.current) return
      const input = event.inputBuffer.getChannelData(0)
      pcmBufferRef.current.push(input.slice())
      pcmSamplesRef.current += input.length

      if (pcmSamplesRef.current < chunkSamples) return

      const merged = new Float32Array(pcmSamplesRef.current)
      let offset = 0
      for (const chunk of pcmBufferRef.current) {
        merged.set(chunk, offset)
        offset += chunk.length
      }
      pcmBufferRef.current = []
      pcmSamplesRef.current = 0
      sendAudioChunk(merged, sampleRate)
    }

    source.connect(processor)
    processor.connect(ctx.destination)
    micStartedRef.current = true
  }, [sendAudioChunk])

  const closeSocket = useCallback(() => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws) {
      ws.onclose = null
      ws.onerror = null
      ws.onmessage = null
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
    }
  }, [])

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }, [])

  const handleMessage = useCallback((msg: ServerPayload) => {
    if (msg.type === 'pong') return
    if (msg.type === 'update' || msg.type === 'final' || msg.type === 'progress' || msg.type === 'error') {
      setState((prev) => applyPayload(prev, msg))
    }
    if (msg.type === 'final') {
      runningRef.current = false
      intentionalStopRef.current = true
      setRunning(false)
      stopMic()
      closeSocket()
      reconnectAttemptRef.current = 0
    }
    if (msg.type === 'error') {
      runningRef.current = false
      intentionalStopRef.current = true
      setRunning(false)
      stopMic()
      closeSocket()
    }
  }, [closeSocket, stopMic])

  const connectSocket = useCallback(async (opts: { reconnect: boolean }) => {
    const wsUrl = wsUrlRef.current || (await resolveWsUrl())
    wsUrlRef.current = wsUrl
    apiBaseRef.current = apiBaseFromWs(wsUrl)

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        handleMessage(JSON.parse(event.data as string) as ServerPayload)
      } catch {
        // ignore malformed payloads
      }
    }

    ws.onerror = () => {
      // onclose handles reconnect / error messaging
    }

    ws.onclose = () => {
      wsRef.current = null
      if (!runningRef.current || intentionalStopRef.current) return

      if (reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
        const attempt = reconnectAttemptRef.current + 1
        reconnectAttemptRef.current = attempt
        const delay = RECONNECT_BASE_MS * 2 ** (attempt - 1)
        setState((prev) => ({
          ...prev,
          reconnecting: true,
          connecting: false,
          phase: 'reconnecting',
          status: `Connection lost — reconnecting (${attempt}/${MAX_RECONNECT_ATTEMPTS})…`,
          error: null,
        }))
        clearReconnectTimer()
        reconnectTimerRef.current = setTimeout(() => {
          void connectSocket({ reconnect: true })
        }, delay)
        return
      }

      runningRef.current = false
      setRunning(false)
      stopMic()
      setState((prev) => ({
        ...prev,
        connecting: false,
        reconnecting: false,
        finalizing: false,
        phase: 'error',
        error: 'Connection lost. Check that the translation server is running, then try again.',
        status: '',
      }))
    }

    ws.onopen = async () => {
      reconnectAttemptRef.current = 0
      setState((prev) => ({
        ...prev,
        connecting: false,
        reconnecting: false,
        phase: 'listening',
        status: opts.reconnect ? 'Listening… (reconnected)' : 'Listening…',
        error: null,
      }))
      try {
        await startMic()
      } catch {
        runningRef.current = false
        intentionalStopRef.current = true
        setRunning(false)
        closeSocket()
        stopMic()
        setState((prev) => ({
          ...prev,
          phase: 'error',
          error: 'Microphone access denied or unavailable.',
          connecting: false,
          reconnecting: false,
        }))
      }
    }
  }, [clearReconnectTimer, closeSocket, handleMessage, startMic, stopMic])

  const refreshConfig = useCallback(async () => {
    try {
      const wsUrl = wsUrlRef.current || (await resolveWsUrl())
      wsUrlRef.current = wsUrl
      const base = apiBaseFromWs(wsUrl) || (typeof window !== 'undefined' ? window.location.origin : '')
      apiBaseRef.current = base

      const readyRes = await fetch(`${base}/ready`, { cache: 'no-store' })
      setServerReady(readyRes.ok)

      const res = await fetch(`${base}/config`, { cache: 'no-store' })
      if (!res.ok) return
      const data = (await res.json()) as RuntimeConfig
      setConfig(data)
      setConfigError(null)
    } catch {
      setServerReady(false)
    }
  }, [])

  const updateConfig = useCallback(
    async (patch: { live_asr?: string; final_asr?: string; live_mt?: string }) => {
      if (runningRef.current) {
        setConfigError('Stop the session before changing models.')
        return
      }
      setConfigBusy(true)
      setConfigError(null)
      try {
        const base =
          apiBaseRef.current ||
          apiBaseFromWs(wsUrlRef.current || (await resolveWsUrl())) ||
          window.location.origin
        const res = await fetch(`${base}/config`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(patch),
        })
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}))
          throw new Error(
            (detail as { detail?: string }).detail || `Config update failed (${res.status})`,
          )
        }
        const data = (await res.json()) as RuntimeConfig
        setConfig(data)
        setServerReady(true)
      } catch (err) {
        setConfigError(err instanceof Error ? err.message : 'Could not update models.')
      } finally {
        setConfigBusy(false)
      }
    },
    [],
  )

  const start = useCallback(async () => {
    if (runningRef.current) return

    clearReconnectTimer()
    intentionalStopRef.current = false
    reconnectAttemptRef.current = 0
    runningRef.current = true
    setRunning(true)
    setState({
      ...EMPTY_STATE,
      connecting: true,
      phase: 'connecting',
      status: 'Connecting…',
    })

    try {
      const wsUrl = await resolveWsUrl()
      wsUrlRef.current = wsUrl
      apiBaseRef.current = apiBaseFromWs(wsUrl)

      // Wait briefly if API is still loading models
      try {
        const readyRes = await fetch(`${apiBaseRef.current}/ready`, { cache: 'no-store' })
        if (!readyRes.ok) {
          setState((prev) => ({
            ...prev,
            status: 'Server is loading models…',
          }))
          for (let i = 0; i < 30 && runningRef.current; i++) {
            await new Promise((r) => setTimeout(r, 1000))
            const again = await fetch(`${apiBaseRef.current}/ready`, { cache: 'no-store' })
            if (again.ok) break
            if (i === 29) {
              runningRef.current = false
              setRunning(false)
              setState({
                ...EMPTY_STATE,
                phase: 'error',
                error: 'Translation server is not ready yet. Try again in a moment.',
              })
              return
            }
          }
        }
        setServerReady(true)
      } catch {
        // WS connect will surface the real error
      }

      await connectSocket({ reconnect: false })
    } catch {
      runningRef.current = false
      setRunning(false)
      setState({
        ...EMPTY_STATE,
        phase: 'error',
        error: 'Could not open WebSocket connection.',
      })
    }
  }, [clearReconnectTimer, connectSocket])

  const stop = useCallback(() => {
    if (!runningRef.current) return

    intentionalStopRef.current = true
    clearReconnectTimer()
    runningRef.current = false
    setRunning(false)
    stopMic()
    setState((prev) => ({
      ...prev,
      finalizing: true,
      reconnecting: false,
      connecting: false,
      phase: 'finalizing',
      status: 'Finalizing with high-quality model…',
    }))

    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
    } else {
      closeSocket()
      setState((prev) => ({
        ...prev,
        finalizing: false,
        phase: 'error',
        error: 'Lost connection before finalize.',
      }))
    }
  }, [clearReconnectTimer, closeSocket, stopMic])

  const reset = useCallback(() => {
    intentionalStopRef.current = true
    clearReconnectTimer()
    stopMic()
    closeSocket()
    runningRef.current = false
    setRunning(false)
    setState(EMPTY_STATE)
  }, [clearReconnectTimer, closeSocket, stopMic])

  useEffect(() => {
    void refreshConfig()
    const id = setInterval(() => {
      void refreshConfig()
    }, 15_000)
    return () => clearInterval(id)
  }, [refreshConfig])

  useEffect(
    () => () => {
      intentionalStopRef.current = true
      clearReconnectTimer()
      stopMic()
      closeSocket()
    },
    [clearReconnectTimer, closeSocket, stopMic],
  )

  return {
    running,
    start,
    stop,
    reset,
    config,
    configBusy,
    configError,
    serverReady,
    updateConfig,
    refreshConfig,
    ...state,
  }
}
