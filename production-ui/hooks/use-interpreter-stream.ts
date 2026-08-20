'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://127.0.0.1:8765/ws'
const MIC_CHUNK_MS = 500

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
}

export type InterpreterState = {
  arabicVerified: string
  arabicProvisional: string
  englishVerified: string
  englishProvisional: string
  status: string
  connecting: boolean
  finalizing: boolean
  finalized: boolean
  error: string | null
}

const EMPTY_STATE: InterpreterState = {
  arabicVerified: '',
  arabicProvisional: '',
  englishVerified: '',
  englishProvisional: '',
  status: '',
  connecting: false,
  finalizing: false,
  finalized: false,
  error: null,
}

async function resolveWsUrl(): Promise<string> {
  try {
    const res = await fetch('/runtime-config.json', { cache: 'no-store' })
    if (res.ok) {
      const data = (await res.json()) as { wsUrl?: string }
      if (data.wsUrl) return data.wsUrl
    }
  } catch {
    // fall through to build-time / default URL
  }
  return DEFAULT_WS_URL
}

function applyPayload(state: InterpreterState, msg: ServerPayload): InterpreterState {
  const finalized = msg.type === 'final' || msg.finalized === true
  return {
    arabicVerified: msg.arabic_verified ?? state.arabicVerified,
    arabicProvisional: finalized ? '' : (msg.arabic_provisional ?? state.arabicProvisional),
    englishVerified: msg.english_verified ?? state.englishVerified,
    englishProvisional: finalized ? '' : (msg.english_provisional ?? state.englishProvisional),
    status: msg.status ?? state.status,
    connecting: false,
    finalizing: finalized ? false : state.finalizing,
    finalized,
    error: null,
  }
}

export function useInterpreterStream() {
  const [running, setRunning] = useState(false)
  const [state, setState] = useState<InterpreterState>(EMPTY_STATE)

  const runningRef = useRef(false)
  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const pcmBufferRef = useRef<Float32Array[]>([])
  const pcmSamplesRef = useRef(0)

  const stopMic = useCallback(() => {
    processorRef.current?.disconnect()
    processorRef.current = null
    audioCtxRef.current?.close()
    audioCtxRef.current = null
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
    pcmBufferRef.current = []
    pcmSamplesRef.current = 0
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
  }, [sendAudioChunk])

  const closeSocket = useCallback(() => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close()
    }
  }, [])

  const handleMessage = useCallback((msg: ServerPayload) => {
    if (msg.type === 'update' || msg.type === 'final') {
      setState((prev) => applyPayload(prev, msg))
    }
    if (msg.type === 'final') {
      runningRef.current = false
      setRunning(false)
      stopMic()
      closeSocket()
    }
  }, [closeSocket, stopMic])

  const start = useCallback(async () => {
    if (runningRef.current) return

    runningRef.current = true
    setRunning(true)
    setState({
      ...EMPTY_STATE,
      connecting: true,
      status: 'Connecting…',
    })

    try {
      const wsUrl = await resolveWsUrl()
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
        runningRef.current = false
        setRunning(false)
        stopMic()
        setState((prev) => ({
          ...prev,
          connecting: false,
          finalizing: false,
          error: 'Connection failed. Is the translation server running?',
        }))
      }

      ws.onclose = () => {
        if (runningRef.current) {
          runningRef.current = false
          setRunning(false)
          stopMic()
          setState((prev) => ({
            ...prev,
            connecting: false,
            error: prev.error ?? 'Connection closed.',
          }))
        }
      }

      ws.onopen = async () => {
        setState((prev) => ({ ...prev, connecting: false, status: 'Listening…' }))
        try {
          await startMic()
        } catch {
          runningRef.current = false
          setRunning(false)
          closeSocket()
          setState((prev) => ({
            ...prev,
            error: 'Microphone access denied or unavailable.',
          }))
        }
      }
    } catch {
      runningRef.current = false
      setRunning(false)
      setState({
        ...EMPTY_STATE,
        error: 'Could not open WebSocket connection.',
      })
    }
  }, [closeSocket, handleMessage, startMic, stopMic])

  const stop = useCallback(() => {
    if (!runningRef.current) return

    runningRef.current = false
    setRunning(false)
    stopMic()
    setState((prev) => ({
      ...prev,
      finalizing: true,
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
        error: 'Lost connection before finalize.',
      }))
    }
  }, [closeSocket, stopMic])

  const reset = useCallback(() => {
    stopMic()
    closeSocket()
    runningRef.current = false
    setRunning(false)
    setState(EMPTY_STATE)
  }, [closeSocket, stopMic])

  useEffect(() => () => {
    stopMic()
    closeSocket()
  }, [closeSocket, stopMic])

  return {
    running,
    start,
    stop,
    reset,
    ...state,
  }
}
