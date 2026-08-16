'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { ALTERNATIVES, GENERIC_ALTERNATES } from '@/lib/translation-data'

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://127.0.0.1:8765/ws'
const PREDICT_MS = 780
const RESOLVE_MS = 480
const HOLD_MS = 1500
const MIC_CHUNK_MS = 500

/** where the correct next word turned out to be among the 3 predictions */
export type Winner = 'above' | 'center' | 'below' | 'miss'

export type Pending = {
  center: string
  above: string
  below: string
  winner: Winner
  target: string
}

export type Phase = 'predicting' | 'resolving'

type StreamState = {
  history: string[]
  committed: string
  pending: Pending | null
  phase: Phase
  revealedSource: string
  done: boolean
}

function pickTwo(pool: string[], exclude: string): [string, string] {
  const options = pool.filter((w) => w.toLowerCase() !== exclude.toLowerCase())
  const a = options[Math.floor(Math.random() * options.length)] ?? 'and'
  const rest = options.filter((w) => w !== a)
  const b = rest[Math.floor(Math.random() * rest.length)] ?? 'then'
  return [a, b]
}

function generatePending(realWord: string): Pending {
  const alts = ALTERNATIVES[realWord] ?? pickTwo(GENERIC_ALTERNATES, realWord)
  const r = Math.random()
  const winner: Winner = r < 0.56 ? 'center' : r < 0.71 ? 'above' : r < 0.86 ? 'below' : 'miss'

  let above = alts[0]
  let center = realWord
  let below = alts[1]

  if (winner === 'center') {
    above = alts[0]
    center = realWord
    below = alts[1]
  } else if (winner === 'above') {
    above = realWord
    center = alts[0]
    below = alts[1]
  } else if (winner === 'below') {
    above = alts[0]
    center = alts[1]
    below = realWord
  } else {
    const extra = pickTwo(GENERIC_ALTERNATES, realWord)
    above = alts[0]
    center = alts[1]
    below = extra[0]
  }

  return { above, center, below, winner, target: realWord }
}

function wordsIn(text: string): string[] {
  const trimmed = text.trim()
  return trimmed ? trimmed.split(/\s+/) : []
}

function sentenceEnded(word: string): boolean {
  return /[.!?؟۔]["']?$/.test(word)
}

async function resolveWsUrl(): Promise<string> {
  try {
    const res = await fetch('/runtime-config.json', { cache: 'no-store' })
    if (res.ok) {
      const data = (await res.json()) as { wsUrl?: string }
      if (data.wsUrl) return data.wsUrl
    }
  } catch {
    // use build-time / default URL
  }
  return DEFAULT_WS_URL
}

export function useInterpreterStream() {
  const [running, setRunning] = useState(false)
  const [state, setState] = useState<StreamState>(() => ({
    history: [],
    committed: '',
    pending: null,
    phase: 'predicting',
    revealedSource: '',
    done: false,
  }))

  const cursor = useRef(0)
  const sentenceStartRef = useRef(0)
  const pendingRef = useRef<Pending | null>(null)
  const historyRef = useRef<string[]>([])
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const runningRef = useRef(false)
  const animatingRef = useRef(false)

  const serverArabicRef = useRef('')
  const serverEnglishRef = useRef('')
  const serverWordsRef = useRef<string[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const pcmBufferRef = useRef<Float32Array[]>([])
  const pcmSamplesRef = useRef(0)

  const clearTimer = () => {
    if (timer.current) {
      clearTimeout(timer.current)
      timer.current = null
    }
  }

  const emit = useCallback((phase: Phase, done: boolean) => {
    const committed = serverWordsRef.current
      .slice(sentenceStartRef.current, sentenceStartRef.current + cursor.current)
      .join(' ')

    setState({
      history: historyRef.current,
      committed,
      pending: done ? null : pendingRef.current,
      phase,
      revealedSource: serverArabicRef.current,
      done,
    })
  }, [])

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
    const data = btoa(binary)

    ws.send(JSON.stringify({ type: 'audio', rate: sampleRate, data }))
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

  const step = useCallback(() => {
    const words = serverWordsRef.current.slice(sentenceStartRef.current)

    if (cursor.current >= words.length) {
      pendingRef.current = null
      emit('predicting', false)
      animatingRef.current = false
      return
    }

    animatingRef.current = true
    const realWord = words[cursor.current]
    pendingRef.current = generatePending(realWord)
    emit('predicting', false)

    timer.current = setTimeout(() => {
      emit('resolving', false)
      const p = pendingRef.current
      const resolveMs = p && p.winner === 'miss' ? 260 + p.target.length * 32 : RESOLVE_MS

      timer.current = setTimeout(() => {
        cursor.current += 1
        const committedLine = serverWordsRef.current
          .slice(sentenceStartRef.current, sentenceStartRef.current + cursor.current)
          .join(' ')

        if (sentenceEnded(realWord)) {
          pendingRef.current = null
          emit('predicting', true)
          timer.current = setTimeout(() => {
            historyRef.current = [...historyRef.current, committedLine]
            sentenceStartRef.current += cursor.current
            cursor.current = 0
            animatingRef.current = false
            if (runningRef.current) step()
          }, HOLD_MS)
          return
        }

        if (runningRef.current) step()
      }, resolveMs)
    }, PREDICT_MS)
  }, [emit])

  const maybeAnimate = useCallback(() => {
    if (!runningRef.current) return
    if (animatingRef.current) return
    const words = serverWordsRef.current.slice(sentenceStartRef.current)
    if (cursor.current < words.length) {
      step()
    }
  }, [step])

  const applyFinalUpdate = useCallback((arabic: string, english: string) => {
    clearTimer()
    animatingRef.current = false
    pendingRef.current = null
    historyRef.current = []
    cursor.current = 0
    sentenceStartRef.current = 0
    serverArabicRef.current = arabic
    serverEnglishRef.current = english
    serverWordsRef.current = wordsIn(english)

    setState({
      history: english.trim() ? [english.trim()] : [],
      committed: '',
      pending: null,
      phase: 'predicting',
      revealedSource: arabic,
      done: true,
    })
  }, [])

  const applyServerUpdate = useCallback(
    (arabic: string, english: string) => {
      const prevEnglish = serverEnglishRef.current
      serverArabicRef.current = arabic
      serverEnglishRef.current = english
      serverWordsRef.current = wordsIn(english)

      if (english && prevEnglish && !english.startsWith(prevEnglish.slice(0, 20))) {
        cursor.current = 0
        sentenceStartRef.current = 0
        historyRef.current = []
        pendingRef.current = null
        clearTimer()
        animatingRef.current = false
      }

      setState((prev) => ({ ...prev, revealedSource: arabic }))
      maybeAnimate()
    },
    [maybeAnimate],
  )

  const closeSocket = useCallback(() => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close()
    }
  }, [])

  const start = useCallback(async () => {
    if (runningRef.current) return

    clearTimer()
    historyRef.current = []
    pendingRef.current = null
    cursor.current = 0
    sentenceStartRef.current = 0
    animatingRef.current = false
    serverArabicRef.current = ''
    serverEnglishRef.current = ''
    serverWordsRef.current = []

    runningRef.current = true
    setRunning(true)
    setState({
      history: [],
      committed: '',
      pending: null,
      phase: 'predicting',
      revealedSource: '',
      done: false,
    })

    const wsUrl = await resolveWsUrl()
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string)
        if (msg.type === 'update') {
          applyServerUpdate(msg.arabic ?? '', msg.english ?? '')
        }
        if (msg.type === 'final') {
          applyFinalUpdate(msg.arabic ?? '', msg.english ?? '')
          runningRef.current = false
          setRunning(false)
          stopMic()
          closeSocket()
        }
      } catch {
        // ignore malformed payloads
      }
    }

    ws.onerror = () => {
      runningRef.current = false
      setRunning(false)
      stopMic()
    }

    ws.onclose = () => {
      if (runningRef.current) {
        runningRef.current = false
        setRunning(false)
        stopMic()
      }
    }

    ws.onopen = async () => {
      try {
        await startMic()
      } catch {
        runningRef.current = false
        setRunning(false)
        closeSocket()
      }
    }
  }, [applyServerUpdate, applyFinalUpdate, closeSocket, startMic, stopMic])

  const stop = useCallback(() => {
    if (!runningRef.current) return
    runningRef.current = false
    setRunning(false)
    clearTimer()
    stopMic()

    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
    } else {
      closeSocket()
    }
  }, [closeSocket, stopMic])

  const restart = useCallback(() => {
    clearTimer()
    stopMic()
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
    }
    closeSocket()
    runningRef.current = false
    setRunning(false)
    historyRef.current = []
    pendingRef.current = null
    cursor.current = 0
    sentenceStartRef.current = 0
    serverArabicRef.current = ''
    serverEnglishRef.current = ''
    serverWordsRef.current = []
    setState({
      history: [],
      committed: '',
      pending: null,
      phase: 'predicting',
      revealedSource: '',
      done: false,
    })
    setTimeout(() => start(), 400)
  }, [closeSocket, start, stopMic])

  useEffect(() => () => {
    clearTimer()
    stopMic()
    closeSocket()
  }, [closeSocket, stopMic])

  return { running, start, stop, restart, ...state }
}
