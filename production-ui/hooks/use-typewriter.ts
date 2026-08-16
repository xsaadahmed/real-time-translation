'use client'

import { useEffect, useRef, useState } from 'react'

/**
 * Types `target` out character-by-character. Unlike a naive typewriter, it
 * does NOT restart from scratch when `target` grows — it only types the newly
 * appended tail. It fully resets only when `target` diverges from what is
 * already on screen (e.g. the interpreter moves to a new sentence).
 */
export function useTypewriter(target: string, speed = 24) {
  const [displayed, setDisplayed] = useState('')
  const targetRef = useRef(target)
  const countRef = useRef(0)

  targetRef.current = target

  useEffect(() => {
    // new sentence (or cleared) — start over
    if (!target.startsWith(displayed)) {
      countRef.current = 0
      setDisplayed('')
    }

    const id = setInterval(() => {
      const t = targetRef.current
      if (countRef.current < t.length) {
        countRef.current += 1
        setDisplayed(t.slice(0, countRef.current))
      }
    }, speed)

    return () => clearInterval(id)
    // `displayed` intentionally excluded: it changes every tick and would
    // otherwise thrash the interval. Divergence is caught via startsWith above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, speed])

  const isTyping = displayed.length < target.length
  return { displayed, isTyping }
}
