'use client'

import { AnimatePresence, motion } from 'motion/react'

export function SourceText({ text }: { text: string }) {
  return (
    <div
      dir="rtl"
      lang="ar"
      aria-label="Arabic source speech"
      className="font-arabic min-h-[1.6em] text-center text-lg text-hint md:text-xl"
    >
      <AnimatePresence mode="wait">
        <motion.span
          key={text}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          className="inline-block leading-relaxed"
        >
          {text || '\u00A0'}
        </motion.span>
      </AnimatePresence>
    </div>
  )
}
