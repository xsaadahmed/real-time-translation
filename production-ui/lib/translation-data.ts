// Simulated Arabic -> English interpreter stream.
// Each segment carries the Arabic source and its English target.
// The simulation confirms English words one by one (typed out like a
// typewriter) while showing a vertical stack of candidate continuations:
// the middle candidate is the "best" one the model commits to, and the
// candidates above/below are plausible alternatives it is still weighing.

export type Segment = {
  /** Arabic source, right-to-left */
  source: string
  /** English target, whitespace-separated */
  target: string
}

export const SEGMENTS: Segment[] = [
  {
    source: 'الكلب قفز فوق السياج وركض نحو المنزل',
    target: 'The dog jumped over the fence and ran toward the house',
  },
  {
    source: 'في الصباح الباكر كانت المدينة هادئة تماما',
    target: 'Early in the morning the city was completely quiet',
  },
  {
    source: 'قالت إنها ستسافر غدا إذا سمح الطقس بذلك',
    target: 'She said she would travel tomorrow if the weather allowed it',
  },
  {
    source: 'نحن نعمل على شيء جديد تماما هذه المرة',
    target: 'We are working on something entirely new this time',
  },
]

/**
 * Plausible-but-wrong alternatives for the NEXT word, keyed by the correct
 * word. The interpreter shows one alternative above and one below the real
 * word, so each key should provide (at least) two options. Missing keys fall
 * back to the generic pool below.
 */
export const ALTERNATIVES: Record<string, [string, string]> = {
  The: ['A', 'One'],
  dog: ['cat', 'fox'],
  jumped: ['leaped', 'climbed'],
  over: ['across', 'onto'],
  fence: ['wall', 'gate'],
  and: ['then', 'so'],
  ran: ['rushed', 'walked'],
  toward: ['into', 'past'],
  house: ['garden', 'yard'],
  Early: ['Late', 'Deep'],
  morning: ['dawn', 'day'],
  city: ['town', 'street'],
  completely: ['almost', 'nearly'],
  quiet: ['empty', 'still'],
  said: ['claimed', 'noted'],
  would: ['might', 'could'],
  travel: ['leave', 'depart'],
  tomorrow: ['today', 'soon'],
  weather: ['sky', 'wind'],
  allowed: ['permitted', 'let'],
  working: ['building', 'focused'],
  something: ['someone', 'anything'],
  entirely: ['totally', 'somewhat'],
  new: ['different', 'strange'],
  this: ['that', 'each'],
  time: ['moment', 'round'],
}

/** Generic fallback continuations used when a word has no specific alternates. */
export const GENERIC_ALTERNATES: string[] = [
  'perhaps',
  'meanwhile',
  'and',
  'while',
  'then',
  'so',
  'yet',
]
