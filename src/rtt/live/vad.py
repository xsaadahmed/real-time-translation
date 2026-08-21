"""Decide where to cut the live audio stream, using voice activity.

The fixed-window scheduler this replaces asks only "has enough time passed?".
That is the wrong question in two ways:

* A sentence that ends mid-window waits out the rest of the window before it
  can be transcribed, even though nothing more is coming.
* A window containing only silence still costs a full ASR pass, and Whisper
  pads every chunk to a fixed 30-second mel window, so that pass is not cheap
  just because the audio is quiet.

Asking "has the speaker paused?" fixes both: a finished utterance is cut and
translated the moment the pause is detected, and silence advances the cursor
for free. It also produces *closed* segments — spans that no later audio can
revise — which the caller can commit outright instead of holding as a
provisional guess that gets retranslated on every pass.

Detection uses the Silero model bundled with faster-whisper, so this adds no
dependency and no download.

Measured caveat, and the reason :class:`~rtt.config.VADConfig` defaults to off:
cutting at pauses produces *more* ASR passes, and on CPU a pass costs ~2.1s
fixed regardless of its length. The extra passes currently cost more than the
pause-triggered emission saves. See VADConfig for the numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import ASR_SAMPLE_RATE, VADConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentDecision:
    """One decision about the pending (un-transcribed) tail of the stream."""

    #: Absolute sample index to advance the cursor to.
    end: int
    #: True when ``end`` sits on silence, so no later audio can revise the text
    #: of this span and it can be committed rather than held as provisional.
    closed: bool
    #: False when the span is pure silence and should skip ASR entirely.
    speech: bool

    @property
    def skip_asr(self) -> bool:
        return not self.speech


class SpeechSegmenter:
    """Chooses segment boundaries in a growing audio buffer."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self._options = None
        #: Buffer size at the last detector run, for the rate gate below.
        self._last_checked_size = 0

    def _vad_options(self):
        if self._options is None:
            from faster_whisper.vad import VadOptions

            self._options = VadOptions(
                threshold=self.config.threshold,
                min_speech_duration_ms=self.config.min_speech_ms,
                min_silence_duration_ms=self.config.min_silence_ms,
                # Padding is applied when cutting, not when detecting, so the
                # trailing-silence measurement below stays honest.
                speech_pad_ms=0,
            )
        return self._options

    def decide(self, audio: np.ndarray, processed_samples: int) -> SegmentDecision | None:
        """Return where to cut, or ``None`` to keep listening."""
        pending_size = audio.size - processed_samples
        if pending_size <= 0:
            return None

        sample_rate = ASR_SAMPLE_RATE
        pending_sec = pending_size / sample_rate
        if pending_sec < self.config.min_pending_sec:
            return None

        # Detection is cheap per call but the processor loop ticks far faster
        # than audio arrives; without this gate it runs continuously and steals
        # CPU from ASR.
        step = int(self.config.min_check_step_sec * sample_rate)
        if audio.size - self._last_checked_size < step:
            return None
        self._last_checked_size = audio.size

        pending = audio[processed_samples:]

        try:
            from faster_whisper.vad import get_speech_timestamps

            timestamps = get_speech_timestamps(pending, self._vad_options())
        except Exception:
            # Never let the detector take the live path down: fall back to the
            # fixed-window behaviour by forcing a cut once enough has queued.
            logger.exception("VAD failed; falling back to a timed cut")
            if pending_sec >= self.config.max_segment_sec:
                return SegmentDecision(end=audio.size, closed=False, speech=True)
            return None

        if not timestamps:
            # Nothing but silence. Skip it outright rather than paying an ASR
            # pass to transcribe nothing.
            if pending_sec >= self.config.silence_skip_sec:
                return SegmentDecision(end=audio.size, closed=True, speech=False)
            return None

        last_speech_end = int(timestamps[-1]["end"])
        trailing_silence = pending_size - last_speech_end
        min_silence = int(self.config.min_silence_ms * sample_rate / 1000)

        if trailing_silence >= min_silence:
            # The speaker has paused: this utterance is complete.
            pad = int(self.config.speech_pad_ms * sample_rate / 1000)
            cut = min(pending_size, last_speech_end + pad)
            return SegmentDecision(
                end=processed_samples + cut, closed=True, speech=True
            )

        if pending_sec >= self.config.max_segment_sec:
            # Unbroken speech. Cut anyway so a monologue still updates, but the
            # tail stays provisional because the next words may revise it.
            return SegmentDecision(end=audio.size, closed=False, speech=True)

        return None


__all__ = ["SegmentDecision", "SpeechSegmenter"]
