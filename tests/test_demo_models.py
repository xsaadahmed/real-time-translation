from rtt.asr.faster_whisper_asr import FasterWhisperASR
from rtt.config import ASRConfig, MTConfig


def test_empty_asr_prompt_is_neutral():
    assert FasterWhisperASR(ASRConfig(initial_prompt=""))._initial_prompt() is None


def test_exhibition_defaults_do_not_select_nllb():
    assert MTConfig().backend == "marian"
