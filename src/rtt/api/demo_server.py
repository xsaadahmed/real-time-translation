"""Hardcoded demo WebSocket API — target draft-and-verify UX for demos.

Same protocol as production_server (verified/provisional fields, /ws, /ready,
/config) but no models. Speaks a scripted Arabic→English session so black
(committed) text grows while grey (provisional) stays ahead — the boss-facing
picture of the final product.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="RTT Demo API (target UX)", version="0.1.0-demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class DemoFrame:
    """One tick of the target interpreter UI."""

    delay_sec: float
    arabic_verified: str
    arabic_provisional: str
    english_verified: str
    english_provisional: str
    status: str


# Target latency story (README): first grey ~0.6s, first black ~1.5s, then
# steady commits while speculative English stays ahead in grey.
# Full script runs ~50–55s before the stable end frame.
DEMO_SCRIPT: tuple[DemoFrame, ...] = (
    DemoFrame(
        0.60,
        "",
        "مرحباً",
        "",
        "Hello",
        "Draft lane · first grey (~600 ms)",
    ),
    DemoFrame(
        0.55,
        "",
        "مرحباً جميعاً",
        "",
        "Hello everyone",
        "Provisional Arabic + speculative English",
    ),
    DemoFrame(
        0.65,
        "مرحباً",
        "جميعاً أريد",
        "Hello",
        "everyone I want",
        "First commit · black text (~1.5 s)",
    ),
    DemoFrame(
        0.60,
        "مرحباً جميعاً",
        "أريد أن أتحدث",
        "Hello everyone",
        "I want to talk",
        "Verify lane agrees · promote to black",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد",
        "أن أتحدث عن",
        "Hello everyone I want",
        "to talk about",
        "Guards clear · commit continues",
    ),
    DemoFrame(
        0.60,
        "مرحباً جميعاً أريد أن",
        "أتحدث عن مشروع",
        "Hello everyone I want to",
        "talk about the project",
        "Steady draft · grey stays ahead",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث",
        "عن مشروع الترجمة",
        "Hello everyone I want to talk",
        "about the translation project",
        "Branch agreement · commit through 'talk'",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن",
        "مشروع الترجمة الفورية",
        "Hello everyone I want to talk about",
        "the simultaneous translation project",
        "Grey extends noun phrase",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع",
        "الترجمة الفورية",
        "Hello everyone I want to talk about the",
        "real-time translation project",
        "Grey revises wording (futures disagree)",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة",
        "الفورية الذي",
        "Hello everyone I want to talk about the translation",
        "project that",
        "Risk model ≥ θ · commit 'translation'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية",
        "الذي نبنيه",
        "Hello everyone I want to talk about the translation project",
        "that we are building",
        "Lag governor steady",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي",
        "نبنيه اليوم",
        "Hello everyone I want to talk about the translation project that",
        "we are building today",
        "Verify hop · promote relative clause head",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه",
        "اليوم الهدف",
        "Hello everyone I want to talk about the translation project that we are building",
        "today the goal",
        "New clause begins in grey",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم",
        "الهدف هو",
        "Hello everyone I want to talk about the translation project that we are building today",
        "the goal is",
        "Commit through 'today'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف",
        "هو تحويل الكلام",
        "Hello everyone I want to talk about the translation project that we are building today the goal",
        "is to turn speech",
        "Speculative English anticipates infinitive",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو",
        "تحويل الكلام العربي",
        "Hello everyone I want to talk about the translation project that we are building today the goal is",
        "to turn Arabic speech",
        "Guards: iḍāfa open · hold 'speech'",
    ),
    DemoFrame(
        0.75,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل",
        "الكلام العربي إلى",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn",
        "Arabic speech into",
        "iḍāfa closes · commit 'turn'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام",
        "العربي إلى إنجليزي",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic",
        "speech into English",
        "Grey stays one phrase ahead",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي",
        "إلى إنجليزي فوري",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech",
        "into live English",
        "Grey revises 'English' → 'live English'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى",
        "إنجليزي فوري على",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into",
        "live English on",
        "Commit through 'into'",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي",
        "فوري على الشاشة",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into English",
        "live on the screen",
        "Branch depth high · commit 'English'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري",
        "على الشاشة والأذن",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English",
        "on the screen and in the ear",
        "Eye vs ear latency gap (by design)",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على",
        "الشاشة والأذن",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on",
        "the screen and the ear",
        "Commit preposition · grey finishes couplet",
    ),
    DemoFrame(
        0.75,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة",
        "والأذن في الوقت",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen",
        "and the ear in real",
        "New temporal phrase in grey",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن",
        "في الوقت الحقيقي",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear",
        "in real time",
        "Commit through 'ear'",
    ),
    DemoFrame(
        0.75,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في",
        "الوقت الحقيقي النص",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in",
        "real time the text",
        "Sentence boundary · grey starts next idea",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت",
        "الحقيقي النص الرمادي",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real",
        "time the grey text",
        "Commit 'real' · explain UI metaphor",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي",
        "النص الرمادي توقع",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time",
        "the grey text is a prediction",
        "Clause complete · new subject in grey",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص",
        "الرمادي توقع قابل",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the",
        "grey text is a retractable prediction",
        "Grey elaborates before commit",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي",
        "توقع قابل للسحب",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey",
        "text is a retractable guess",
        "Grey revises 'prediction' → 'guess'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع",
        "قابل للسحب والأسود",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text",
        "is retractable and the black",
        "Commit 'grey text'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل",
        "للسحب والأسود مؤكد",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is",
        "retractable and the black is committed",
        "Contrast pair forming in grey",
    ),
    DemoFrame(
        0.75,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب",
        "والأسود مؤكد قبل",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable",
        "and the black is committed before",
        "Commit through 'retractable'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود",
        "مؤكد قبل الأذن",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black",
        "is committed before the ear",
        "Product principle · verify before the ear",
    ),
    DemoFrame(
        0.75,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد",
        "قبل الأذن وأخيراً",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed",
        "before the ear and finally",
        "Commit 'committed' · closing beat in grey",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل",
        "الأذن وأخيراً أعتقد",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before",
        "the ear and finally I think",
        "Garden-path safe · no forced sentence wait",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن",
        "وأخيراً أعتقد أننا",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear",
        "and finally I think we",
        "Commit through 'ear'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً",
        "أعتقد أننا يجب",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally",
        "I think we should",
        "Callback to cancel-meeting example",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد",
        "أننا يجب أن نلغي",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think",
        "we should cancel",
        "Grey anticipates cancel",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا",
        "يجب أن نلغي الاجتماع",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we",
        "should cancel the meeting",
        "Commit 'we' · classic hard case in grey",
    ),
    DemoFrame(
        0.75,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا يجب",
        "أن نلغي الاجتماع غداً",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we should",
        "cancel the meeting tomorrow",
        "Grey revises then stabilizes on 'tomorrow'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا يجب أن",
        "نلغي الاجتماع غداً بسبب",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we should",
        "cancel tomorrow's meeting because of",
        "Grey wording flickers then settles",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا يجب أن نلغي",
        "الاجتماع غداً بسبب السفر",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we should cancel",
        "the meeting tomorrow because of the trip",
        "Commit 'cancel' · reason still grey",
    ),
    DemoFrame(
        0.75,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا يجب أن نلغي الاجتماع",
        "غداً بسبب السفر",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we should cancel the meeting",
        "tomorrow because of the trip",
        "Commit through 'meeting'",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا يجب أن نلغي الاجتماع غداً",
        "بسبب السفر",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we should cancel the meeting tomorrow",
        "because of the trip",
        "Almost done · speculative TTS muted",
    ),
    DemoFrame(
        0.70,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا يجب أن نلغي الاجتماع غداً بسبب",
        "السفر",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we should cancel the meeting tomorrow because of",
        "the trip",
        "Final grey word · awaiting verify",
    ),
    DemoFrame(
        0.65,
        "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد أننا يجب أن نلغي الاجتماع غداً بسبب السفر",
        "",
        "Hello everyone I want to talk about the translation project that we are building today the goal is to turn Arabic speech into live English on the screen and the ear in real time the grey text is retractable and the black is committed before the ear and finally I think we should cancel the meeting tomorrow because of the trip",
        "",
        "Utterance stable · waiting for stop",
    ),
)

FINAL_AR = (
    "مرحباً جميعاً أريد أن أتحدث عن مشروع الترجمة الفورية الذي نبنيه اليوم "
    "الهدف هو تحويل الكلام العربي إلى إنجليزي فوري على الشاشة والأذن في الوقت "
    "الحقيقي النص الرمادي توقع قابل للسحب والأسود مؤكد قبل الأذن وأخيراً أعتقد "
    "أننا يجب أن نلغي الاجتماع غداً بسبب السفر"
)
FINAL_EN = (
    "Hello everyone I want to talk about the translation project that we are "
    "building today the goal is to turn Arabic speech into live English on the "
    "screen and the ear in real time the grey text is retractable and the black "
    "is committed before the ear and finally I think we should cancel the "
    "meeting tomorrow because of the trip"
)


def _payload(
    *,
    arabic_verified: str,
    arabic_provisional: str,
    english_verified: str,
    english_provisional: str,
    status: str,
    duration_sec: float,
    finalized: bool = False,
    phase: str = "listening",
) -> dict[str, Any]:
    arabic = " ".join(p for p in (arabic_verified, arabic_provisional) if p)
    english = " ".join(p for p in (english_verified, english_provisional) if p)
    return {
        "type": "final" if finalized else "update",
        "arabic": arabic,
        "english": english,
        "arabic_verified": arabic_verified,
        "arabic_provisional": "" if finalized else arabic_provisional,
        "english_verified": english_verified,
        "english_provisional": "" if finalized else english_provisional,
        "status": status,
        "duration_sec": duration_sec,
        "finalized": finalized,
        "phase": phase,
    }


class ConfigUpdate(BaseModel):
    live_asr: str | None = Field(default=None)
    final_asr: str | None = Field(default=None)
    live_mt: str | None = Field(default=None)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "demo"}


@app.get("/ready")
async def ready() -> JSONResponse:
    return JSONResponse({"status": "ready", "ready": True, "mode": "demo"})


@app.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    return {
        "ready": True,
        "mode": "demo",
        "sessions_started": 0,
        "note": "Demo server — no live model metrics",
    }


@app.get("/config")
async def get_config() -> dict[str, Any]:
    return {
        "ready": True,
        "mode": "demo",
        "source_lang": "ar",
        "target_lang": "en",
        "device": "demo",
        "live_asr": "demo-draft",
        "live_mt": "demo-branched",
        "final_asr": "demo-verify",
        "final_mt": "demo-commit",
        "options": {
            "live_asr": ["demo-draft"],
            "final_asr": ["demo-verify"],
            "live_mt": ["demo-branched"],
        },
    }


@app.put("/config")
async def put_config(body: ConfigUpdate) -> dict[str, Any]:
    _ = body
    return await get_config()


@app.websocket("/ws")
async def interpreter_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    t0 = time.perf_counter()
    stop_event = asyncio.Event()
    frame_index = 0
    last: DemoFrame | None = None

    async def send_frame(frame: DemoFrame, *, phase: str = "listening") -> None:
        nonlocal last
        last = frame
        await websocket.send_json(
            _payload(
                arabic_verified=frame.arabic_verified,
                arabic_provisional=frame.arabic_provisional,
                english_verified=frame.english_verified,
                english_provisional=frame.english_provisional,
                status=frame.status,
                duration_sec=time.perf_counter() - t0,
                phase=phase,
            )
        )

    async def play_script() -> None:
        nonlocal frame_index
        try:
            while frame_index < len(DEMO_SCRIPT) and not stop_event.is_set():
                frame = DEMO_SCRIPT[frame_index]
                await asyncio.sleep(frame.delay_sec)
                if stop_event.is_set():
                    break
                await send_frame(frame)
                frame_index += 1
            if frame_index >= len(DEMO_SCRIPT) and not stop_event.is_set():
                await websocket.send_json(
                    _payload(
                        arabic_verified=FINAL_AR,
                        arabic_provisional="",
                        english_verified=FINAL_EN,
                        english_provisional="",
                        status="Demo script complete — press Stop for a polished final, or New session",
                        duration_sec=time.perf_counter() - t0,
                        phase="listening",
                    )
                )
        except Exception:
            logger.exception("Demo script playback failed")

    await websocket.send_json(
        _payload(
            arabic_verified="",
            arabic_provisional="",
            english_verified="",
            english_provisional="",
            status="Demo mode · target UX (no models) — speak or wait; script starts now",
            duration_sec=0.0,
            phase="listening",
        )
    )

    script_task = asyncio.create_task(play_script())

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                # Mic audio ignored — script drives the UI.
                continue

            if "text" not in message:
                continue

            payload = json.loads(message["text"])
            msg_type = payload.get("type")

            if msg_type == "audio":
                continue

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "stop":
                stop_event.set()
                script_task.cancel()
                try:
                    await script_task
                except asyncio.CancelledError:
                    pass

                await websocket.send_json(
                    {
                        "type": "progress",
                        "status": "Committing remaining speculative text…",
                        "phase": "finalize",
                        "finalizing": True,
                        "finalized": False,
                    }
                )
                await asyncio.sleep(0.45)

                await websocket.send_json(
                    _payload(
                        arabic_verified=FINAL_AR,
                        arabic_provisional="",
                        english_verified=FINAL_EN,
                        english_provisional="",
                        status="Done — target UX demo (black = committed, grey was speculative)",
                        duration_sec=time.perf_counter() - t0,
                        finalized=True,
                        phase="final",
                    )
                )
                break
    except WebSocketDisconnect:
        logger.info("Demo WebSocket disconnected")
    finally:
        stop_event.set()
        if not script_task.done():
            script_task.cancel()
            try:
                await script_task
            except asyncio.CancelledError:
                pass
