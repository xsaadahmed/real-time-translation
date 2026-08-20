"""Learned, calibrated commit-risk model (README step 8).

Trains on the retrospective labels harvest.harvest_utterance produces:
for every candidate English prefix considered at commit time, whether it
actually survived to the final decoded sentence. See README "Learned,
calibrated risk head".
"""

from __future__ import annotations

from .features import FEATURE_NAMES, record_to_features
from .model import RiskModel

__all__ = ["FEATURE_NAMES", "RiskModel", "record_to_features"]
