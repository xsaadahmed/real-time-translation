"""Small gradient-boosted, calibrated P(survives) classifier (README step 8).

CPU-only, sub-millisecond inference by design: a shallow GradientBoosting
ensemble plus isotonic calibration, not a neural net - the commit policy
calls this once per candidate per tick and cannot afford more.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from .features import FEATURE_NAMES, record_to_features

DEFAULT_MODEL_PATH = Path("models") / "risk_model.joblib"


def _to_matrix(records: list[dict]) -> np.ndarray:
    rows = [record_to_features(r) for r in records]
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)


class RiskModel:
    """Wraps a calibrated classifier over :data:`FEATURE_NAMES`.

    Calibration matters more than raw accuracy here: README's commit policy
    picks a threshold theta ("commit at 99% predicted survival") and trusts
    the model's probabilities to mean what they say, not just to rank
    candidates correctly.
    """

    def __init__(self, classifier: CalibratedClassifierCV | None = None) -> None:
        self.classifier = classifier

    @property
    def is_fitted(self) -> bool:
        return self.classifier is not None

    def fit(
        self,
        records: list[dict],
        *,
        random_state: int = 0,
        n_estimators: int = 100,
        max_depth: int = 3,
        calibration_cv: int = 3,
    ) -> "RiskModel":
        X = _to_matrix(records)
        y = np.array([bool(r["survived"]) for r in records], dtype=np.int64)

        base = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        # Isotonic calibration needs enough folds to hold out; with a small
        # or heavily imbalanced harvest, fall back to fewer folds rather
        # than erroring, since the minority class count can be lower than
        # calibration_cv.
        min_class_count = min(np.bincount(y)) if len(np.unique(y)) > 1 else 1
        cv = max(2, min(calibration_cv, int(min_class_count)))
        self.classifier = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        self.classifier.fit(X, y)
        return self

    def predict_survival_proba(self, record: dict) -> float:
        """P(this candidate prefix survives to the final sentence)."""
        if self.classifier is None:
            raise RuntimeError("RiskModel not fitted or loaded")
        X = _to_matrix([record])
        return float(self.classifier.predict_proba(X)[0, 1])

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        if self.classifier is None:
            raise RuntimeError("Nothing to save - RiskModel not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "RiskModel":
        classifier = joblib.load(Path(path))
        return cls(classifier)


__all__ = ["DEFAULT_MODEL_PATH", "RiskModel"]
