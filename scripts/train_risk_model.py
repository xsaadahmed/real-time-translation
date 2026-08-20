"""Train the commit-risk model on harvested labels (README step 8).

    python scripts/train_risk_model.py --labels outputs/harvest_labels_200.jsonl

Splits by session (not by row) so ticks from the same utterance never leak
across train/test, fits a calibrated GradientBoosting classifier, and
reports both a ranking metric (ROC-AUC) and a calibration metric (Brier
score) - README's operating point (theta) is only meaningful if predicted
probabilities are actually calibrated, not just well-ranked.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from rtt.risk import RiskModel  # noqa: E402
from rtt.risk.model import DEFAULT_MODEL_PATH  # noqa: E402

logger = logging.getLogger("train_risk_model")


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def split_by_session(records: list[dict], test_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    sessions = sorted({r["session_id"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    n_test = max(1, int(len(sessions) * test_frac)) if len(sessions) > 1 else 0
    test_sessions = set(sessions[:n_test])
    train = [r for r in records if r["session_id"] not in test_sessions]
    test = [r for r in records if r["session_id"] in test_sessions]
    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True, help="Harvest JSONL file(s)")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = [Path(p) for p in args.labels]
    records = load_records(paths)
    if not records:
        logger.error("No records loaded from %s", paths)
        return 1

    positive = sum(1 for r in records if r["survived"])
    logger.info(
        "Loaded %d records from %d session(s), %d positive (%.1f%%)",
        len(records),
        len({r["session_id"] for r in records}),
        positive,
        100 * positive / len(records),
    )
    if positive < 10:
        logger.warning(
            "Fewer than 10 positive labels - this run is a plumbing check, "
            "not a usable model. Harvest more clips before trusting it."
        )

    train, test = split_by_session(records, args.test_frac, args.seed)
    logger.info("Train: %d records (%d sessions) / Test: %d records (%d sessions)",
                len(train), len({r["session_id"] for r in train}),
                len(test), len({r["session_id"] for r in test}))

    model = RiskModel().fit(train, random_state=args.seed)

    if test and len({r["survived"] for r in test}) > 1:
        y_true = np.array([bool(r["survived"]) for r in test], dtype=np.int64)
        y_prob = np.array([model.predict_survival_proba(r) for r in test])
        auc = roc_auc_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        logger.info("Held-out ROC-AUC: %.3f | Brier score: %.3f (lower is better)", auc, brier)
    else:
        logger.warning("Test split has only one class or is empty - skipping held-out metrics")

    model.save(args.output)
    logger.info("Saved calibrated risk model to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
