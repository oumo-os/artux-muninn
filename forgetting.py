"""
forgetting.py — Confidence decay and memory maintenance.

Implements: confidence = confidence * exp(-λ * elapsed_days)

Lifecycle:
  • run_decay()        — apply time-based decay to all LTM entries
  • run_maintenance()  — archive weak entries; delete very stale archive scars
  • reinforce()        — manually boost an entry's confidence
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Optional

from .db import Database
from .ltm import LTMManager


# Default decay constant.  λ = 0.01 → ~50% confidence after 70 days.
DEFAULT_LAMBDA = 0.01

# Entries below this threshold are demoted to archive (scar).
ARCHIVE_THRESHOLD = 0.2

# Archive scars older than this many days are deleted permanently.
ARCHIVE_TTL_DAYS = 365


class ForgettingEngine:
    def __init__(
        self,
        db: Database,
        ltm: LTMManager,
        decay_lambda: float = DEFAULT_LAMBDA,
        archive_threshold: float = ARCHIVE_THRESHOLD,
        archive_ttl_days: int = ARCHIVE_TTL_DAYS,
    ):
        self.db = db
        self.ltm = ltm
        self.lam = decay_lambda
        self.archive_threshold = archive_threshold
        self.archive_ttl_days = archive_ttl_days

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def run_decay(self, reference_time: Optional[datetime] = None) -> int:
        """
        Apply exponential decay to every LTM entry based on age.
        Returns the number of entries updated.
        """
        now = reference_time or datetime.utcnow()
        entries = self.ltm.get_all(min_confidence=0.0)
        updated = 0

        for entry in entries:
            elapsed_days = (now - entry.timestamp).total_seconds() / 86400.0
            new_conf = entry.confidence * math.exp(-self.lam * elapsed_days)
            if abs(new_conf - entry.confidence) > 0.001:
                self.ltm.update_confidence(entry.id, new_conf)
                updated += 1

        return updated

    # ------------------------------------------------------------------
    # Maintenance (archive + cleanup)
    # ------------------------------------------------------------------

    def run_maintenance(self, reference_time: Optional[datetime] = None) -> dict:
        """
        1. Archive LTM entries below the threshold.
        2. Permanently delete archive scars older than archive_ttl_days.

        Returns a summary dict.
        """
        now = reference_time or datetime.utcnow()
        archived = 0
        deleted_scars = 0

        # --- Step 1: demote weak LTM entries to archive ---
        entries = self.ltm.get_all(min_confidence=0.0)
        for entry in entries:
            if entry.confidence < self.archive_threshold:
                self.ltm.archive_entry(
                    content=entry.content,
                    original_type="ltm",
                    original_id=entry.id,
                    reason=f"confidence below threshold ({entry.confidence:.3f})",
                )
                self.ltm.delete(entry.id)
                archived += 1

        # --- Step 2: purge old archive scars ---
        cutoff = (now - timedelta(days=self.archive_ttl_days)).isoformat()
        with self.db.connection() as conn:
            result = conn.execute(
                "DELETE FROM archive WHERE timestamp < ? AND rehydrated = 0",
                (cutoff,),
            )
            deleted_scars = result.rowcount

        return {
            "archived_to_scar": archived,
            "deleted_old_scars": deleted_scars,
            "run_at": now.isoformat(),
        }

    # ------------------------------------------------------------------
    # Manual reinforcement
    # ------------------------------------------------------------------

    def reinforce(self, entry_id: str, amount: float = 0.1) -> Optional[float]:
        """
        Boost an entry's confidence by `amount` (capped at 1.0).
        Returns the new confidence or None if entry not found.
        """
        entry = self.ltm.get(entry_id)
        if entry is None:
            return None
        new_conf = min(1.0, entry.confidence + amount)
        self.ltm.update_confidence(entry_id, new_conf)
        return new_conf

    def decay_entry(self, entry_id: str, amount: float = 0.1) -> Optional[float]:
        """
        Manually reduce an entry's confidence by `amount`.
        Returns the new confidence or None if not found.
        """
        entry = self.ltm.get(entry_id)
        if entry is None:
            return None
        new_conf = max(0.0, entry.confidence - amount)
        self.ltm.update_confidence(entry_id, new_conf)
        return new_conf
