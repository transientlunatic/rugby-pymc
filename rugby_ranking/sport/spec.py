"""What a sport *is*, separately from any model of it: scoring events, their
point values over time, and which events imply others.

Laws change. A ``SportSpec`` holds a timeline of ``ScoringRules`` (eras), each
effective from a date, so historical matches are scored under the rules of
their day and models can ask "how many points is a try worth *then*".
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class EventSpec:
    points: int
    # Events the laws award automatically alongside this one, e.g. a penalty
    # try carries an automatic conversion. Not recorded as separate events in
    # most data sources, so reconciliation has to add them.
    implies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoringRules:
    effective_from: date
    events: Mapping[str, EventSpec]
    note: str = ""


def _as_date(when) -> date | None:
    if when is None:
        return None
    if isinstance(when, datetime):
        return when.date()
    if isinstance(when, date):
        return when
    import pandas as pd

    return pd.Timestamp(when).date()


@dataclass(frozen=True)
class SportSpec:
    name: str
    eras: tuple[ScoringRules, ...]
    starters: int
    allows_draws: bool = True
    # canonical event -> accepted spellings in source data (compared after
    # lower-casing and replacing spaces/hyphens with underscores).
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Recognised source labels that score nothing (missed kicks, ...), so an
    # audit can tell them apart from labels it does not understand.
    non_scoring: tuple[str, ...] = ()

    def __post_init__(self):
        dates = [e.effective_from for e in self.eras]
        if not self.eras or dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError("eras must be non-empty with strictly increasing effective_from")

    # -- rules by date ------------------------------------------------------

    def rules_at(self, when=None) -> ScoringRules:
        """Rules in force on ``when`` (default: the latest era). Dates before
        the first era use the first era, and callers can check ``covers``."""
        d = _as_date(when)
        if d is None:
            return self.eras[-1]
        i = bisect_right([e.effective_from for e in self.eras], d) - 1
        return self.eras[max(i, 0)]

    def covers(self, when) -> bool:
        """False for dates before the first era, where the rules are a guess."""
        return _as_date(when) >= self.eras[0].effective_from

    def points(self, event: str, when=None) -> int:
        return self.rules_at(when).events[event].points

    def score(self, counts: Mapping[str, "int | np.ndarray"], when=None):
        """Total points for event counts (scalars or numpy arrays), including
        events implied by others (a penalty try's automatic conversion)."""
        rules = self.rules_at(when)
        total = 0
        for event, n in counts.items():
            spec = rules.events[event]
            total = total + n * (spec.points + sum(rules.events[i].points for i in spec.implies))
        return total

    # -- reading source data --------------------------------------------------

    def canonical(self, raw: str | None) -> str | None:
        """Canonical scoring event for a source label, or None if it isn't a
        scoring event (missed kicks, cards, ...)."""
        key = re.sub(r"[\s\-]+", "_", (raw or "").strip().lower())
        for event, names in self.aliases.items():
            if key == event or key in names:
                return event
        return None

    def is_non_scoring(self, raw: str | None) -> bool:
        return re.sub(r"[\s\-]+", "_", (raw or "").strip().lower()) in self.non_scoring

    def event_points(self, events: Iterable[Mapping], when=None) -> tuple[int, dict[str, int]]:
        """Points implied by a list of source scoring events, and the
        canonical event counts. Unknown labels are ignored (see ``audit``)."""
        counts: dict[str, int] = {}
        for e in events or ():
            c = self.canonical(e.get("type"))
            if c is not None:
                counts[c] = counts.get(c, 0) + 1
        return self.score(counts, when), counts
