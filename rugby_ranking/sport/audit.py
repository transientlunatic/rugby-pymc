"""Does the recorded scoring detail agree with the final scores under the
rules of the day? Run this before trusting any event-level model on a slice
of history."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from rugby_ranking.sport.spec import SportSpec


def audit_scoring(matches, spec: SportSpec) -> pd.DataFrame:
    """One row per (competition, season).

    ``matches`` are ``MatchData``-like (competition, season, date, home/away
    score and scoring-event lists). Columns: team_games, with_events (share of
    team-games that have scoring events), reconciled (share of those whose
    events sum to the final score under ``spec`` at that date),
    value_mismatch (events whose recorded ``value`` differs from the spec) and
    unknown_types (most common unrecognised labels).
    """
    acc: dict[tuple[str, str], dict] = {}
    for m in matches:
        if m.home_score is None or m.away_score is None:
            continue
        a = acc.setdefault((m.competition, m.season), dict(
            team_games=0, with_events=0, reconciled=0, value_mismatch=0, unknown=Counter()))
        for events, score in ((m.home_scores, m.home_score), (m.away_scores, m.away_score)):
            a["team_games"] += 1
            if not events:
                continue
            a["with_events"] += 1
            pts, _ = spec.event_points(events, m.date)
            a["reconciled"] += int(pts == score)
            for e in events:
                c = spec.canonical(e.get("type"))
                if c is None:
                    if not spec.is_non_scoring(e.get("type")):
                        a["unknown"][e.get("type")] += 1
                elif e.get("value") is not None and e["value"] != spec.points(c, m.date):
                    a["value_mismatch"] += 1
    rows = []
    for (comp, season), a in sorted(acc.items()):
        we = a["with_events"]
        rows.append(dict(
            competition=comp, season=season, team_games=a["team_games"],
            with_events=we / a["team_games"],
            reconciled=(a["reconciled"] / we) if we else float("nan"),
            value_mismatch=a["value_mismatch"],
            unknown_types=dict(a["unknown"].most_common(3)),
        ))
    return pd.DataFrame(rows)
