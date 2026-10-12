"""SportSpec: scoring rules over time, and the audit that checks recorded
scoring detail against final scores."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from rugby_ranking.model.predictions import team_points
from rugby_ranking.sport import RUGBY_UNION, EventSpec, ScoringRules, SportSpec, audit_scoring


class TestEras:
    def test_try_value_changes_in_1992(self):
        assert RUGBY_UNION.points("try", date(1991, 10, 1)) == 4   # 1991 World Cup
        assert RUGBY_UNION.points("try", date(1992, 7, 1)) == 5
        assert RUGBY_UNION.points("try", "2026-10-12") == 5
        assert RUGBY_UNION.points("try") == 5                      # default: current rules

    def test_accepts_datetimes_and_timestamps(self):
        import pandas as pd
        assert RUGBY_UNION.points("try", datetime(1990, 1, 1, tzinfo=timezone.utc)) == 4
        assert RUGBY_UNION.points("try", pd.Timestamp("1990-01-01", tz="UTC")) == 4

    def test_covers_flags_dates_before_first_era(self):
        assert RUGBY_UNION.covers("1987-06-01")
        assert not RUGBY_UNION.covers("1850-01-01")

    def test_eras_must_be_sorted_and_unique(self):
        a = ScoringRules(date(2000, 1, 1), {"goal": EventSpec(1)})
        b = ScoringRules(date(1990, 1, 1), {"goal": EventSpec(2)})
        with pytest.raises(ValueError):
            SportSpec("x", (a, b), starters=11)
        with pytest.raises(ValueError):
            SportSpec("x", (a, a), starters=11)
        with pytest.raises(ValueError):
            SportSpec("x", (), starters=11)

    def test_a_second_sport_needs_only_a_spec(self):
        league = SportSpec(
            "rugby-league",
            (ScoringRules(date(1983, 1, 1), {"try": EventSpec(4), "goal": EventSpec(2), "field_goal": EventSpec(1)}),),
            starters=13,
        )
        assert league.score({"try": 3, "goal": 2, "field_goal": 1}) == 17


class TestScore:
    def test_penalty_try_carries_an_automatic_conversion(self):
        assert RUGBY_UNION.score({"penalty_try": 1}, "2020-01-01") == 7
        assert RUGBY_UNION.score({"penalty_try": 1}, "1990-01-01") == 6   # 4 + 2

    def test_vectorised_over_numpy_arrays(self):
        t, c, p = np.array([3, 0]), np.array([2, 0]), np.array([1, 4])
        out = RUGBY_UNION.score({"try": t, "conversion": c, "penalty": p, "drop_goal": np.array([0, 1])})
        assert out.tolist() == [3 * 5 + 2 * 2 + 3, 4 * 3 + 3]

    def test_team_points_matches_the_old_hard_coded_formula(self):
        rng = np.random.default_rng(0)
        t, c, p = (rng.integers(0, 12, 1000) for _ in range(3))
        assert (team_points(t, c, p) == t * 5 + c * 2 + p * 3).all()


class TestLabels:
    @pytest.mark.parametrize("raw,expected", [
        ("Try", "try"), ("try", "try"), ("Penalty Try", "penalty_try"), ("penalty-try", "penalty_try"),
        ("Conversion", "conversion"), ("Penalty", "penalty"), ("Drop goal", "drop_goal"),
        ("Missed conversion", None), ("Yellow card", None), (None, None), ("", None),
    ])
    def test_canonical(self, raw, expected):
        assert RUGBY_UNION.canonical(raw) == expected

    def test_missed_kicks_are_known_non_scoring(self):
        assert RUGBY_UNION.is_non_scoring("Missed conversion")
        assert not RUGBY_UNION.is_non_scoring("Banana")

    def test_event_points(self):
        ev = [{"type": "Try"}, {"type": "Conversion"}, {"type": "Penalty"}, {"type": "Missed penalty"}]
        pts, counts = RUGBY_UNION.event_points(ev, "2020-01-01")
        assert pts == 10 and counts == {"try": 1, "conversion": 1, "penalty": 1}


def _match(competition, season, d, home_events, home_score, away_events=None, away_score=0):
    return SimpleNamespace(
        competition=competition, season=season, date=d, home_score=home_score, away_score=away_score,
        home_scores=home_events, away_scores=away_events or [],
    )


class TestAudit:
    D = datetime(2024, 1, 6, tzinfo=timezone.utc)

    def test_reconciles_including_implied_conversion(self):
        ok = [{"type": "Try", "value": 5}, {"type": "Conversion", "value": 2}, {"type": "Penalty", "value": 3}]
        pen_try = [{"type": "Penalty Try", "value": 5}]
        df = audit_scoring([_match("l", "2023-2024", self.D, ok, 10), _match("l", "2023-2024", self.D, pen_try, 7)], RUGBY_UNION)
        row = df.iloc[0]
        assert row["team_games"] == 4 and row["with_events"] == 0.5
        assert row["reconciled"] == 1.0 and row["value_mismatch"] == 0

    def test_flags_unreconciled_and_wrong_recorded_values(self):
        tries_only = [{"type": "Try", "value": 4}]   # source records 4 where the rules say 5
        df = audit_scoring([_match("l", "2023-2024", self.D, tries_only, 5)], RUGBY_UNION)
        row = df.iloc[0]
        assert row["reconciled"] == 1.0              # 1 try at 5 = 5
        assert row["value_mismatch"] == 1

        kicks_missing = [{"type": "Try", "value": 5}]
        df = audit_scoring([_match("l", "2023-2024", self.D, kicks_missing, 12)], RUGBY_UNION)
        assert df.iloc[0]["reconciled"] == 0.0

    def test_unknown_labels_reported_but_missed_kicks_are_not(self):
        ev = [{"type": "Try", "value": 5}, {"type": "Missed conversion", "value": 0}, {"type": "Banana", "value": 1}]
        df = audit_scoring([_match("l", "2023-2024", self.D, ev, 5)], RUGBY_UNION)
        assert df.iloc[0]["unknown_types"] == {"Banana": 1}

    def test_unplayed_matches_ignored(self):
        m = _match("l", "2023-2024", self.D, [], None)
        assert audit_scoring([m], RUGBY_UNION).empty
