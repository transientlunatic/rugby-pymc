"""Crediting scoring events to lineup players (``attribute_scores``)."""
from __future__ import annotations

from rugby_ranking.model.data import attribute_scores


def tries(lineup, scorers, kind="Try"):
    events = [{"type": kind, "player": s} for s in scorers]
    return [c["tries"] if kind == "Try" else c["penalties"] for c in attribute_scores(lineup, events)]


def test_full_name_ignores_case():
    # the events spell it "van der", the lineup "Van Der"
    assert tries(["Josh Van Der Flier", "Cian Healy"], ["Josh van der Flier"]) == [1, 0]


def test_surname_only_credits_a_unique_surname():
    assert tries(["Mako Vunipola", "Owen Farrell", "Jamie George"], ["Farrell", "Farrell"]) == [0, 2, 0]


def test_shared_surname_is_left_unattributed():
    # brothers in the same lineup: crediting both would double count, crediting the first would be a guess
    assert tries(["Mako Vunipola", "Billy Vunipola", "Owen Farrell"], ["Vunipola"]) == [0, 0, 0]


def test_initial_and_surname_separates_brothers():
    assert tries(["Mako Vunipola", "Billy Vunipola"], ["B Vunipola", "B. Vunipola", "M Vunipola"]) == [1, 2]


def test_initial_and_surname_needs_the_right_initial():
    assert tries(["Sam Simmonds", "Tom Simmonds"], ["S Simmonds"]) == [1, 0]
    assert tries(["Sam Simmonds"], ["T Simmonds"]) == [0]


def test_full_name_beats_a_shared_surname():
    assert tries(["Mako Vunipola", "Billy Vunipola"], ["Billy Vunipola"]) == [0, 1]


def test_compound_surname_alone():
    assert tries(["Josh Van Der Flier", "Cian Healy"], ["Van Der Flier", "van der flier"]) == [2, 0]
    assert tries(["Willie le Roux", "Faf de Klerk"], ["le Roux", "de Klerk"]) == [1, 1]


def test_unknown_scorer_and_penalty_tries_are_ignored():
    assert tries(["Owen Farrell"], ["Penalty", "Somebody Else", ""]) == [0]


def test_other_score_types_and_missing_fields():
    events = [{"type": "Penalty", "player": "Farrell"}, {"type": "Conversion", "player": "Farrell"},
              {"type": "Drop Goal", "player": "Farrell"}, {"type": "Missed conversion", "player": "Farrell"},
              {"type": "Try"}]
    assert attribute_scores(["Owen Farrell"], events) == [
        {"tries": 0, "conversions": 1, "penalties": 1, "drop_goals": 1}]


def test_empty_lineup_and_no_scores():
    assert attribute_scores([], [{"type": "Try", "player": "Farrell"}]) == []
    assert attribute_scores(["Owen Farrell"], []) == [{"tries": 0, "conversions": 0, "penalties": 0, "drop_goals": 0}]
