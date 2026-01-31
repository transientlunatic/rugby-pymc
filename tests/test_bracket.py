"""
Tests for knockout bracket structures and prediction.
"""

import pytest
import pandas as pd
import numpy as np

from rugby_ranking.model.bracket import (
    TBD,
    BracketMatch,
    BracketStructure,
    create_world_cup_bracket,
    create_champions_cup_bracket,
    create_urc_playoffs_bracket,
)


class TestTBD:
    """Test TBD placeholder functionality."""

    def test_create_tbd(self):
        """Test creating TBD placeholder."""
        tbd = TBD("Pool A winner")
        assert tbd.source == "Pool A winner"
        assert tbd.criteria == {}

    def test_tbd_with_criteria(self):
        """Test TBD with additional criteria."""
        tbd = TBD("Best runner-up", criteria={"method": "points"})
        assert tbd.source == "Best runner-up"
        assert tbd.criteria["method"] == "points"

    def test_tbd_string_representation(self):
        """Test string representation of TBD."""
        tbd = TBD("Winner QF1")
        assert str(tbd) == "TBD: Winner QF1"


class TestBracketMatch:
    """Test BracketMatch functionality."""

    def test_create_match_with_teams(self):
        """Test creating match with known teams."""
        match = BracketMatch(
            id="QF1",
            round_type="quarterfinal",
            round_number=1,
            home_team="France",
            away_team="Ireland",
        )

        assert match.id == "QF1"
        assert match.home_team == "France"
        assert match.away_team == "Ireland"
        assert match.is_determined()

    def test_create_match_with_tbd(self):
        """Test creating match with TBD teams."""
        match = BracketMatch(
            id="QF1",
            round_type="quarterfinal",
            round_number=1,
            home_team=TBD("Pool A winner"),
            away_team=TBD("Pool B runner-up"),
        )

        assert match.id == "QF1"
        assert isinstance(match.home_team, TBD)
        assert isinstance(match.away_team, TBD)
        assert not match.is_determined()

    def test_match_dependencies(self):
        """Test match dependency tracking."""
        match = BracketMatch(
            id="SF1",
            round_type="semifinal",
            round_number=1,
            home_team=TBD("Winner QF1"),
            away_team=TBD("Winner QF2"),
            depends_on=["QF1", "QF2"],
            winner_advances_to="Final",
        )

        assert match.has_dependencies()
        assert len(match.depends_on) == 2
        assert "QF1" in match.depends_on
        assert match.winner_advances_to == "Final"

    def test_get_tbd_placeholders(self):
        """Test extracting TBD placeholders from match."""
        match = BracketMatch(
            id="QF1",
            round_type="quarterfinal",
            round_number=1,
            home_team=TBD("Pool A winner"),
            away_team=TBD("Pool B runner-up"),
        )

        placeholders = match.get_tbd_placeholders()
        assert len(placeholders) == 2
        assert all(isinstance(p, TBD) for p in placeholders)


class TestBracketStructure:
    """Test BracketStructure functionality."""

    def test_create_simple_bracket(self):
        """Test creating a simple bracket structure."""
        structure = {
            "rounds": ["quarterfinal", "semifinal", "final"],
            "matches": [
                {
                    "id": "QF1",
                    "round_type": "quarterfinal",
                    "round_number": 1,
                    "home": {"team": "Team A"},
                    "away": {"team": "Team B"},
                    "winner_advances_to": "SF1",
                },
                {
                    "id": "SF1",
                    "round_type": "semifinal",
                    "round_number": 1,
                    "home": {"team": "TBC", "source": "Winner QF1"},
                    "away": {"team": "TBC", "source": "Winner QF2"},
                    "depends_on": ["QF1", "QF2"],
                    "winner_advances_to": "Final",
                },
            ],
        }

        bracket = BracketStructure("Test Tournament", structure)

        assert bracket.name == "Test Tournament"
        assert len(bracket.rounds) == 3
        assert len(bracket.matches) == 2
        assert "QF1" in bracket.matches
        assert "SF1" in bracket.matches

    def test_get_round_matches(self):
        """Test getting matches for a specific round."""
        bracket = create_world_cup_bracket()

        qf_matches = bracket.get_round_matches("quarterfinal")
        assert len(qf_matches) == 4

        sf_matches = bracket.get_round_matches("semifinal")
        assert len(sf_matches) == 2

        final_matches = bracket.get_round_matches("final")
        assert len(final_matches) == 1

    def test_get_dependencies(self):
        """Test getting match dependencies."""
        bracket = create_world_cup_bracket()

        # Quarterfinals have no dependencies
        qf1_deps = bracket.get_dependencies("QF1")
        assert len(qf1_deps) == 0

        # Semifinals depend on quarterfinals
        sf1_deps = bracket.get_dependencies("SF1")
        assert len(sf1_deps) == 2
        assert "QF1" in sf1_deps
        assert "QF3" in sf1_deps

    def test_get_dependent_matches(self):
        """Test finding matches that depend on a given match."""
        bracket = create_world_cup_bracket()

        # QF1 feeds into SF1
        qf1_dependents = bracket.get_dependent_matches("QF1")
        assert "SF1" in qf1_dependents

        # SF1 feeds into Final
        sf1_dependents = bracket.get_dependent_matches("SF1")
        assert "Final" in sf1_dependents

    def test_is_match_determined(self):
        """Test checking if match participants are known."""
        structure = {
            "rounds": ["final"],
            "matches": [
                {
                    "id": "Final1",
                    "round_type": "final",
                    "round_number": 1,
                    "home": {"team": "Team A"},
                    "away": {"team": "Team B"},
                },
                {
                    "id": "Final2",
                    "round_type": "final",
                    "round_number": 2,
                    "home": {"team": "TBC", "source": "Winner SF1"},
                    "away": {"team": "TBC", "source": "Winner SF2"},
                },
            ],
        }

        bracket = BracketStructure("Test", structure)

        assert bracket.is_match_determined("Final1")
        assert not bracket.is_match_determined("Final2")

    def test_get_all_tbd_placeholders(self):
        """Test getting all TBD placeholders in bracket."""
        bracket = create_world_cup_bracket()

        placeholders = bracket.get_all_tbd_placeholders()
        # World Cup bracket has many TBD placeholders
        assert len(placeholders) > 0

        # Each placeholder should be a (match_id, TBD) tuple
        for match_id, tbd in placeholders:
            assert isinstance(match_id, str)
            assert isinstance(tbd, TBD)

    def test_to_dataframe(self):
        """Test converting bracket to DataFrame."""
        bracket = create_world_cup_bracket()

        df = bracket.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "id" in df.columns
        assert "round_type" in df.columns
        assert "home_team" in df.columns
        assert "away_team" in df.columns
        assert len(df) == len(bracket.matches)

    def test_clone_bracket(self):
        """Test cloning a bracket."""
        bracket = create_world_cup_bracket()
        cloned = bracket.clone()

        assert cloned.name == bracket.name
        assert len(cloned.matches) == len(bracket.matches)
        # Ensure it's a deep copy
        assert cloned is not bracket
        assert cloned.matches is not bracket.matches


class TestBracketTemplates:
    """Test pre-defined bracket templates."""

    def test_world_cup_bracket(self):
        """Test World Cup bracket template."""
        bracket = create_world_cup_bracket()

        assert bracket.name == "Rugby World Cup"
        assert len(bracket.rounds) == 4
        assert "quarterfinal" in bracket.rounds
        assert "semifinal" in bracket.rounds
        assert "final" in bracket.rounds
        assert "third_place" in bracket.rounds

        # Should have 8 matches (4 QF + 2 SF + 1 Final + 1 3rd place)
        assert len(bracket.matches) == 8

        # Check QF structure
        qf_matches = bracket.get_round_matches("quarterfinal")
        assert len(qf_matches) == 4

        # All QF matches should have TBD teams from pools
        for match in qf_matches:
            assert isinstance(match.home_team, TBD)
            assert isinstance(match.away_team, TBD)
            assert "Pool" in match.home_team.source
            assert not match.home_advantage  # Neutral venues

    def test_champions_cup_bracket(self):
        """Test Champions Cup bracket template."""
        bracket = create_champions_cup_bracket()

        assert bracket.name == "Champions Cup"
        assert len(bracket.rounds) == 4

        # Should have 15 matches (8 R16 + 4 QF + 2 SF + 1 Final)
        assert len(bracket.matches) == 15

        # Check R16 structure
        r16_matches = bracket.get_round_matches("round_of_16")
        assert len(r16_matches) == 8

        # R16 matches should have home advantage
        for match in r16_matches:
            assert match.home_advantage

    def test_urc_playoffs_bracket(self):
        """Test URC playoffs bracket template."""
        bracket = create_urc_playoffs_bracket()

        assert bracket.name == "URC Playoffs"
        assert len(bracket.rounds) == 3

        # Should have 7 matches (4 QF + 2 SF + 1 Final)
        assert len(bracket.matches) == 7

        # Check seeding structure
        qf_matches = bracket.get_round_matches("quarterfinal")
        assert len(qf_matches) == 4

        # QF matches should have home advantage (higher seed)
        for match in qf_matches:
            assert match.home_advantage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
