"""Sport definitions: scoring rules over time, independent of any model."""

from rugby_ranking.sport.audit import audit_scoring
from rugby_ranking.sport.rugby_union import RUGBY_UNION
from rugby_ranking.sport.spec import EventSpec, ScoringRules, SportSpec

__all__ = ["EventSpec", "ScoringRules", "SportSpec", "RUGBY_UNION", "audit_scoring"]
