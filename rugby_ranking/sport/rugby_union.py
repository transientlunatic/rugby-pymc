"""Rugby union scoring through the eras present in the data.

Edit the ``ERAS`` table to record law changes; everything that asks the spec
for points picks them up. Dates are approximate adoption dates and should be
confirmed against World Rugby's law history (and per competition, since
unions adopted changes at different times).
"""

from datetime import date

from rugby_ranking.sport.spec import EventSpec, ScoringRules, SportSpec

_KICKS = {
    "conversion": EventSpec(2),
    "penalty": EventSpec(3),
    "drop_goal": EventSpec(3),
}

ERAS = (
    ScoringRules(
        effective_from=date(1900, 1, 1),
        events={"try": EventSpec(4), "penalty_try": EventSpec(4, implies=("conversion",)), **_KICKS},
        note="Try worth 4 points (to 1992). Drop goal 3 since 1948.",
    ),
    ScoringRules(
        effective_from=date(1992, 7, 1),
        events={"try": EventSpec(5), "penalty_try": EventSpec(5, implies=("conversion",)), **_KICKS},
        note="Try raised from 4 to 5 points (adopted across unions around the 1992-93 season).",
    ),
)

RUGBY_UNION = SportSpec(
    name="rugby-union",
    eras=ERAS,
    starters=15,
    aliases={
        "try": ("tr",),
        "penalty_try": ("pen_try", "penalty_try"),
        "conversion": ("conv",),
        "penalty": ("pen", "penalty_goal"),
        "drop_goal": ("dropgoal", "drop", "dg"),
    },
    non_scoring=("missed_conversion", "missed_penalty", "missed_drop_goal", "missed_penalty_goal"),
)
