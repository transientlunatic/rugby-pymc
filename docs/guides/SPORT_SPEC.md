# Sport specs: scoring rules over time

`rugby_ranking.sport.RUGBY_UNION` records what scores points and how many,
as a timeline of eras, so historical matches are interpreted under the laws of
their day and models don't carry point values around as literals.

```python
from rugby_ranking.sport import RUGBY_UNION

RUGBY_UNION.points("try", "1991-10-01")    # 4
RUGBY_UNION.points("try", "2026-10-12")    # 5
RUGBY_UNION.score({"try": 2, "conversion": 1, "penalty_try": 1}, "2020-01-01")  # 2*5 + 2 + 7
```

A penalty try carries an automatic conversion (7 points). Source data records
it as a single event worth 5, so the spec's `implies` makes reconciliation add
the extra 2.

## Recording a law change

Edit `rugby_ranking/sport/rugby_union.py`: add a `ScoringRules` to `ERAS` with
its `effective_from` date. Everything that asks the spec for points picks it
up. Dates are approximate adoption dates; confirm against World Rugby's law
history, and per competition where unions adopted changes at different times.
Law changes that alter *rates* but not points (breakdown and tackle laws, the
50:22, card rules) aren't scoring rules; model them as era effects on scoring
level rather than here.

## Auditing the data

```bash
python -m rugby_ranking.sport audit --data-dir ../Rugby-Data --min-year 2023
```

One row per competition-season: how many team-games have scoring events, what
share of those sum to the final score under the rules of the day, how many
events carry a value that disagrees with the spec, and any labels the spec
doesn't recognise. As of 2026-10 the five domestic leagues and the European
cups reconcile at 100%; Super Rugby, NPC and Currie Cup record tries but not
kicks (about 1–9% reconcile), and internationals sit at 79–97%. Detailed
scoring events only exist from about 2011; earlier matches have final scores
only, so event-level models can't train on them but score-only forecasters can.

## Adding a sport

A `SportSpec` needs a name, its eras, a roster size and (optionally) label
aliases. Rugby league, for example, would be a new spec with 4-point tries,
2-point goals and 1-point field goals. The `forecast` and `backtest` packages
need nothing else for any sport with two teams and a scoreline.
