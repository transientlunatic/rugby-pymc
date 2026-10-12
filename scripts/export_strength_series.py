"""Export season-by-season team strengths for the blog's strength chart.

Two models are covered, the ones that carry a per-team strength:

* **Elo**: ``EloRatingSystem`` run chronologically over every played match in a
  Rugby-Data checkout, with (k, home advantage) picked by ``calibrate_elo`` on
  the most recent 3000 matches, as ``EloForecaster`` does, but regressing to the
  mean once a year (see ``elo_season_ends``). We record each team's rating after
  its last match of each season.
* **Bayesian**: the team-season strength (expected margin against a reference side,
  with a 90% interval) from ``dashboard/data/team_strength_rankings.json``, or from
  ``--bayes-file`` (the output of ``fit_strength_history.py``, which covers more seasons).

Teams and seasons follow ``dashboard/data/league_placings.json`` so the strength
chart lines up with the bump chart of final placings. Output is one JSON file.

    python scripts/export_strength_series.py --data-dir ../Rugby-Data --out strengths.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Same franchise, different name / sponsor (mirrors Rugby-Data's placings builder).
CANON = {'Borders': 'Border Reivers', 'Llanelli Scarlets': 'Scarlets',
         'Newport Gwent Dragons': 'Dragons RFC', 'DHL Stormers': 'Stormers',
         'Vodacom Bulls': 'Bulls', 'Hollywoodbets Sharks': 'Sharks',
         'Emirates Lions': 'Lions'}
# Non-club fixtures that leak into celtic-2013-2014.json.
EXCLUDE = {'England', 'Canada Women', 'France Women', 'Ireland Women'}
# The Celtic 2017-18 and 2018-19 files carry the wrong calendar year on their
# Aug-Dec dates (Rugby-Data issue #33); without a fix they interleave with the
# next season's matches and trigger spurious season regressions.
BAD_YEAR_SEASONS = {'2017-2018', '2018-2019'}


def _import_light():
    """Import the forecast/model submodules. If the package ``__init__`` can't
    run (it imports pymc, which Elo doesn't need), register empty parents so
    the pure-pandas submodules still import."""
    try:
        import rugby_ranking.forecast.store  # noqa: F401
        return
    except ModuleNotFoundError as err:
        if err.name != 'pymc' and not str(err.name).startswith('pymc'):
            raise
    for name in [n for n in sys.modules if n.startswith('rugby_ranking')]:
        del sys.modules[name]
    pkg = ROOT / 'rugby_ranking'
    for name, path in (('rugby_ranking', pkg), ('rugby_ranking.model', pkg / 'model'),
                       ('rugby_ranking.forecast', pkg / 'forecast')):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod


def elo_season_ends(data_dir: Path) -> dict[tuple[str, str, str], float]:
    _import_light()
    from rugby_ranking.forecast.store import MatchStore
    from rugby_ranking.model.elo import EloRatingSystem, calibrate_elo

    m = MatchStore.from_data_dir(data_dir).played.copy()
    m = m[~m['home_team'].isin(EXCLUDE) & ~m['away_team'].isin(EXCLUDE)]
    bad = m['competition'].eq('celtic') & m['season'].isin(BAD_YEAR_SEASONS) & (m['date'].dt.month >= 8)
    m.loc[bad, 'date'] = m.loc[bad, 'date'].apply(lambda d: d.replace(year=d.year - 1))
    m = m.sort_values('date', kind='stable').reset_index(drop=True)

    # EloRatingSystem regresses all ratings toward the mean whenever the season
    # label changes between consecutive matches. Pooled over competitions whose
    # seasons overlap, that happens ~1700 times (not once a year) and flattens
    # every rating, so regress once a year instead: the "Elo year" starts on 1 Aug.
    elo_year = m['date'].dt.year - (m['date'].dt.month < 8).astype(int)
    pooled = m.assign(season=elo_year.astype(str))

    tuned = calibrate_elo(pooled.tail(3000))
    print(f'Elo: k={tuned.k}, home advantage={tuned.home_advantage}', file=sys.stderr)
    elo = EloRatingSystem(k=tuned.k, home_advantage=tuned.home_advantage)
    last: dict[tuple[str, str, str], float] = {}
    for r, year in zip(m.itertuples(index=False), pooled['season']):
        elo.update(r.home_team, r.away_team, r.home_score, r.away_score, year)
        for team in (r.home_team, r.away_team):
            last[(CANON.get(team, team), r.season, r.competition)] = elo.ratings[team]
    return last


def bayes_strengths(path: Path, min_matches: int = 0) -> dict[tuple[str, str], dict]:
    """Team-season strengths, skipping team-seasons whose estimate rests on fewer than
    ``min_matches`` matches (rows carry ``n_matches`` when written by
    ``fit_strength_history.py``; the dashboard export has no such field)."""
    acc = collections.defaultdict(list)
    for r in json.load(open(path)):
        if r.get('n_matches', min_matches) < min_matches:
            continue
        acc[(CANON.get(r['team'], r['team']), r['season'])].append(r)
    mean = lambda rows, k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
    return {key: dict(mean=mean(rows, 'ref_margin_mean'), lo=mean(rows, 'ref_margin_lower'),
                      hi=mean(rows, 'ref_margin_upper'))
            for key, rows in acc.items()}


def build(data_dir: Path, bayes_file: Path | None = None, min_matches: int = 8) -> list[dict]:
    placings = json.load(open(data_dir / 'dashboard/data/league_placings.json'))
    elo = elo_season_ends(data_dir)
    bayes = bayes_strengths(bayes_file or data_dir / 'dashboard/data/team_strength_rankings.json', min_matches)
    out = []
    for league in placings:
        seasons = []
        for s in league['seasons']:
            teams = []
            for row in s['table']:
                e = elo.get((row['team'], s['season'], league['league']))
                b = bayes.get((row['team'], s['season']))
                teams.append(dict(team=row['short'], elo=None if e is None else round(e, 1),
                                  bayes=None if b is None else {k: round(v, 2) for k, v in b.items()}))
            seasons.append(dict(season=s['season'], status=s['status'], teams=teams))
        out.append(dict(league=league['league'], label=league['label'], seasons=seasons))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', type=Path, required=True, help='Rugby-Data checkout')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bayes-file', type=Path,
                    help='team-strength rows (default: the dashboard export; use the output of '
                         'fit_strength_history.py for a longer history)')
    ap.add_argument('--min-matches', type=int, default=8,
                    help='drop Bayesian team-seasons with fewer fitted matches than this')
    args = ap.parse_args()
    data = build(args.data_dir, args.bayes_file, args.min_matches)
    args.out.write_text(json.dumps(data, separators=(',', ':')))
    for lg in data:
        have = sum(1 for s in lg['seasons'] for t in s['teams'] if t['bayes'])
        total = sum(len(s['teams']) for s in lg['seasons'])
        print(f"{lg['league']}: {total} team-seasons, Bayesian strengths for {have}", file=sys.stderr)
