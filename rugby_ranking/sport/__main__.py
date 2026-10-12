"""python -m rugby_ranking.sport audit --data-dir ../Rugby-Data"""

import argparse
from pathlib import Path

import pandas as pd

from rugby_ranking.sport import RUGBY_UNION, audit_scoring


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["audit"])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--min-year", type=int, default=2010)
    args = p.parse_args(argv)

    from rugby_ranking.model.data import MatchDataset

    ds = MatchDataset(str(args.data_dir), fuzzy_match_names=False)
    ds.load_json_files()
    df = audit_scoring(ds.matches, RUGBY_UNION)
    df = df[df["season"].str.extract(r"(\d{4})")[0].astype(float) >= args.min_year]
    pd.set_option("display.width", 220, "display.max_rows", 500, "display.float_format", "{:.2f}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
