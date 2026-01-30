#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
import yfinance as yf
import pandas as pd


DEFAULT_TICKER = "^SSMI"
DEFAULT_START = "2005-01-01"
DEFAULT_OUT = Path("../../data/smi_yahoo_daily.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Swiss Market Index (SMI) daily data from Yahoo Finance."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Yahoo Finance ticker.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD.")
    parser.add_argument(
        "--end",
        default=dt.date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data = yf.download(
        args.ticker,
        start=args.start,
        end=args.end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        print("No data returned. Check ticker/date range.", file=sys.stderr)
        return 2

    data = data.reset_index()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Wrote {len(data)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
