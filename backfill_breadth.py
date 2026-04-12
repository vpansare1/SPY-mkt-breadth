#!/usr/bin/env python
"""
Manual backfill for cap-weighted breadth on specific dates.

Use when a scheduled GitHub Action ran with partial data and the row
was skipped (or needs to be overwritten). Fetches fresh prices, computes
cap-weighted breadth for each target date, and merges into
sp500_breadth_history.csv.

Usage:
    1. Edit BACKFILL_DATES below with the dates you want to fill.
    2. Run: python backfill_breadth.py
       (Run from the repo root so it picks up sp500_breadth_history.csv
        and sp500_weights_history.csv in the working directory.)

Notes:
    - Uses point-in-time weights from sp500_weights_history.csv when available
      (exact date match, or nearest within +/- 5 days). Falls back to scraping
      current weights only if no nearby snapshot exists.
    - Applies the same 90% coverage threshold; refuses to backfill bad data.
    - Safe to re-run: existing rows for the target dates are replaced.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from MarketBreadth import (
    MOMENTUM_WINDOWS,
    DATA_FILE,
    WEIGHTS_FILE,
    scrape_sp500_components,
    download_price_data,
    calculate_momentum,
)


def load_weights_for_date(target_date, max_gap_days=5):
    """Resolve point-in-time weights for a target date from sp500_weights_history.csv.

    Strategy:
      1. Exact column match for target_date.
      2. Nearest available column within +/- max_gap_days calendar days.
      3. Return None if nothing usable found (caller should fall back to scrape).

    Returns (weights_df, source_label) where weights_df has columns ['symbol', 'weight'].
    """
    if not os.path.exists(WEIGHTS_FILE):
        return None, None

    hist = pd.read_csv(WEIGHTS_FILE, index_col=0)
    available = pd.to_datetime(hist.columns, format="mixed")
    target = pd.to_datetime(target_date)

    # 1. Exact match
    exact = hist.columns[available == target]
    if len(exact) > 0:
        col = exact[0]
        w = hist[col].dropna()
        return pd.DataFrame({'symbol': w.index, 'weight': w.values}), f"exact match ({col})"

    # 2. Nearest within gap
    deltas = (available - target).days
    within = [(abs(d), d, i) for i, d in enumerate(deltas) if abs(d) <= max_gap_days]
    if within:
        within.sort()  # smallest absolute delta first; ties broken by signed delta (earlier wins)
        _, signed, idx = within[0]
        col = hist.columns[idx]
        w = hist[col].dropna()
        direction = "before" if signed < 0 else "after"
        return (pd.DataFrame({'symbol': w.index, 'weight': w.values}),
                f"nearest {direction} ({col}, {abs(signed)}d away)")

    return None, None

# ---- EDIT THIS LIST ----
BACKFILL_DATES = [
    "2026-04-07",
    "2026-04-08",
]
# ------------------------

MIN_TOTAL_WEIGHT = 90.0


def calculate_cap_weighted_breadth_for_date(prices_df, weights_df, target_date,
                                            min_total_weight=MIN_TOTAL_WEIGHT):
    """Same math as MarketBreadth.calculate_cap_weighted_breadth, but for an
    arbitrary date rather than prices_df.index[-1]."""
    target_ts = pd.to_datetime(target_date)

    # Find the matching trading day in the index (prices_df index may be tz-aware)
    idx = prices_df.index
    if idx.tz is not None:
        target_ts = target_ts.tz_localize(idx.tz) if target_ts.tzinfo is None else target_ts.tz_convert(idx.tz)

    # Match by calendar date (ignore time-of-day)
    matches = idx[idx.normalize() == target_ts.normalize()]
    if len(matches) == 0:
        print(f"  {target_date}: no trading day found in price index (weekend/holiday?)")
        return None
    actual_date = matches[0]

    weight_map = dict(zip(weights_df['symbol'], weights_df['weight']))
    results = []

    for window_name, window_days in MOMENTUM_WINDOWS.items():
        momentum = calculate_momentum(prices_df, window_days)
        row = momentum.loc[actual_date]

        total_weight = 0.0
        positive_weight = 0.0
        for symbol in row.index:
            if pd.notna(row[symbol]) and symbol in weight_map:
                w = weight_map[symbol]
                total_weight += w
                if row[symbol] > 0:
                    positive_weight += w

        if total_weight < min_total_weight:
            print(f"  {target_date} {window_name}: SKIP - total_weight={total_weight:.2f} "
                  f"below threshold {min_total_weight}")
            continue

        breadth_pct = (positive_weight / total_weight * 100) if total_weight > 0 else 0
        results.append({
            'Window': window_name,
            'Date': actual_date.strftime('%Y-%m-%d'),
            'Breadth_Pct': round(breadth_pct, 2),
            'Positive_Weight': round(positive_weight, 2),
            'Total_Weight': round(total_weight, 2),
        })

    if not results:
        return None
    return pd.DataFrame(results)


def merge_into_history(new_rows_list):
    """Merge a list of DataFrames into sp500_breadth_history.csv, replacing
    any existing rows on the same (Date, Window) pair."""
    new_df = pd.concat([df for df in new_rows_list if df is not None], ignore_index=True)
    if new_df.empty:
        print("\nNothing to merge.")
        return

    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        # Drop any existing rows that collide with the new rows on (Date, Window)
        keys = set(zip(new_df['Date'], new_df['Window']))
        mask = [not (d, w) in keys for d, w in zip(existing['Date'], existing['Window'])]
        existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Sort for readability: by Date, then by window order
    window_order = {name: i for i, name in enumerate(MOMENTUM_WINDOWS.keys())}
    combined['_w'] = combined['Window'].map(window_order)
    combined = combined.sort_values(['Date', '_w']).drop(columns='_w').reset_index(drop=True)

    combined.to_csv(DATA_FILE, index=False)
    print(f"\nMerged {len(new_df)} row(s) into {DATA_FILE}")
    print("\nBackfilled rows:")
    print(new_df.to_string(index=False))


def main():
    print("=" * 70)
    print("BACKFILL CAP-WEIGHTED BREADTH")
    print("=" * 70)
    print(f"Target dates: {BACKFILL_DATES}")

    # 1. Resolve weights for each target date (point-in-time from history file)
    print("\n" + "-" * 70)
    print("RESOLVING WEIGHTS PER DATE")
    print("-" * 70)
    per_date_weights = {}  # date_str -> (weights_df, source_label)
    current_weights_cache = None
    for d in BACKFILL_DATES:
        wdf, source = load_weights_for_date(d)
        if wdf is None:
            # Fallback: scrape current weights once and reuse
            if current_weights_cache is None:
                print(f"  {d}: no history within gap - scraping current weights as fallback")
                current_weights_cache = scrape_sp500_components()
            wdf, source = current_weights_cache, "current scrape (fallback)"
        print(f"  {d}: {source} - {len(wdf)} tickers, total weight {wdf['weight'].sum():.2f}")
        per_date_weights[d] = (wdf, source)

    # 2. Build union symbol universe and download prices once
    all_symbols = set()
    for wdf, _ in per_date_weights.values():
        all_symbols.update(wdf['symbol'].tolist())
    symbols = sorted(all_symbols)

    max_window_days = max(MOMENTUM_WINDOWS.values())
    earliest_target = min(pd.to_datetime(d) for d in BACKFILL_DATES)
    start_date = earliest_target - timedelta(days=int(max_window_days * 1.7) + 30)
    latest_target = max(pd.to_datetime(d) for d in BACKFILL_DATES)
    end_date = latest_target + timedelta(days=3)

    prices_df = download_price_data(symbols, start_date, end_date)
    if prices_df.empty:
        print("ERROR: no prices downloaded")
        return

    # 3. Coverage check on target dates
    print("\n" + "-" * 70)
    print("COVERAGE ON TARGET DATES")
    print("-" * 70)
    total_tickers = prices_df.shape[1]
    for d in BACKFILL_DATES:
        ts = pd.to_datetime(d)
        idx = prices_df.index
        if idx.tz is not None:
            ts = ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)
        matches = idx[idx.normalize() == ts.normalize()]
        if len(matches) == 0:
            print(f"  {d}: NOT IN INDEX")
            continue
        n = prices_df.loc[matches[0]].notna().sum()
        pct = n / total_tickers * 100
        print(f"  {d}: {n}/{total_tickers} ({pct:.1f}%)")

    # 4. Compute breadth per target date using its own weight snapshot
    print("\n" + "-" * 70)
    print("COMPUTING BREADTH")
    print("-" * 70)
    all_rows = []
    for d in BACKFILL_DATES:
        wdf, _ = per_date_weights[d]
        print(f"\n{d}:")
        rows = calculate_cap_weighted_breadth_for_date(prices_df, wdf, d)
        if rows is not None:
            print(rows.to_string(index=False))
        all_rows.append(rows)

    # 5. Merge
    merge_into_history(all_rows)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("Review the changes in sp500_breadth_history.csv, then commit.")


if __name__ == "__main__":
    main()
