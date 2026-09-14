# -*- coding: utf-8 -*-
"""
Download a synchronized Divvy 2017 dataset suitable for physics-consistent
bike-sharing experiments.

Official City of Chicago Socrata datasets used:
  1) Divvy Trips:                        fg6s-gzvg
  2) Divvy Bicycle Stations - Historical: eq45-8inv

Why 2017?
---------
The historical Divvy trip table contains a true bike-level identifier
(`bike_id`) together with start/end stations and times. The historical station
status table contains station inventory (`available_bikes`) and dock status.
That combination is suitable for reconstructing user trips, bike continuity,
and aggregate station inventory dynamics.

Default study window
--------------------
Main window: 2017-07-01 00:00:00 <= t < 2017-08-01 00:00:00

Trip data are additionally downloaded with +/- 7 days of context so that the
previous/next public trip of the same bike can later be inspected near the
study boundaries. Historical station inventory is downloaded with +/- 1 day
of context to support state alignment at the start/end of the study window.

Important
---------
* Socrata timestamps in these historical datasets are "floating" local times.
  Treat them as Chicago local wall-clock times unless you explicitly localize
  them later.
* This script ONLY downloads and audits the raw synchronized data. It does not
  infer operator rebalancing or create the final closed-system training set.
* An optional Socrata app token can be supplied through the environment
  variable SOCRATA_APP_TOKEN. The script can work without one, but a token
  generally reduces rate-limit problems.

Dependencies
------------
    pip install requests pandas

Designed for Windows/Spyder as well as normal Python execution.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests


# =============================================================================
# USER SETTINGS
# =============================================================================

# Change only these values if you want a different 2017 continuous window.
START_TIME = "2017-07-01 00:00:00"
END_TIME_EXCLUSIVE = "2017-08-01 00:00:00"

# Context around the main window.
TRIP_CONTEXT_DAYS = 7
INVENTORY_CONTEXT_DAYS = 1

# Local output directory.
OUTPUT_ROOT = Path(r"D:/physic_predict_bike/divvy_2017_physics_raw")

# Socrata API settings.
SOCRATA_DOMAIN = "https://data.cityofchicago.org"
TRIP_DATASET_ID = "fg6s-gzvg"
INVENTORY_DATASET_ID = "eq45-8inv"
PAGE_SIZE = 50_000
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 2.0

# Save one CSV per day as the authoritative raw download. Combined CSVs are
# then generated from these daily chunks.
COMBINE_DAILY_FILES = True

# If True, existing non-empty daily files are reused. Set False to force a
# clean redownload.
REUSE_EXISTING_DAILY_FILES = True

# Optional Socrata token. Leave environment variable unset if you do not have
# one. Never hard-code private tokens into a public repository.
SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN", "").strip()


# =============================================================================
# OFFICIAL FIELD DEFINITIONS USED BY THIS SCRIPT
# =============================================================================

TRIP_COLUMNS = [
    "trip_id",
    "start_time",
    "stop_time",
    "bike_id",
    "trip_duration",
    "from_station_id",
    "from_station_name",
    "to_station_id",
    "to_station_name",
    "user_type",
    "gender",
    "birth_year",
]

INVENTORY_COLUMNS = [
    "id",
    "timestamp",
    "station_name",
    "total_docks",
    "docks_in_service",
    "available_docks",
    "available_bikes",
    "status",
    "latitude",
    "longitude",
]


# =============================================================================
# PATHS
# =============================================================================

TRIP_DIR = OUTPUT_ROOT / "trips"
TRIP_DAILY_DIR = TRIP_DIR / "daily_context"
INVENTORY_DIR = OUTPUT_ROOT / "station_inventory"
INVENTORY_DAILY_DIR = INVENTORY_DIR / "daily_context"
METADATA_DIR = OUTPUT_ROOT / "metadata"
AUDIT_DIR = OUTPUT_ROOT / "audit"

TRIP_CONTEXT_COMBINED = TRIP_DIR / "divvy_trips_context.csv"
TRIP_MAIN_COMBINED = TRIP_DIR / "divvy_trips_main_window.csv"
INVENTORY_CONTEXT_COMBINED = INVENTORY_DIR / "divvy_station_inventory_context.csv"
INVENTORY_MAIN_COMBINED = INVENTORY_DIR / "divvy_station_inventory_main_window.csv"
STATION_METADATA_FILE = METADATA_DIR / "station_metadata_from_historical_window.csv"
SUMMARY_JSON = AUDIT_DIR / "download_summary.json"
SUMMARY_TXT = AUDIT_DIR / "download_summary.txt"


# =============================================================================
# HELPERS
# =============================================================================


def parse_local_time(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is not None:
        # Socrata historical timestamps are floating local timestamps. Keep the
        # query naive to avoid silently shifting the requested window.
        ts = ts.tz_localize(None)
    return ts


START_TS = parse_local_time(START_TIME)
END_TS = parse_local_time(END_TIME_EXCLUSIVE)

if END_TS <= START_TS:
    raise ValueError("END_TIME_EXCLUSIVE must be later than START_TIME")

if START_TS.year != 2017 or (END_TS - pd.Timedelta(microseconds=1)).year != 2017:
    raise ValueError(
        "This downloader is intentionally restricted to a window fully inside 2017. "
        "Change the script only after re-auditing the historical schemas."
    )

TRIP_CONTEXT_START = START_TS - pd.Timedelta(days=TRIP_CONTEXT_DAYS)
TRIP_CONTEXT_END = END_TS + pd.Timedelta(days=TRIP_CONTEXT_DAYS)
INV_CONTEXT_START = START_TS - pd.Timedelta(days=INVENTORY_CONTEXT_DAYS)
INV_CONTEXT_END = END_TS + pd.Timedelta(days=INVENTORY_CONTEXT_DAYS)


for p in [
    OUTPUT_ROOT,
    TRIP_DIR,
    TRIP_DAILY_DIR,
    INVENTORY_DIR,
    INVENTORY_DAILY_DIR,
    METADATA_DIR,
    AUDIT_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "DivvyPhysicsDatasetDownloader/1.0",
        "Accept": "text/csv",
    }
)
if SOCRATA_APP_TOKEN:
    SESSION.headers.update({"X-App-Token": SOCRATA_APP_TOKEN})



def socrata_csv_endpoint(dataset_id: str) -> str:
    # The public SODA resource endpoint is convenient for SoQL filtering.
    return f"{SOCRATA_DOMAIN}/resource/{dataset_id}.csv"



def fmt_soql_timestamp(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000")



def request_csv(params: Dict[str, str]) -> pd.DataFrame:
    """GET a Socrata CSV page with retry/backoff and parse into DataFrame."""
    url = params.pop("__url")
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if r.status_code == 429 or 500 <= r.status_code < 600:
                wait_s = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                wait_s = min(wait_s, 60.0)
                print(
                    f"  HTTP {r.status_code}; retry {attempt}/{MAX_RETRIES} "
                    f"after {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                continue

            r.raise_for_status()

            content_type = (r.headers.get("content-type") or "").lower()
            text_head = r.text[:500].lower() if r.content else ""
            if "json" in content_type and "error" in text_head:
                raise RuntimeError(f"Socrata returned an error payload: {r.text[:1000]}")

            if not r.content:
                return pd.DataFrame()

            return pd.read_csv(io.BytesIO(r.content), dtype="string")

        except Exception as exc:  # network or parser issue
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_s = min(RETRY_BASE_SECONDS * (2 ** (attempt - 1)), 60.0)
            print(
                f"  request error on attempt {attempt}/{MAX_RETRIES}: {exc}\n"
                f"  retrying after {wait_s:.1f}s"
            )
            time.sleep(wait_s)

    token_note = (
        " A SOCRATA_APP_TOKEN is configured."
        if SOCRATA_APP_TOKEN
        else " No SOCRATA_APP_TOKEN is configured; adding one may help with rate limits."
    )
    raise RuntimeError(f"Socrata request failed after retries.{token_note}") from last_error



def daterange_days(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> Iterable[pd.Timestamp]:
    day = start.normalize()
    final_day = (end_exclusive - pd.Timedelta(microseconds=1)).normalize()
    while day <= final_day:
        yield day
        day += pd.Timedelta(days=1)



def download_one_day(
    *,
    dataset_id: str,
    time_field: str,
    columns: Sequence[str],
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    output_file: Path,
    order_fields: Sequence[str],
) -> pd.DataFrame:
    """
    Download one local-calendar day, paginating within the day.

    The daily file is the authoritative raw chunk for restartability. The
    returned DataFrame is also used for lightweight auditing.
    """
    if REUSE_EXISTING_DAILY_FILES and output_file.exists() and output_file.stat().st_size > 0:
        try:
            existing = pd.read_csv(output_file, dtype="string")
            missing = set(columns) - set(existing.columns)
            if missing:
                raise ValueError(f"missing columns {sorted(missing)}")
            print(f"  reuse {output_file.name}: {len(existing):,} rows")
            return existing
        except Exception as exc:
            print(f"  existing file invalid, redownloading {output_file.name}: {exc}")

    where = (
        f"{time_field} >= '{fmt_soql_timestamp(day_start)}' AND "
        f"{time_field} < '{fmt_soql_timestamp(day_end)}'"
    )
    order = ",".join(f"{f} ASC" for f in order_fields)
    select = ",".join(columns)

    pieces: List[pd.DataFrame] = []
    offset = 0
    page_no = 0

    while True:
        page_no += 1
        params = {
            "__url": socrata_csv_endpoint(dataset_id),
            "$select": select,
            "$where": where,
            "$order": order,
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        }
        df = request_csv(params)

        if df.empty:
            break

        missing = set(columns) - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Dataset {dataset_id} did not return expected columns: {sorted(missing)}. "
                f"Returned columns: {list(df.columns)}"
            )

        # Keep exactly the requested official columns and preserve IDs as text.
        df = df.loc[:, list(columns)].copy()
        pieces.append(df)
        print(
            f"  {day_start.date()} page {page_no}: {len(df):,} rows "
            f"(offset {offset:,})"
        )

        if len(df) < PAGE_SIZE:
            break
        offset += len(df)

    if pieces:
        day_df = pd.concat(pieces, ignore_index=True)
    else:
        day_df = pd.DataFrame(columns=list(columns))

    # Deduplicate exact duplicate records only. We deliberately do not collapse
    # distinct records just because they share a timestamp/station identifier.
    before = len(day_df)
    day_df = day_df.drop_duplicates().reset_index(drop=True)
    if len(day_df) != before:
        print(f"  removed {before - len(day_df):,} exact duplicate rows")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    day_df.to_csv(output_file, index=False)
    return day_df



def combine_csv_files(files: Sequence[Path], output_file: Path) -> None:
    """Combine same-schema CSVs without loading the whole dataset into RAM."""
    files = [Path(f) for f in files if Path(f).exists() and Path(f).stat().st_size > 0]
    if not files:
        raise RuntimeError(f"No files available to combine into {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as fout:
        wrote_header = False
        for f in files:
            with f.open("r", encoding="utf-8", newline="") as fin:
                header = fin.readline()
                if not header:
                    continue
                if not wrote_header:
                    fout.write(header)
                    wrote_header = True
                shutil.copyfileobj(fin, fout, length=1024 * 1024)



def filter_combined_csv_by_time(
    input_file: Path,
    output_file: Path,
    time_field: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    chunksize: int = 250_000,
) -> int:
    """Filter a large combined CSV into the exact main study window."""
    wrote_header = False
    n_written = 0
    if output_file.exists():
        output_file.unlink()

    for chunk in pd.read_csv(input_file, dtype="string", chunksize=chunksize):
        t = pd.to_datetime(chunk[time_field], errors="coerce")
        mask = (t >= start) & (t < end_exclusive)
        out = chunk.loc[mask].copy()
        if out.empty:
            continue
        out.to_csv(output_file, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        n_written += len(out)

    if not wrote_header:
        pd.DataFrame(columns=pd.read_csv(input_file, nrows=0).columns).to_csv(output_file, index=False)
    return n_written


# =============================================================================
# DOWNLOAD TRIPS
# =============================================================================


def download_trips() -> Tuple[List[Path], Dict[str, object]]:
    print("\n" + "=" * 100)
    print("DOWNLOAD DIVVY TRIPS (OFFICIAL CITY OF CHICAGO: fg6s-gzvg)")
    print("=" * 100)
    print(f"Trip context: {TRIP_CONTEXT_START} <= start_time < {TRIP_CONTEXT_END}")

    daily_files: List[Path] = []
    main_rows = 0
    context_rows = 0
    unique_bikes_context: Set[str] = set()
    unique_from_context: Set[str] = set()
    unique_to_context: Set[str] = set()
    missing_bike_rows = 0

    for day in daterange_days(TRIP_CONTEXT_START, TRIP_CONTEXT_END):
        day_end = min(day + pd.Timedelta(days=1), TRIP_CONTEXT_END)
        actual_start = max(day, TRIP_CONTEXT_START)
        if day_end <= actual_start:
            continue

        out_file = TRIP_DAILY_DIR / f"divvy_trips_{day.strftime('%Y-%m-%d')}.csv"
        df = download_one_day(
            dataset_id=TRIP_DATASET_ID,
            time_field="start_time",
            columns=TRIP_COLUMNS,
            day_start=actual_start,
            day_end=day_end,
            output_file=out_file,
            order_fields=["start_time", "trip_id"],
        )
        daily_files.append(out_file)
        context_rows += len(df)

        if not df.empty:
            bike = df["bike_id"].astype("string")
            missing_bike_rows += int((bike.isna() | (bike.fillna("").str.strip() == "")).sum())
            unique_bikes_context.update(x for x in bike.dropna().astype(str) if x.strip())
            unique_from_context.update(x for x in df["from_station_id"].dropna().astype(str) if x.strip())
            unique_to_context.update(x for x in df["to_station_id"].dropna().astype(str) if x.strip())

            t = pd.to_datetime(df["start_time"], errors="coerce")
            main_rows += int(((t >= START_TS) & (t < END_TS)).sum())

    if context_rows == 0:
        raise RuntimeError("No 2017 Divvy trip rows were downloaded. Check connectivity/API status.")

    # bike_id is the key reason for using this historical data. Fail loudly if
    # it is unexpectedly absent in the selected period.
    if missing_bike_rows > 0:
        raise RuntimeError(
            f"Downloaded trip data contain {missing_bike_rows:,} rows with missing bike_id. "
            "Do not proceed to bike-continuity reconstruction until audited."
        )

    stats = {
        "trip_context_rows": int(context_rows),
        "trip_main_window_rows": int(main_rows),
        "unique_bikes_context": int(len(unique_bikes_context)),
        "unique_from_station_ids_context": int(len(unique_from_context)),
        "unique_to_station_ids_context": int(len(unique_to_context)),
        "missing_bike_id_rows": int(missing_bike_rows),
    }
    return daily_files, stats


# =============================================================================
# DOWNLOAD HISTORICAL INVENTORY
# =============================================================================


def download_inventory() -> Tuple[List[Path], Dict[str, object]]:
    print("\n" + "=" * 100)
    print("DOWNLOAD HISTORICAL STATION INVENTORY (OFFICIAL CITY OF CHICAGO: eq45-8inv)")
    print("=" * 100)
    print(f"Inventory context: {INV_CONTEXT_START} <= timestamp < {INV_CONTEXT_END}")

    daily_files: List[Path] = []
    context_rows = 0
    main_rows = 0
    unique_station_ids: Set[str] = set()
    unique_timestamps: Set[pd.Timestamp] = set()
    negative_available_rows = 0

    for day in daterange_days(INV_CONTEXT_START, INV_CONTEXT_END):
        day_end = min(day + pd.Timedelta(days=1), INV_CONTEXT_END)
        actual_start = max(day, INV_CONTEXT_START)
        if day_end <= actual_start:
            continue

        out_file = INVENTORY_DAILY_DIR / f"divvy_station_inventory_{day.strftime('%Y-%m-%d')}.csv"
        df = download_one_day(
            dataset_id=INVENTORY_DATASET_ID,
            time_field="timestamp",
            columns=INVENTORY_COLUMNS,
            day_start=actual_start,
            day_end=day_end,
            output_file=out_file,
            order_fields=["timestamp", "id"],
        )
        daily_files.append(out_file)
        context_rows += len(df)

        if not df.empty:
            unique_station_ids.update(x for x in df["id"].dropna().astype(str) if x.strip())
            tt = pd.to_datetime(df["timestamp"], errors="coerce")
            unique_timestamps.update(x for x in tt.dropna().tolist())
            main_rows += int(((tt >= START_TS) & (tt < END_TS)).sum())

            av = pd.to_numeric(df["available_bikes"], errors="coerce")
            negative_available_rows += int((av < 0).sum())

    if context_rows == 0:
        raise RuntimeError("No historical station inventory rows were downloaded.")

    sorted_ts = np.array(sorted(unique_timestamps), dtype="datetime64[ns]")
    median_interval_seconds = None
    if len(sorted_ts) >= 2:
        diffs = np.diff(sorted_ts).astype("timedelta64[s]").astype(np.int64)
        diffs = diffs[diffs > 0]
        if len(diffs):
            median_interval_seconds = float(np.median(diffs))

    stats = {
        "inventory_context_rows": int(context_rows),
        "inventory_main_window_rows": int(main_rows),
        "unique_inventory_station_ids_context": int(len(unique_station_ids)),
        "unique_inventory_timestamps_context": int(len(unique_timestamps)),
        "median_global_inventory_timestamp_interval_seconds": median_interval_seconds,
        "negative_available_bikes_rows_context": int(negative_available_rows),
    }
    return daily_files, stats


# =============================================================================
# DERIVED DOWNLOAD PRODUCTS / AUDITS
# =============================================================================


def build_station_metadata_from_inventory(main_inventory_file: Path) -> pd.DataFrame:
    """
    Build a historical-window station table from the downloaded inventory.

    For each station we keep the last record in the main study window. This is
    NOT current 2026 metadata; it comes from the same historical 2017 window.
    """
    usecols = [
        "id",
        "timestamp",
        "station_name",
        "total_docks",
        "docks_in_service",
        "status",
        "latitude",
        "longitude",
    ]

    latest_parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(main_inventory_file, dtype="string", usecols=usecols, chunksize=250_000):
        chunk["_time"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["id", "_time"])
        if chunk.empty:
            continue
        latest_parts.append(
            chunk.sort_values("_time").groupby("id", as_index=False, sort=False).tail(1)
        )

    if not latest_parts:
        out = pd.DataFrame(columns=usecols)
    else:
        merged = pd.concat(latest_parts, ignore_index=True)
        out = (
            merged.sort_values("_time")
            .groupby("id", as_index=False, sort=False)
            .tail(1)
            .drop(columns=["_time"])
            .sort_values("id", kind="stable")
            .reset_index(drop=True)
        )

    out.to_csv(STATION_METADATA_FILE, index=False)
    return out



def audit_station_overlap(trip_file: Path, inventory_file: Path) -> Dict[str, object]:
    trip_station_ids: Set[str] = set()
    for chunk in pd.read_csv(
        trip_file,
        dtype="string",
        usecols=["from_station_id", "to_station_id"],
        chunksize=250_000,
    ):
        for c in ["from_station_id", "to_station_id"]:
            trip_station_ids.update(x for x in chunk[c].dropna().astype(str) if x.strip())

    inv_station_ids: Set[str] = set()
    for chunk in pd.read_csv(
        inventory_file,
        dtype="string",
        usecols=["id"],
        chunksize=250_000,
    ):
        inv_station_ids.update(x for x in chunk["id"].dropna().astype(str) if x.strip())

    overlap = trip_station_ids & inv_station_ids
    only_trip = trip_station_ids - inv_station_ids
    only_inv = inv_station_ids - trip_station_ids

    return {
        "trip_station_ids_main_window": len(trip_station_ids),
        "inventory_station_ids_main_window": len(inv_station_ids),
        "station_id_overlap": len(overlap),
        "trip_station_ids_absent_from_inventory": len(only_trip),
        "inventory_station_ids_without_main_window_trip": len(only_inv),
        "example_trip_station_ids_absent_from_inventory": sorted(only_trip)[:25],
    }



def write_summary(summary: Dict[str, object]) -> None:
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "DIVVY 2017 PHYSICS DATA DOWNLOAD SUMMARY",
        "=" * 80,
        f"Main study window: {START_TS} <= t < {END_TS}",
        f"Trip context: {TRIP_CONTEXT_START} <= start_time < {TRIP_CONTEXT_END}",
        f"Inventory context: {INV_CONTEXT_START} <= timestamp < {INV_CONTEXT_END}",
        "",
        "OFFICIAL SOURCES",
        f"Trips dataset: {SOCRATA_DOMAIN}/d/{TRIP_DATASET_ID}",
        f"Historical inventory dataset: {SOCRATA_DOMAIN}/d/{INVENTORY_DATASET_ID}",
        "",
        "KEY COUNTS",
    ]

    for key, value in summary.items():
        if key in {
            "official_sources",
            "files",
            "main_window",
            "trip_context_window",
            "inventory_context_window",
        }:
            continue
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "FILES",
            f"Trip context combined: {TRIP_CONTEXT_COMBINED}",
            f"Trip main window: {TRIP_MAIN_COMBINED}",
            f"Inventory context combined: {INVENTORY_CONTEXT_COMBINED}",
            f"Inventory main window: {INVENTORY_MAIN_COMBINED}",
            f"Historical station metadata: {STATION_METADATA_FILE}",
            "",
            "IMPORTANT",
            "The trip file contains bike_id. Do not substitute ride/trip ID for bike ID.",
            "This downloader does not yet infer operator rebalancing or build a closed system.",
        ]
    )

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print("=" * 100)
    print("DIVVY 2017 SYNCHRONIZED PHYSICS DATA DOWNLOADER")
    print("=" * 100)
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Main window: {START_TS} <= t < {END_TS}")
    print(f"Socrata token configured: {bool(SOCRATA_APP_TOKEN)}")
    print("Trips: fg6s-gzvg | Historical inventory: eq45-8inv")

    trip_daily_files, trip_stats = download_trips()
    inv_daily_files, inv_stats = download_inventory()

    if COMBINE_DAILY_FILES:
        print("\n" + "=" * 100)
        print("COMBINE DAILY FILES")
        print("=" * 100)

        combine_csv_files(trip_daily_files, TRIP_CONTEXT_COMBINED)
        print(f"Trip context combined -> {TRIP_CONTEXT_COMBINED}")
        n_trip_main = filter_combined_csv_by_time(
            TRIP_CONTEXT_COMBINED,
            TRIP_MAIN_COMBINED,
            "start_time",
            START_TS,
            END_TS,
        )
        print(f"Trip main window -> {TRIP_MAIN_COMBINED} ({n_trip_main:,} rows)")

        combine_csv_files(inv_daily_files, INVENTORY_CONTEXT_COMBINED)
        print(f"Inventory context combined -> {INVENTORY_CONTEXT_COMBINED}")
        n_inv_main = filter_combined_csv_by_time(
            INVENTORY_CONTEXT_COMBINED,
            INVENTORY_MAIN_COMBINED,
            "timestamp",
            START_TS,
            END_TS,
        )
        print(f"Inventory main window -> {INVENTORY_MAIN_COMBINED} ({n_inv_main:,} rows)")

        station_meta = build_station_metadata_from_inventory(INVENTORY_MAIN_COMBINED)
        print(f"Historical station metadata -> {STATION_METADATA_FILE} ({len(station_meta):,} stations)")

        overlap_stats = audit_station_overlap(TRIP_MAIN_COMBINED, INVENTORY_MAIN_COMBINED)
    else:
        n_trip_main = int(trip_stats["trip_main_window_rows"])
        n_inv_main = int(inv_stats["inventory_main_window_rows"])
        overlap_stats = {}

    # Internal consistency check between streaming counts and filtered combined files.
    if COMBINE_DAILY_FILES:
        if n_trip_main != int(trip_stats["trip_main_window_rows"]):
            raise RuntimeError(
                "Trip main-window row count mismatch between daily audit and combined filter: "
                f"{trip_stats['trip_main_window_rows']} vs {n_trip_main}"
            )
        if n_inv_main != int(inv_stats["inventory_main_window_rows"]):
            raise RuntimeError(
                "Inventory main-window row count mismatch between daily audit and combined filter: "
                f"{inv_stats['inventory_main_window_rows']} vs {n_inv_main}"
            )

    summary: Dict[str, object] = {
        "main_window": [str(START_TS), str(END_TS)],
        "trip_context_window": [str(TRIP_CONTEXT_START), str(TRIP_CONTEXT_END)],
        "inventory_context_window": [str(INV_CONTEXT_START), str(INV_CONTEXT_END)],
        "official_sources": {
            "trips_dataset_id": TRIP_DATASET_ID,
            "historical_inventory_dataset_id": INVENTORY_DATASET_ID,
        },
        **trip_stats,
        **inv_stats,
        **overlap_stats,
        "files": {
            "trip_main": str(TRIP_MAIN_COMBINED),
            "trip_context": str(TRIP_CONTEXT_COMBINED),
            "inventory_main": str(INVENTORY_MAIN_COMBINED),
            "inventory_context": str(INVENTORY_CONTEXT_COMBINED),
            "station_metadata": str(STATION_METADATA_FILE),
        },
    }

    write_summary(summary)

    print("\n" + "=" * 100)
    print("DOWNLOAD COMPLETE")
    print("=" * 100)
    print(f"Trip main rows: {summary['trip_main_window_rows']:,}")
    print(f"Trip context rows: {summary['trip_context_rows']:,}")
    print(f"Unique bikes in trip context: {summary['unique_bikes_context']:,}")
    print(f"Missing bike_id rows: {summary['missing_bike_id_rows']:,}")
    print(f"Inventory main rows: {summary['inventory_main_window_rows']:,}")
    print(f"Inventory context rows: {summary['inventory_context_rows']:,}")
    print(f"Historical inventory stations: {summary['unique_inventory_station_ids_context']:,}")
    print(
        "Median global inventory timestamp interval (s): "
        f"{summary['median_global_inventory_timestamp_interval_seconds']}"
    )
    if overlap_stats:
        print(f"Trip/inventory station ID overlap: {summary['station_id_overlap']:,}")
        print(
            "Trip station IDs absent from inventory: "
            f"{summary['trip_station_ids_absent_from_inventory']:,}"
        )
    print(f"Summary: {SUMMARY_TXT}")
    print(f"JSON audit: {SUMMARY_JSON}")
    print("\nNext step: audit bike continuity + inventory balance BEFORE constructing a closed system.")


if __name__ == "__main__":
    main()
