# -*- coding: utf-8 -*-
"""
download_divvy_2017_june_history_for_july_cohort.py

Download all June-2017 Divvy public trips and use them only to locate the
fixed July-2017 bike cohort before t0 = 2017-07-01 00:00:00.

This script follows the same City of Chicago Socrata source and download logic
as the previous downloader, but it downloads TRIPS ONLY. No June station
inventory is needed for this step.

Existing July data are READ ONLY.

Outputs
-------
D:/physic_predict_bike/divvy_2017_physics_raw/trips/june_2017_history/

  daily/
      divvy_trips_2017-06-01.csv
      ...
      divvy_trips_2017-06-30.csv

  divvy_trips_2017_06_all.csv
  divvy_trips_2017_06_july_cohort.csv
  july_cohort_last_completed_pre_t0_trip.csv
  july_cohort_active_at_t0_from_june.csv
  july_cohort_june_history_audit.csv
  july_cohort_without_any_june_trip.csv
  june_history_summary.txt
  june_history_summary.json
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

JUNE_START = pd.Timestamp("2017-06-01 00:00:00")
T0 = pd.Timestamp("2017-07-01 00:00:00")
JULY_END = pd.Timestamp("2017-08-01 00:00:00")

ROOT = Path(r"D:/physic_predict_bike/divvy_2017_physics_raw")

# Existing July file from your previous downloader.
JULY_MAIN_PATH = ROOT / "trips" / "divvy_trips_main_window.csv"

# New June-history directory; does not overwrite the old +/-7-day context.
OUT_ROOT = ROOT / "trips" / "june_2017_history"
DAILY_DIR = OUT_ROOT / "daily"

JUNE_ALL_PATH = OUT_ROOT / "divvy_trips_2017_06_all.csv"
JUNE_JULY_COHORT_PATH = OUT_ROOT / "divvy_trips_2017_06_july_cohort.csv"
LAST_PRE_T0_PATH = OUT_ROOT / "july_cohort_last_completed_pre_t0_trip.csv"
ACTIVE_T0_PATH = OUT_ROOT / "july_cohort_active_at_t0_from_june.csv"
AUDIT_PATH = OUT_ROOT / "july_cohort_june_history_audit.csv"
NO_JUNE_HISTORY_PATH = OUT_ROOT / "july_cohort_without_any_june_trip.csv"
SUMMARY_TXT = OUT_ROOT / "june_history_summary.txt"
SUMMARY_JSON = OUT_ROOT / "june_history_summary.json"

SOCRATA_DOMAIN = "https://data.cityofchicago.org"
TRIP_DATASET_ID = "fg6s-gzvg"

PAGE_SIZE = 50_000
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 2.0

# Rerunning the script reuses valid June daily files.
REUSE_EXISTING_DAILY_FILES = True

# Optional. Do not hard-code a private token.
SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN", "").strip()

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

OUT_ROOT.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "DivvyJuneHistoryDownloader/1.0",
        "Accept": "text/csv",
    }
)
if SOCRATA_APP_TOKEN:
    SESSION.headers.update({"X-App-Token": SOCRATA_APP_TOKEN})


# =============================================================================
# HELPERS
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required existing file not found: {path}")


def normalize_id_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )
    out = out.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    return out


def socrata_csv_endpoint(dataset_id: str) -> str:
    return f"{SOCRATA_DOMAIN}/resource/{dataset_id}.csv"


def fmt_soql_timestamp(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000")


def request_csv(params: Dict[str, str]) -> pd.DataFrame:
    params = dict(params)
    url = params.pop("__url")
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

            if r.status_code == 429 or 500 <= r.status_code < 600:
                wait_s = min(RETRY_BASE_SECONDS * (2 ** (attempt - 1)), 60.0)
                print(
                    f"  HTTP {r.status_code}; retry {attempt}/{MAX_RETRIES} "
                    f"after {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                continue

            r.raise_for_status()

            if not r.content:
                return pd.DataFrame()

            return pd.read_csv(io.BytesIO(r.content), dtype="string")

        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break

            wait_s = min(RETRY_BASE_SECONDS * (2 ** (attempt - 1)), 60.0)
            print(
                f"  request error {attempt}/{MAX_RETRIES}: {exc}\n"
                f"  retry after {wait_s:.1f}s"
            )
            time.sleep(wait_s)

    note = (
        " A SOCRATA_APP_TOKEN is configured."
        if SOCRATA_APP_TOKEN
        else " No SOCRATA_APP_TOKEN is configured; adding one may help with rate limits."
    )
    raise RuntimeError(f"Socrata request failed after retries.{note}") from last_error


def daterange_days(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> Iterable[pd.Timestamp]:
    day = start.normalize()
    final_day = (end_exclusive - pd.Timedelta(microseconds=1)).normalize()
    while day <= final_day:
        yield day
        day += pd.Timedelta(days=1)


def download_one_day(day_start: pd.Timestamp, day_end: pd.Timestamp, output_file: Path) -> pd.DataFrame:
    if REUSE_EXISTING_DAILY_FILES and output_file.exists() and output_file.stat().st_size > 0:
        try:
            old = pd.read_csv(output_file, dtype="string")
            missing = set(TRIP_COLUMNS) - set(old.columns)
            if missing:
                raise ValueError(f"missing columns {sorted(missing)}")
            print(f"  reuse {output_file.name}: {len(old):,} rows")
            return old
        except Exception as exc:
            print(f"  existing file invalid; redownloading {output_file.name}: {exc}")

    where = (
        f"start_time >= '{fmt_soql_timestamp(day_start)}' AND "
        f"start_time < '{fmt_soql_timestamp(day_end)}'"
    )

    pieces: List[pd.DataFrame] = []
    offset = 0
    page_no = 0

    while True:
        page_no += 1
        params = {
            "__url": socrata_csv_endpoint(TRIP_DATASET_ID),
            "$select": ",".join(TRIP_COLUMNS),
            "$where": where,
            "$order": "start_time ASC,trip_id ASC",
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        }

        df = request_csv(params)
        if df.empty:
            break

        missing = set(TRIP_COLUMNS) - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Dataset {TRIP_DATASET_ID} did not return expected columns: {sorted(missing)}"
            )

        df = df.loc[:, TRIP_COLUMNS].copy()
        pieces.append(df)

        print(
            f"  {day_start.date()} page {page_no}: {len(df):,} rows "
            f"(offset {offset:,})"
        )

        if len(df) < PAGE_SIZE:
            break
        offset += len(df)

    day_df = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=TRIP_COLUMNS)
    )

    before = len(day_df)
    day_df = day_df.drop_duplicates().reset_index(drop=True)
    if len(day_df) != before:
        print(f"  removed {before - len(day_df):,} exact duplicate rows")

    day_df.to_csv(output_file, index=False)
    return day_df


def combine_csv_files(files: Sequence[Path], output_file: Path) -> None:
    files = [Path(f) for f in files if Path(f).exists() and Path(f).stat().st_size > 0]
    if not files:
        raise RuntimeError("No June daily files available to combine.")

    if output_file.exists():
        output_file.unlink()

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


# =============================================================================
# JULY COHORT
# =============================================================================

def load_july_cohort() -> Set[str]:
    banner("LOAD EXISTING JULY BIKE COHORT")
    require_file(JULY_MAIN_PATH)

    header = pd.read_csv(JULY_MAIN_PATH, nrows=0).columns.tolist()
    missing = {"bike_id", "start_time"} - set(header)
    if missing:
        raise ValueError(f"July main file missing columns: {sorted(missing)}")

    bikes: Set[str] = set()
    july_rows = 0

    for chunk in pd.read_csv(
        JULY_MAIN_PATH,
        dtype="string",
        usecols=["bike_id", "start_time"],
        chunksize=250_000,
    ):
        chunk["bike_id"] = normalize_id_series(chunk["bike_id"])
        tt = pd.to_datetime(chunk["start_time"], errors="coerce")
        mask = (
            (tt >= T0)
            & (tt < JULY_END)
            & chunk["bike_id"].notna()
        )
        july_rows += int(mask.sum())
        bikes.update(chunk.loc[mask, "bike_id"].astype(str).tolist())

    if not bikes:
        raise RuntimeError("No July bike IDs found.")

    print(f"July rows: {july_rows:,}")
    print(f"Fixed July bike cohort: {len(bikes):,}")
    return bikes


# =============================================================================
# DOWNLOAD JUNE
# =============================================================================

def download_june() -> List[Path]:
    banner("DOWNLOAD COMPLETE JUNE 2017 TRIP HISTORY")
    print(f"Window: {JUNE_START} <= start_time < {T0}")
    print(f"Official trip dataset: {TRIP_DATASET_ID}")

    daily_files: List[Path] = []
    total_rows = 0
    unique_bikes: Set[str] = set()
    missing_bike_rows = 0

    for day in daterange_days(JUNE_START, T0):
        day_end = min(day + pd.Timedelta(days=1), T0)
        out_file = DAILY_DIR / f"divvy_trips_{day.strftime('%Y-%m-%d')}.csv"

        df = download_one_day(day, day_end, out_file)
        daily_files.append(out_file)
        total_rows += len(df)

        if len(df):
            bike = normalize_id_series(df["bike_id"])
            missing_bike_rows += int(bike.isna().sum())
            unique_bikes.update(bike.dropna().astype(str).tolist())

    if total_rows == 0:
        raise RuntimeError("No June trip rows downloaded.")

    if missing_bike_rows > 0:
        raise RuntimeError(
            f"June data contain {missing_bike_rows:,} rows with missing bike_id."
        )

    print(f"June rows downloaded: {total_rows:,}")
    print(f"Unique June bikes: {len(unique_bikes):,}")
    return daily_files


# =============================================================================
# FILTER TO JULY COHORT
# =============================================================================

def filter_june_to_july_cohort(july_bikes: Set[str]) -> Dict[str, int]:
    banner("FILTER JUNE HISTORY TO JULY COHORT")

    if JUNE_JULY_COHORT_PATH.exists():
        JUNE_JULY_COHORT_PATH.unlink()

    wrote_header = False
    june_rows = 0
    cohort_rows = 0
    june_bikes: Set[str] = set()
    cohort_bikes: Set[str] = set()

    for chunk in pd.read_csv(JUNE_ALL_PATH, dtype="string", chunksize=250_000):
        chunk["bike_id"] = normalize_id_series(chunk["bike_id"])
        june_rows += len(chunk)
        june_bikes.update(chunk["bike_id"].dropna().astype(str).tolist())

        mask = chunk["bike_id"].isin(july_bikes)
        out = chunk.loc[mask].copy()
        if out.empty:
            continue

        cohort_rows += len(out)
        cohort_bikes.update(out["bike_id"].dropna().astype(str).tolist())

        out.to_csv(
            JUNE_JULY_COHORT_PATH,
            mode="a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

    if not wrote_header:
        pd.DataFrame(columns=TRIP_COLUMNS).to_csv(JUNE_JULY_COHORT_PATH, index=False)

    no_june = july_bikes - cohort_bikes

    print(f"All June rows: {june_rows:,}")
    print(f"June rows belonging to July cohort: {cohort_rows:,}")
    print(f"July bikes seen somewhere in June: {len(cohort_bikes):,}/{len(july_bikes):,}")
    print(f"July bikes with NO June trip: {len(no_june):,}")

    return {
        "june_rows": int(june_rows),
        "unique_june_bikes": int(len(june_bikes)),
        "june_rows_for_july_cohort": int(cohort_rows),
        "july_bikes_found_in_june": int(len(cohort_bikes)),
        "july_bikes_without_any_june_trip": int(len(no_june)),
    }


# =============================================================================
# PRE-t0 LOCATION AUDIT
# =============================================================================

def build_pre_t0_audit(july_bikes: Set[str]) -> Dict[str, int]:
    banner("BUILD JULY-COHORT PRE-t0 LOCATION AUDIT")

    june = pd.read_csv(JUNE_JULY_COHORT_PATH, dtype="string", low_memory=False)

    for c in ["trip_id", "bike_id", "from_station_id", "to_station_id"]:
        june[c] = normalize_id_series(june[c])

    june["start_time"] = pd.to_datetime(june["start_time"], errors="coerce")
    june["stop_time"] = pd.to_datetime(june["stop_time"], errors="coerce")
    june["valid_time"] = (
        june["start_time"].notna()
        & june["stop_time"].notna()
        & (june["stop_time"] >= june["start_time"])
    )

    invalid_rows = int((~june["valid_time"]).sum())
    valid = june[june["valid_time"]].copy()

    # A. Any June-started trip crossing t0.
    active = valid[
        (valid["start_time"] <= T0)
        & (valid["stop_time"] > T0)
    ].copy()

    active["duration_minutes"] = (
        active["stop_time"] - active["start_time"]
    ).dt.total_seconds() / 60.0

    active = active.sort_values(
        ["bike_id", "start_time", "stop_time", "trip_id"]
    ).reset_index(drop=True)
    active.to_csv(ACTIVE_T0_PATH, index=False)

    # B. Last completed June trip for each July bike.
    completed = valid[valid["stop_time"] <= T0].copy()

    if len(completed):
        last_pre = (
            completed.sort_values(
                ["bike_id", "stop_time", "start_time", "trip_id"]
            )
            .groupby("bike_id", as_index=False, sort=False)
            .tail(1)
            .sort_values("bike_id", kind="stable")
            .reset_index(drop=True)
        )
    else:
        last_pre = pd.DataFrame(columns=valid.columns)

    last_pre.to_csv(LAST_PRE_T0_PATH, index=False)

    # C. All June activity counts for July bikes.
    agg = (
        valid.groupby("bike_id", as_index=False)
        .agg(
            june_trip_count=("trip_id", "count"),
            first_june_start_time=("start_time", "min"),
            last_june_stop_time=("stop_time", "max"),
        )
    )

    last_cols = last_pre[
        [
            "bike_id",
            "trip_id",
            "start_time",
            "stop_time",
            "from_station_id",
            "to_station_id",
        ]
    ].rename(
        columns={
            "trip_id": "last_pre_t0_trip_id",
            "start_time": "last_pre_t0_start_time",
            "stop_time": "last_pre_t0_stop_time",
            "from_station_id": "last_pre_t0_from_station_id",
            "to_station_id": "last_pre_t0_to_station_id",
        }
    )

    if len(active):
        active_one = (
            active.sort_values(["bike_id", "start_time", "stop_time", "trip_id"])
            .groupby("bike_id", as_index=False, sort=False)
            .tail(1)
            [
                [
                    "bike_id",
                    "trip_id",
                    "start_time",
                    "stop_time",
                    "from_station_id",
                    "to_station_id",
                    "duration_minutes",
                ]
            ]
            .rename(
                columns={
                    "trip_id": "active_t0_trip_id",
                    "start_time": "active_t0_start_time",
                    "stop_time": "active_t0_stop_time",
                    "from_station_id": "active_t0_from_station_id",
                    "to_station_id": "active_t0_to_station_id",
                    "duration_minutes": "active_t0_duration_minutes",
                }
            )
        )
    else:
        active_one = pd.DataFrame(
            columns=[
                "bike_id",
                "active_t0_trip_id",
                "active_t0_start_time",
                "active_t0_stop_time",
                "active_t0_from_station_id",
                "active_t0_to_station_id",
                "active_t0_duration_minutes",
            ]
        )

    cohort_df = pd.DataFrame(
        {
            "bike_id": sorted(
                july_bikes,
                key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
            )
        }
    )

    audit = cohort_df.merge(agg, on="bike_id", how="left", validate="one_to_one")
    audit = audit.merge(last_cols, on="bike_id", how="left", validate="one_to_one")
    audit = audit.merge(active_one, on="bike_id", how="left", validate="one_to_one")

    audit["has_any_june_trip"] = audit["june_trip_count"].notna()
    audit["has_completed_pre_t0_trip"] = audit["last_pre_t0_trip_id"].notna()
    audit["active_at_t0_from_june_history"] = audit["active_t0_trip_id"].notna()
    audit["june_trip_count"] = audit["june_trip_count"].fillna(0).astype(int)

    audit["t0_location_evidence_type"] = np.select(
        [
            audit["active_at_t0_from_june_history"],
            audit["has_completed_pre_t0_trip"],
            audit["has_any_june_trip"],
        ],
        [
            "active_user_trip_at_t0",
            "last_completed_june_trip_destination",
            "june_trip_exists_but_no_completed_pre_t0_trip",
        ],
        default="no_june_trip_history",
    )

    audit.to_csv(AUDIT_PATH, index=False)

    no_history = audit[~audit["has_any_june_trip"]].copy()
    no_history.to_csv(NO_JUNE_HISTORY_PATH, index=False)

    print(f"Invalid June trip-time rows: {invalid_rows:,}")
    print(f"July bikes with any June trip: {int(audit['has_any_june_trip'].sum()):,}")
    print(
        "July bikes with completed pre-t0 trip: "
        f"{int(audit['has_completed_pre_t0_trip'].sum()):,}"
    )
    print(
        "July bikes active in a June-started trip at t0: "
        f"{int(audit['active_at_t0_from_june_history'].sum()):,}"
    )
    print(f"July bikes with no June trip history: {len(no_history):,}")

    return {
        "invalid_june_trip_time_rows": invalid_rows,
        "july_bikes_with_any_june_trip": int(audit["has_any_june_trip"].sum()),
        "july_bikes_with_completed_pre_t0_trip": int(
            audit["has_completed_pre_t0_trip"].sum()
        ),
        "july_bikes_active_at_t0_from_june_history": int(
            audit["active_at_t0_from_june_history"].sum()
        ),
        "july_bikes_without_any_june_trip_history": int(len(no_history)),
    }


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary(
    july_bike_count: int,
    download_stats: Dict[str, int],
    audit_stats: Dict[str, int],
) -> None:
    summary = {
        "download_window": [str(JUNE_START), str(T0)],
        "july_cohort_size": int(july_bike_count),
        "official_trip_dataset_id": TRIP_DATASET_ID,
        **download_stats,
        **audit_stats,
        "files": {
            "june_all": str(JUNE_ALL_PATH),
            "june_july_cohort": str(JUNE_JULY_COHORT_PATH),
            "last_completed_pre_t0": str(LAST_PRE_T0_PATH),
            "active_at_t0": str(ACTIVE_T0_PATH),
            "history_audit": str(AUDIT_PATH),
            "no_june_history": str(NO_JUNE_HISTORY_PATH),
        },
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "DIVVY JUNE-2017 HISTORY FOR JULY COHORT",
        "=" * 90,
        "",
        f"June window: {JUNE_START} <= start_time < {T0}",
        f"July cohort size: {july_bike_count}",
        "",
        "DOWNLOAD",
    ]

    for k, v in download_stats.items():
        lines.append(f"{k}: {v}")

    lines.extend(["", "PRE-t0 LOCATION AUDIT"])
    for k, v in audit_stats.items():
        lines.append(f"{k}: {v}")

    lines.extend(
        [
            "",
            "IMPORTANT",
            "A completed June trip with stop_time <= t0 provides a pre-t0 station anchor.",
            "A trip with start_time <= t0 < stop_time is treated as active at t0.",
            "This script downloads no June station inventory; it only improves July bike-ID initialization.",
            "",
            f"All June trips: {JUNE_ALL_PATH}",
            f"June trips of July cohort: {JUNE_JULY_COHORT_PATH}",
            f"Last completed pre-t0 trips: {LAST_PRE_T0_PATH}",
            f"Active at t0: {ACTIVE_T0_PATH}",
            f"Per-bike audit: {AUDIT_PATH}",
        ]
    )

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    banner("DIVVY JUNE-2017 HISTORY DOWNLOAD FOR JULY COHORT")

    print(f"Existing July main file: {JULY_MAIN_PATH}")
    print(f"June output root: {OUT_ROOT}")
    print(f"Socrata token configured: {bool(SOCRATA_APP_TOKEN)}")
    print("EXISTING JULY FILES ARE READ ONLY.")

    # 1. Existing July cohort.
    july_bikes = load_july_cohort()

    # 2. Download every June trip.
    daily_files = download_june()

    # 3. Combine daily files.
    banner("COMBINE JUNE DAILY FILES")
    combine_csv_files(daily_files, JUNE_ALL_PATH)
    print(f"Combined June file: {JUNE_ALL_PATH}")

    # 4. Retain only June history of the July cohort.
    download_stats = filter_june_to_july_cohort(july_bikes)

    # 5. Determine the last pre-t0 position and t0-active bikes.
    audit_stats = build_pre_t0_audit(july_bikes)

    # 6. Save summary.
    write_summary(len(july_bikes), download_stats, audit_stats)

    banner("JUNE HISTORY DOWNLOAD COMPLETE")
    print(f"July cohort: {len(july_bikes):,}")
    print(f"July bikes found in June: {audit_stats['july_bikes_with_any_june_trip']:,}")
    print(
        "July bikes with completed pre-t0 trip: "
        f"{audit_stats['july_bikes_with_completed_pre_t0_trip']:,}"
    )
    print(
        "July bikes active at t0: "
        f"{audit_stats['july_bikes_active_at_t0_from_june_history']:,}"
    )
    print(
        "July bikes with NO June trip history: "
        f"{audit_stats['july_bikes_without_any_june_trip_history']:,}"
    )
    print(f"Summary: {SUMMARY_TXT}")
    print(f"Audit CSV: {AUDIT_PATH}")
    print("DONE.")


if __name__ == "__main__":
    main()
