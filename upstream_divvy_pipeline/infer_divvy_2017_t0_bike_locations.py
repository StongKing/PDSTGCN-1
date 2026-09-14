# -*- coding: utf-8 -*-
"""
infer_divvy_2017_t0_bike_locations_v6_4_prior_envelope_spatial.py

Purpose
-------
Infer the state/location of EVERY July-2017 cohort bike at

    t0 = 2017-07-01 00:00:00

using a PRIOR-ENVELOPE-CONSTRAINED SPATIAL CALIBRATION.

This is the third reconstruction philosophy after:

    V6.2:
        spatial inventory calibration + weighted temporal penalty

    V6.3:
        pure stochastic trajectory realization

V6.4 removes the arbitrary spatial/temporal trade-off weight.

For each internal ambiguous bike b:

    A_b = last completed June destination
    B_b = first July origin

and

    x_b = 0 -> choose A_b at t0
    x_b = 1 -> choose B_b at t0.

The uniform relocation-time prior gives

    p_b = P(x_b = 1)
        = (t0 - last_June_stop)
          / (first_July_start - last_June_stop).

The complete A/B assignment has mean negative log-prior probability

    NLL(x)
      = -(1/M) sum_b [
            x_b log(p_b)
            + (1-x_b) log(1-p_b)
        ].

The Monte-Carlo V6.3 audit provides an empirical upper envelope for this NLL.
By default V6.4 uses the 97.5th percentile of the pure-trajectory Monte-Carlo
distribution:

    NLL(x) <= q_0.975.

For the user's current 1,000-run audit this is approximately

    q_0.975 = 0.545855.

Optimization
------------
Stage 1:
    minimize station-level spatial L1 distance to the contemporaneous
    available_bikes spatial-share vector

subject to:
    - x_b in {0,1};
    - every bike remains at its own A_b or B_b only;
    - mean prior NLL <= Monte-Carlo threshold.

Stage 2 (lexicographic tie-break):
    keep the Stage-1 minimum spatial L1 (within numerical tolerance)
    and minimize prior NLL.

Thus there is NO arbitrary weight such as

    L_spatial + 0.25 L_temporal.

The temporal prior is instead an explicit admissibility constraint.

Capacity policy
---------------
Historical total_docks is NOT used as an assignment constraint in V6.4.
It is retained for final audit only.

This is deliberate:
    - V6.2 showed that raw total_docks conflicts with some trajectory evidence;
    - capacity-floor repair would introduce another modeling layer;
    - V6.4 is designed to isolate the trade-off between trajectory-prior
      plausibility and contemporaneous inventory spatial information.

available_bikes policy
----------------------
available_bikes is:
    - NOT a station-wise lower bound;
    - NOT proportionally scaled from 4,437 to 5,208;
    - used only as a normalized station spatial target.

Conditional relocation timestamps
----------------------------------
After the optimized A/B side is fixed, one actual relocation timestamp is
sampled uniformly CONDITIONAL on that side:

    choose B -> Uniform(last_June_stop, t0)
    choose A -> Uniform(t0, first_July_start)

The sampled timestamp is written to output and should be reused downstream.
It must NOT be resampled later for the same A->B relocation.

Inputs
------
Raw data:
D:/physic_predict_bike/divvy_2017_physics_raw/
    trips/divvy_trips_main_window.csv
    trips/divvy_trips_context.csv
    trips/june_2017_history/july_cohort_last_completed_pre_t0_trip.csv
    station_inventory/divvy_station_inventory_main_window.csv

Monte-Carlo audit (preferred threshold source):
D:/physic_predict_bike/divvy_2017_t0_monte_carlo_audit/
    monte_carlo_realizations_v6_3.csv

Outputs
-------
D:/physic_predict_bike/divvy_2017_physics_closed_active_v6_4_prior_envelope_t0/

    t0_bike_locations_v6_4.csv
    t0_station_inventory_v6_4.csv
    t0_active_user_trip_bikes_v6_4.csv
    ambiguous_bike_decisions_v6_4.csv
    sampled_pre_t0_internal_relocations_v6_4.csv
    sampled_post_t0_pre_first_internal_relocations_v6_4.csv
    implied_boundary_pre_t0_relocations_v6_4.csv
    t0_available_spatial_reference_v6_4.csv
    optimization_audit_v6_4.csv
    summary_v6_4.txt
    summary_v6_4.json

Raw inputs are READ ONLY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_ROOT = Path(r"D:/physic_predict_bike/divvy_2017_physics_raw")

JULY_MAIN_PATH = RAW_ROOT / "trips" / "divvy_trips_main_window.csv"
TRIP_CONTEXT_PATH = RAW_ROOT / "trips" / "divvy_trips_context.csv"
JUNE_LAST_PRE_PATH = (
    RAW_ROOT
    / "trips"
    / "june_2017_history"
    / "july_cohort_last_completed_pre_t0_trip.csv"
)
INVENTORY_MAIN_PATH = (
    RAW_ROOT
    / "station_inventory"
    / "divvy_station_inventory_main_window.csv"
)

OUT_ROOT = Path(
    r"D:/physic_predict_bike/divvy_2017_physics_closed_active_v6_4_prior_envelope_t0"
)

MC_REALIZATIONS_PATH = Path(
    r"D:/physic_predict_bike/divvy_2017_t0_monte_carlo_audit/"
    r"monte_carlo_realizations_v6_3.csv"
)

T0 = pd.Timestamp("2017-07-01 00:00:00")
JULY_END = pd.Timestamp("2017-08-01 00:00:00")

# Use the empirical 97.5th percentile from the V6.3 Monte-Carlo distribution.
NLL_QUANTILE = 0.975

# Fallback only if the Monte-Carlo realization file is unavailable.
# This is the value obtained from the user's current 1,000-run audit.
FALLBACK_NLL_THRESHOLD = 0.545855

# Numerical clipping only for log(p), not for the actual probability model.
LOG_EPS = 1e-12

# Stage 2 keeps the globally minimum Stage-1 spatial L1 within this tolerance
# and minimizes NLL as a lexicographic tie-break.
LEXICOGRAPHIC_SPATIAL_TOL = 1e-9

# Reproducible draw of the actual relocation timestamp after the optimized
# pre/post-t0 side has been selected.
CONDITIONAL_TIME_RANDOM_SEED = 20260702

# Existing lifecycle rule used to reconstruct the t0 inventory reference.
LIFECYCLE_RULES = {
    "624": "zero_before_first_observation",
    "193": "zero_after_last_observation",
    "392": "zero_after_last_observation",
}


# =============================================================================
# HELPERS / INPUT PIPELINE
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

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

    out = out.str.replace(
        r"^(-?\d+)\.0$",
        r"\1",
        regex=True,
    )

    return out


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def jsonable(v):
    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, np.floating):
        return float(v)

    if isinstance(v, np.ndarray):
        return v.tolist()

    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    return v


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 3:
        return np.nan

    xx = x[m]
    yy = y[m]

    if np.std(xx) == 0 or np.std(yy) == 0:
        return np.nan

    return float(
        np.corrcoef(xx, yy)[0, 1]
    )


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def load_july_main():
    banner("LOAD JULY MAIN TRIPS")

    df = pd.read_csv(
        JULY_MAIN_PATH,
        low_memory=False,
    )

    required = [
        "trip_id",
        "bike_id",
        "start_time",
        "stop_time",
        "from_station_id",
        "to_station_id",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"July main missing columns: {missing}"
        )

    for c in [
        "trip_id",
        "bike_id",
        "from_station_id",
        "to_station_id",
    ]:
        df[c] = normalize_id_series(
            df[c]
        )

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        errors="coerce",
    )

    df["stop_time"] = pd.to_datetime(
        df["stop_time"],
        errors="coerce",
    )

    valid = (
        df["start_time"].notna()
        & df["stop_time"].notna()
        & (df["stop_time"] >= df["start_time"])
        & (df["start_time"] >= T0)
        & (df["start_time"] < JULY_END)
        & df["bike_id"].notna()
        & df["from_station_id"].notna()
        & df["to_station_id"].notna()
    )

    july = df[valid].copy()

    july_bikes = sorted(
        july[
            "bike_id"
        ]
        .astype(str)
        .unique()
        .tolist(),
        key=lambda x: (
            not x.isdigit(),
            int(x) if x.isdigit() else x,
        ),
    )

    station_ids = sorted(
        set(
            july[
                "from_station_id"
            ].astype(str)
        )
        | set(
            july[
                "to_station_id"
            ].astype(str)
        ),
        key=lambda x: (
            not x.isdigit(),
            int(x) if x.isdigit() else x,
        ),
    )

    first = (
        july.sort_values(
            [
                "bike_id",
                "start_time",
                "stop_time",
                "trip_id",
            ]
        )
        .groupby(
            "bike_id",
            as_index=False,
            sort=False,
        )
        .head(1)
        [
            [
                "bike_id",
                "trip_id",
                "start_time",
                "stop_time",
                "from_station_id",
                "to_station_id",
            ]
        ]
        .rename(
            columns={
                "trip_id": "first_july_trip_id",
                "start_time": "first_july_start_time",
                "stop_time": "first_july_stop_time",
                "from_station_id": "first_july_origin",
                "to_station_id": "first_july_destination",
            }
        )
        .reset_index(drop=True)
    )

    if len(first) != len(july_bikes):
        raise RuntimeError(
            "First-July-trip table does not contain exactly one row per July bike."
        )

    print(f"July trips: {len(july):,}")
    print(f"Fixed July cohort: {len(july_bikes):,}")
    print(f"Active station domain: {len(station_ids):,}")

    return (
        july,
        july_bikes,
        station_ids,
        first,
    )


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def load_last_pre_t0(
    july_bikes: List[str],
) -> pd.DataFrame:
    banner("LOAD COMPLETE-JUNE LAST PRE-t0 TRIP")

    df = pd.read_csv(
        JUNE_LAST_PRE_PATH,
        low_memory=False,
    )

    required = [
        "trip_id",
        "bike_id",
        "start_time",
        "stop_time",
        "from_station_id",
        "to_station_id",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "June last-pre file missing columns: "
            f"{missing}"
        )

    for c in [
        "trip_id",
        "bike_id",
        "from_station_id",
        "to_station_id",
    ]:
        df[c] = normalize_id_series(
            df[c]
        )

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        errors="coerce",
    )

    df["stop_time"] = pd.to_datetime(
        df["stop_time"],
        errors="coerce",
    )

    july_set = set(july_bikes)

    df = df[
        df["bike_id"].astype(str).isin(july_set)
        & df["stop_time"].notna()
        & (df["stop_time"] <= T0)
    ].copy()

    # Defensive one-row-per-bike re-selection.
    df = (
        df.sort_values(
            [
                "bike_id",
                "stop_time",
                "start_time",
                "trip_id",
            ]
        )
        .groupby(
            "bike_id",
            as_index=False,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    out = (
        df[
            [
                "bike_id",
                "trip_id",
                "start_time",
                "stop_time",
                "from_station_id",
                "to_station_id",
            ]
        ]
        .rename(
            columns={
                "trip_id": "last_pre_trip_id",
                "start_time": "last_pre_start_time",
                "stop_time": "last_pre_stop_time",
                "from_station_id": "last_pre_origin",
                "to_station_id": "last_pre_destination",
            }
        )
        .reset_index(drop=True)
    )

    print(
        "July bikes with completed June pre-t0 trip: "
        f"{out['bike_id'].nunique():,}"
    )

    return out


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def load_active_at_t0(
    july_bikes: List[str],
) -> pd.DataFrame:
    banner("BUILD EXACT ACTIVE-AT-t0 SET")

    df = pd.read_csv(
        TRIP_CONTEXT_PATH,
        low_memory=False,
    )

    required = [
        "trip_id",
        "bike_id",
        "start_time",
        "stop_time",
        "from_station_id",
        "to_station_id",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Trip context missing columns: {missing}"
        )

    for c in [
        "trip_id",
        "bike_id",
        "from_station_id",
        "to_station_id",
    ]:
        df[c] = normalize_id_series(
            df[c]
        )

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        errors="coerce",
    )

    df["stop_time"] = pd.to_datetime(
        df["stop_time"],
        errors="coerce",
    )

    july_set = set(july_bikes)

    active = df[
        df["bike_id"].astype(str).isin(july_set)
        & df["start_time"].notna()
        & df["stop_time"].notna()
        & (df["start_time"] <= T0)
        & (df["stop_time"] > T0)
    ].copy()

    active["duration_minutes"] = (
        active["stop_time"]
        - active["start_time"]
    ).dt.total_seconds() / 60.0

    duplicate_active = (
        active["bike_id"]
        .value_counts()
        .loc[lambda s: s > 1]
    )

    if len(duplicate_active):
        print(
            "WARNING: multiple active-at-t0 rows for "
            f"{len(duplicate_active)} bike IDs."
        )

    active_one = (
        active.sort_values(
            [
                "bike_id",
                "start_time",
                "stop_time",
                "trip_id",
            ]
        )
        .groupby(
            "bike_id",
            as_index=False,
            sort=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    started_before = int(
        (
            active_one[
                "start_time"
            ] < T0
        ).sum()
    )

    started_exactly = int(
        (
            active_one[
                "start_time"
            ] == T0
        ).sum()
    )

    print(
        f"Unique active bikes at t0: "
        f"{len(active_one):,}"
    )

    print(
        f"  started before t0: "
        f"{started_before:,}"
    )

    print(
        f"  started exactly at t0: "
        f"{started_exactly:,}"
    )

    return active_one


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def build_t0_available_reference(
    station_ids: List[str],
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Back-propagate the first July available_bikes snapshot from ~00:05:17 to t0:

        available_i(t0)
          = available_i(anchor)
            - arrivals_i(t0, anchor]
            + departures_i(t0, anchor]

    The resulting vector is a SOFT spatial reference only.
    It is NOT a lower bound for the July bike cohort.
    """
    banner("BUILD t0 AVAILABLE_BIKES SPATIAL REFERENCE")

    inv = pd.read_csv(
        INVENTORY_MAIN_PATH,
        low_memory=False,
    )

    required = [
        "id",
        "timestamp",
        "available_bikes",
    ]

    missing = [
        c for c in required
        if c not in inv.columns
    ]

    if missing:
        raise ValueError(
            f"Inventory missing columns: {missing}"
        )

    inv["id"] = normalize_id_series(
        inv["id"]
    )

    inv["timestamp"] = pd.to_datetime(
        inv["timestamp"],
        errors="coerce",
    )

    inv["available_bikes"] = pd.to_numeric(
        inv["available_bikes"],
        errors="coerce",
    )

    if "total_docks" in inv.columns:
        inv["total_docks"] = pd.to_numeric(
            inv["total_docks"],
            errors="coerce",
        )

    station_set = set(station_ids)

    inv = inv[
        inv["id"].isin(station_set)
        & inv["timestamp"].notna()
        & (inv["timestamp"] >= T0)
        & (inv["timestamp"] < JULY_END)
    ].copy()

    inv = (
        inv.sort_values(
            ["timestamp", "id"]
        )
        .drop_duplicates(
            ["timestamp", "id"],
            keep="last",
        )
        .copy()
    )

    if inv.empty:
        raise RuntimeError(
            "No July inventory rows found."
        )

    anchor_time = pd.Timestamp(
        inv["timestamp"].min()
    )

    anchor_rows = inv[
        inv["timestamp"] == anchor_time
    ].copy()

    anchor_available = (
        anchor_rows
        .set_index("id")[
            "available_bikes"
        ]
        .reindex(station_ids)
    )

    # Explicit lifecycle assumption for station 624 before its first observation.
    for sid, rule in LIFECYCLE_RULES.items():
        if (
            sid in station_set
            and rule
            == "zero_before_first_observation"
            and pd.isna(
                anchor_available.loc[sid]
            )
        ):
            anchor_available.loc[sid] = 0.0

    missing_anchor = (
        anchor_available[
            anchor_available.isna()
        ]
        .index
        .tolist()
    )

    if missing_anchor:
        raise RuntimeError(
            "Missing first-snapshot available_bikes outside "
            "the explicit lifecycle rule. "
            f"Examples={missing_anchor[:30]}"
        )

    # Capacity.
    capacity = pd.Series(
        np.nan,
        index=station_ids,
        dtype=float,
    )

    capacity_source = pd.Series(
        "missing",
        index=station_ids,
        dtype="string",
    )

    if "total_docks" in inv.columns:
        anchor_cap = (
            anchor_rows
            .set_index("id")[
                "total_docks"
            ]
            .reindex(station_ids)
        )

        for sid in station_ids:
            if pd.notna(
                anchor_cap.loc[sid]
            ):
                capacity.loc[sid] = float(
                    anchor_cap.loc[sid]
                )

                capacity_source.loc[sid] = (
                    "anchor_total_docks"
                )

            else:
                s = inv[
                    (inv["id"] == sid)
                    & inv[
                        "total_docks"
                    ].notna()
                ].sort_values(
                    "timestamp"
                )

                if len(s):
                    capacity.loc[sid] = float(
                        s.iloc[0][
                            "total_docks"
                        ]
                    )

                    capacity_source.loc[sid] = (
                        "earliest_july_total_docks_fallback"
                    )

    # Back-propagate ALL public trips during the 5-minute interval.
    context = pd.read_csv(
        TRIP_CONTEXT_PATH,
        low_memory=False,
    )

    for c in [
        "from_station_id",
        "to_station_id",
    ]:
        context[c] = normalize_id_series(
            context[c]
        )

    context["start_time"] = pd.to_datetime(
        context["start_time"],
        errors="coerce",
    )

    context["stop_time"] = pd.to_datetime(
        context["stop_time"],
        errors="coerce",
    )

    dep = context[
        context["start_time"].notna()
        & (context["start_time"] > T0)
        & (
            context[
                "start_time"
            ] <= anchor_time
        )
        & context[
            "from_station_id"
        ].isin(station_set)
    ][
        "from_station_id"
    ].value_counts()

    arr = context[
        context["stop_time"].notna()
        & (context["stop_time"] > T0)
        & (
            context[
                "stop_time"
            ] <= anchor_time
        )
        & context[
            "to_station_id"
        ].isin(station_set)
    ][
        "to_station_id"
    ].value_counts()

    rows = []

    for sid in station_ids:
        av_anchor = int(
            round(
                float(
                    anchor_available.loc[
                        sid
                    ]
                )
            )
        )

        d = int(
            dep.get(sid, 0)
        )

        a = int(
            arr.get(sid, 0)
        )

        av_t0 = (
            av_anchor
            - a
            + d
        )

        if av_t0 < 0:
            raise RuntimeError(
                "Negative available_bikes t0 reference "
                f"at station {sid}: {av_t0}"
            )

        rows.append(
            {
                "station_id": sid,
                "anchor_time": anchor_time,
                "anchor_available_bikes": av_anchor,
                "departures_t0_to_anchor": d,
                "arrivals_t0_to_anchor": a,
                "available_bikes_t0_reference": av_t0,
                "capacity_total_docks": (
                    float(
                        capacity.loc[sid]
                    )
                    if pd.notna(
                        capacity.loc[sid]
                    )
                    else np.nan
                ),
                "capacity_source": str(
                    capacity_source.loc[
                        sid
                    ]
                ),
            }
        )

    ref = pd.DataFrame(rows)

    av_total = int(
        ref[
            "available_bikes_t0_reference"
        ].sum()
    )

    if av_total <= 0:
        raise RuntimeError(
            "available_bikes t0 reference total is nonpositive."
        )

    ref[
        "available_bikes_spatial_share"
    ] = (
        ref[
            "available_bikes_t0_reference"
        ]
        / av_total
    )

    print(
        f"First inventory anchor: "
        f"{anchor_time}"
    )

    print(
        f"Anchor available total: "
        f"{ref['anchor_available_bikes'].sum():,}"
    )

    print(
        "All-trip departures t0->anchor: "
        f"{ref['departures_t0_to_anchor'].sum():,}"
    )

    print(
        "All-trip arrivals t0->anchor: "
        f"{ref['arrivals_t0_to_anchor'].sum():,}"
    )

    print(
        "t0 available reference total: "
        f"{av_total:,}"
    )

    print(
        "IMPORTANT: this is a SOFT spatial reference, "
        "not a cohort lower bound."
    )

    return (
        ref,
        anchor_time,
    )


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def build_bike_evidence(
    july_bikes: List[str],
    station_ids: List[str],
    first_july: pd.DataFrame,
    last_pre: pd.DataFrame,
    active_t0: pd.DataFrame,
) -> pd.DataFrame:
    banner("BUILD PER-BIKE CANDIDATE SETS")

    cohort = pd.DataFrame(
        {
            "bike_id": july_bikes
        }
    )

    ev = cohort.merge(
        first_july,
        on="bike_id",
        how="left",
        validate="one_to_one",
    )

    ev = ev.merge(
        last_pre,
        on="bike_id",
        how="left",
        validate="one_to_one",
    )

    active_cols = (
        active_t0[
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
                "from_station_id": "active_t0_origin",
                "to_station_id": "active_t0_destination",
                "duration_minutes": "active_t0_duration_minutes",
            }
        )
    )

    ev = ev.merge(
        active_cols,
        on="bike_id",
        how="left",
        validate="one_to_one",
    )

    station_set = set(
        station_ids
    )

    ev[
        "active_at_t0"
    ] = (
        ev[
            "active_t0_trip_id"
        ].notna()
    )

    ev[
        "has_completed_june_trip"
    ] = (
        ev[
            "last_pre_trip_id"
        ].notna()
    )

    ev[
        "last_pre_destination_inside_domain"
    ] = (
        ev[
            "last_pre_destination"
        ]
        .astype("string")
        .isin(station_set)
    )

    ev[
        "last_pre_same_as_first_july"
    ] = (
        ev[
            "last_pre_destination_inside_domain"
        ]
        & (
            ev[
                "last_pre_destination"
            ].astype("string")
            ==
            ev[
                "first_july_origin"
            ].astype("string")
        )
    )

    ev[
        "last_pre_differs_from_first_july"
    ] = (
        ev[
            "last_pre_destination_inside_domain"
        ]
        & (
            ev[
                "last_pre_destination"
            ].astype("string")
            !=
            ev[
                "first_july_origin"
            ].astype("string")
        )
    )

    ev[
        "first_use_hours_after_t0"
    ] = (
        ev[
            "first_july_start_time"
        ]
        - T0
    ).dt.total_seconds() / 3600.0

    ev[
        "evidence_type"
    ] = np.select(
        [
            ev[
                "active_at_t0"
            ],
            ev[
                "last_pre_same_as_first_july"
            ],
            ev[
                "last_pre_differs_from_first_july"
            ],
            (
                ev[
                    "has_completed_june_trip"
                ]
                & ~ev[
                    "last_pre_destination_inside_domain"
                ]
            ),
        ],
        [
            "active_user_trip_at_t0",
            "june_destination_equals_first_july_origin",
            "june_destination_vs_first_origin_ambiguous",
            "june_destination_outside_july_domain",
        ],
        default=(
            "no_june_location_first_july_origin_only"
        ),
    )

    # Candidate A/B.
    ev[
        "candidate_A"
    ] = pd.NA

    ev[
        "candidate_B"
    ] = pd.NA

    # Active -> node0.
    m = ev[
        "active_at_t0"
    ]

    ev.loc[
        m,
        "candidate_A"
    ] = "0"

    ev.loc[
        m,
        "candidate_B"
    ] = "0"

    # Same -> one station.
    m = ev[
        "last_pre_same_as_first_july"
    ]

    ev.loc[
        m,
        "candidate_A"
    ] = ev.loc[
        m,
        "first_july_origin"
    ]

    ev.loc[
        m,
        "candidate_B"
    ] = ev.loc[
        m,
        "first_july_origin"
    ]

    # Ambiguous -> exactly June destination vs first July origin.
    m = ev[
        "last_pre_differs_from_first_july"
    ]

    ev.loc[
        m,
        "candidate_A"
    ] = ev.loc[
        m,
        "last_pre_destination"
    ]

    ev.loc[
        m,
        "candidate_B"
    ] = ev.loc[
        m,
        "first_july_origin"
    ]

    # No-June / previous outside domain -> first July origin only.
    m = (
        ~ev[
            "active_at_t0"
        ]
        & ~ev[
            "last_pre_same_as_first_july"
        ]
        & ~ev[
            "last_pre_differs_from_first_july"
        ]
    )

    ev.loc[
        m,
        "candidate_A"
    ] = ev.loc[
        m,
        "first_july_origin"
    ]

    ev.loc[
        m,
        "candidate_B"
    ] = ev.loc[
        m,
        "first_july_origin"
    ]

    # Temporal prior for ambiguous bikes.
    ev[
        "relocation_probability_before_t0"
    ] = np.nan

    ambiguous = ev[
        ev[
            "last_pre_differs_from_first_july"
        ]
        & ~ev[
            "active_at_t0"
        ]
    ].copy()

    if len(ambiguous):
        denominator = (
            ambiguous[
                "first_july_start_time"
            ]
            - ambiguous[
                "last_pre_stop_time"
            ]
        ).dt.total_seconds()

        numerator = (
            T0
            - ambiguous[
                "last_pre_stop_time"
            ]
        ).dt.total_seconds()

        bad = (
            denominator <= 0
        )

        if bad.any():
            examples = (
                ambiguous.loc[
                    bad,
                    [
                        "bike_id",
                        "last_pre_stop_time",
                        "first_july_start_time",
                    ],
                ]
                .head(20)
                .to_dict(
                    "records"
                )
            )

            raise RuntimeError(
                "Ambiguous bike has nonpositive relocation window. "
                f"Examples={examples}"
            )

        p = (
            numerator
            / denominator
        )

        p = np.clip(
            p,
            0.0,
            1.0,
        )

        ev.loc[
            ambiguous.index,
            "relocation_probability_before_t0"
        ] = p.to_numpy(
            dtype=float
        )

    # Candidate validation.
    nonactive = ev[
        ~ev[
            "active_at_t0"
        ]
    ]

    bad_A = (
        ~nonactive[
            "candidate_A"
        ]
        .astype("string")
        .isin(station_set)
    )

    bad_B = (
        ~nonactive[
            "candidate_B"
        ]
        .astype("string")
        .isin(station_set)
    )

    if bad_A.any() or bad_B.any():
        raise RuntimeError(
            "Non-active bike has candidate outside the modeled domain."
        )

    print(
        ev[
            "evidence_type"
        ]
        .value_counts()
        .to_string()
    )

    actual_ambiguous_count = int(
        (
            ev["evidence_type"]
            == "june_destination_vs_first_origin_ambiguous"
        ).sum()
    )

    print(
        "Ambiguous two-station bikes used in A/B optimization: "
        f"{actual_ambiguous_count:,}"
    )

    return ev


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def build_assignment_problem(
    evidence: pd.DataFrame,
    station_ids: List[str],
):
    banner("BUILD CANDIDATE-RESTRICTED ASSIGNMENT PROBLEM")

    station_to_idx = {
        sid: j
        for j, sid in enumerate(
            station_ids
        )
    }

    fixed_counts = np.zeros(
        len(station_ids),
        dtype=np.int64,
    )

    active_count = 0

    fixed_rows = []

    ambiguous_rows = []

    for row in evidence.itertuples(
        index=False
    ):
        bike = str(
            row.bike_id
        )

        etype = str(
            row.evidence_type
        )

        A = str(
            row.candidate_A
        )

        B = str(
            row.candidate_B
        )

        if etype == "active_user_trip_at_t0":
            active_count += 1

            fixed_rows.append(
                {
                    "bike_id": bike,
                    "fixed_state": "0",
                    "evidence_type": etype,
                }
            )

            continue

        if (
            etype
            ==
            "june_destination_vs_first_origin_ambiguous"
        ):
            if A == B:
                raise RuntimeError(
                    f"Ambiguous bike {bike} has identical A/B candidates."
                )

            p = float(
                row.relocation_probability_before_t0
            )

            if not np.isfinite(p):
                raise RuntimeError(
                    f"Ambiguous bike {bike} has missing temporal probability."
                )

            ambiguous_rows.append(
                {
                    "bike_id": bike,
                    "candidate_A_june_destination": A,
                    "candidate_B_first_july_origin": B,
                    "A_index": station_to_idx[A],
                    "B_index": station_to_idx[B],
                    "last_pre_stop_time": row.last_pre_stop_time,
                    "first_july_start_time": row.first_july_start_time,
                    "relocation_probability_before_t0": p,
                    "temporal_cost_choose_A": p,
                    "temporal_cost_choose_B": 1.0 - p,
                }
            )

        else:
            # One-station fixed candidate.
            if A != B:
                raise RuntimeError(
                    f"Fixed bike {bike} unexpectedly has two candidates."
                )

            fixed_counts[
                station_to_idx[A]
            ] += 1

            fixed_rows.append(
                {
                    "bike_id": bike,
                    "fixed_state": A,
                    "evidence_type": etype,
                }
            )

    ambiguous_df = pd.DataFrame(
        ambiguous_rows
    )

    fixed_df = pd.DataFrame(
        fixed_rows
    )

    station_bikes = (
        len(evidence)
        - active_count
    )

    if (
        int(
            fixed_counts.sum()
        )
        + len(ambiguous_df)
        != station_bikes
    ):
        raise RuntimeError(
            "Fixed + ambiguous station-bike counts do not match."
        )

    print(
        f"Active node0 bikes: "
        f"{active_count:,}"
    )

    print(
        f"Fixed station bikes: "
        f"{fixed_counts.sum():,}"
    )

    print(
        f"Ambiguous station bikes: "
        f"{len(ambiguous_df):,}"
    )

    print(
        f"Total station bikes: "
        f"{station_bikes:,}"
    )

    return (
        fixed_counts,
        ambiguous_df,
        fixed_df,
        active_count,
        station_bikes,
    )


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def counts_from_choices(
    fixed_counts: np.ndarray,
    ambiguous_df: pd.DataFrame,
    x: np.ndarray,
) -> np.ndarray:
    counts = fixed_counts.astype(
        np.int64
    ).copy()

    if len(ambiguous_df) == 0:
        return counts

    x = np.asarray(
        x,
        dtype=np.int64,
    )

    A_idx = ambiguous_df[
        "A_index"
    ].to_numpy(
        dtype=np.int64
    )

    B_idx = ambiguous_df[
        "B_index"
    ].to_numpy(
        dtype=np.int64
    )

    choose_A = (
        x == 0
    )

    choose_B = (
        x == 1
    )

    np.add.at(
        counts,
        A_idx[
            choose_A
        ],
        1,
    )

    np.add.at(
        counts,
        B_idx[
            choose_B
        ],
        1,
    )

    return counts


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def spatial_l1(
    counts: np.ndarray,
    available_share: np.ndarray,
) -> float:
    station_total = float(
        np.sum(counts)
    )

    if station_total <= 0:
        return np.nan

    p_id = (
        counts.astype(float)
        / station_total
    )

    return float(
        np.sum(
            np.abs(
                p_id
                - available_share
            )
        )
    )


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def temporal_disagreement(
    ambiguous_df: pd.DataFrame,
    x: np.ndarray,
) -> float:
    if len(ambiguous_df) == 0:
        return 0.0

    p = ambiguous_df[
        "relocation_probability_before_t0"
    ].to_numpy(
        dtype=float
    )

    x = np.asarray(
        x,
        dtype=np.int64,
    )

    cost = np.where(
        x == 1,
        1.0 - p,
        p,
    )

    return float(
        np.mean(cost)
    )


# =============================================================================
# REUSED DATA/EVIDENCE HELPER
# =============================================================================

def build_capacity_arrays(
    reference: pd.DataFrame,
    station_ids: List[str],
):
    capacity_float = (
        reference
        .set_index(
            "station_id"
        )
        .reindex(
            station_ids
        )[
            "capacity_total_docks"
        ]
        .to_numpy(
            dtype=float
        )
    )

    finite_cap = np.isfinite(
        capacity_float
    )

    capacity = np.full(
        len(station_ids),
        np.iinfo(
            np.int32
        ).max,
        dtype=np.int64,
    )

    capacity[
        finite_cap
    ] = np.rint(
        capacity_float[
            finite_cap
        ]
    ).astype(
        np.int64
    )

    return (
        capacity,
        finite_cap,
    )



# =============================================================================
# PRIOR NLL
# =============================================================================

def prior_mean_nll(
    ambiguous_df: pd.DataFrame,
    x: np.ndarray,
) -> float:
    """
    Mean negative log probability of the complete A/B assignment under

        x_b ~ Bernoulli(p_b),

    where x=1 means choose B (relocation before t0).
    """
    if len(ambiguous_df) == 0:
        return 0.0

    p = ambiguous_df[
        "relocation_probability_before_t0"
    ].to_numpy(dtype=float)

    x = np.asarray(x, dtype=np.int64)

    if len(x) != len(p):
        raise ValueError("x length does not match ambiguous-bike table.")

    if not np.all((x == 0) | (x == 1)):
        raise ValueError("A/B decision vector is not binary.")

    p_log = np.clip(p, LOG_EPS, 1.0 - LOG_EPS)

    losses = -(
        x.astype(float) * np.log(p_log)
        + (1.0 - x.astype(float)) * np.log(1.0 - p_log)
    )

    return float(np.mean(losses))


def brier_score(
    ambiguous_df: pd.DataFrame,
    x: np.ndarray,
) -> float:
    if len(ambiguous_df) == 0:
        return 0.0

    p = ambiguous_df[
        "relocation_probability_before_t0"
    ].to_numpy(dtype=float)

    x = np.asarray(x, dtype=float)

    return float(np.mean((x - p) ** 2))


def load_nll_threshold() -> Tuple[float, str]:
    """
    Preferred:
        empirical NLL_QUANTILE of the existing pure-trajectory Monte Carlo.

    Fallback:
        the threshold recorded from the user's current 1,000-run audit.
    """
    banner("LOAD TEMPORAL-PRIOR NLL ENVELOPE")

    if MC_REALIZATIONS_PATH.exists():
        mc = pd.read_csv(MC_REALIZATIONS_PATH)

        if "prior_mean_NLL" not in mc.columns:
            raise ValueError(
                "Monte-Carlo file exists but does not contain "
                "'prior_mean_NLL': "
                f"{MC_REALIZATIONS_PATH}"
            )

        vals = pd.to_numeric(
            mc["prior_mean_NLL"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        if len(vals) < 100:
            raise RuntimeError(
                "Too few valid Monte-Carlo NLL realizations to define "
                f"the prior envelope: n={len(vals)}"
            )

        threshold = float(np.quantile(vals, NLL_QUANTILE))
        source = (
            f"empirical q={NLL_QUANTILE:.3f} from "
            f"{len(vals):,} Monte-Carlo realizations"
        )

        print(f"Monte-Carlo file: {MC_REALIZATIONS_PATH}")
        print(f"Valid Monte-Carlo realizations: {len(vals):,}")
        print(f"NLL quantile: {NLL_QUANTILE:.3f}")
        print(f"NLL threshold: {threshold:.12f}")

        return threshold, source

    print(
        "WARNING: Monte-Carlo realization file not found; "
        "using configured fallback threshold."
    )
    print(f"Fallback NLL threshold: {FALLBACK_NLL_THRESHOLD:.12f}")

    return (
        float(FALLBACK_NLL_THRESHOLD),
        "configured_fallback_from_previous_1000_run_audit",
    )


# =============================================================================
# GLOBAL MILP: MINIMIZE SPATIAL L1 UNDER PRIOR-NLL ENVELOPE
# =============================================================================

def solve_prior_envelope_spatial_milp(
    fixed_counts: np.ndarray,
    ambiguous_df: pd.DataFrame,
    available_share: np.ndarray,
    station_total: int,
    nll_threshold: float,
):
    """
    Two-stage lexicographic MILP.

    Variables
    ---------
    x_e in {0,1}, e=1,...,M
        x=0 -> choose A (June destination)
        x=1 -> choose B (first-July origin)

    d_i >= 0, i=1,...,N
        absolute station-share discrepancy auxiliaries.

    Station count
    -------------
    Start with every ambiguous bike at A:

        count(x) = base + C x.

    Spatial objective
    -----------------
        L1(x) = sum_i |count_i / station_total - available_share_i|.

    Temporal-prior admissibility
    ----------------------------
        mean NLL(x) <= nll_threshold.

    Because Bernoulli NLL is affine in binary x:

        NLL(x)
          = mean[-log(1-p)]
            + sum_e x_e * [log(1-p_e)-log(p_e)] / M.

    Stage 1
    -------
        globally minimize spatial L1 subject to the NLL envelope.

    Stage 2
    -------
        require spatial L1 <= Stage1 optimum + tolerance,
        then globally minimize NLL.

    No total_docks capacity constraint is imposed.
    """
    banner("SOLVE V6.4 PRIOR-ENVELOPE SPATIAL CALIBRATION")

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, hstack, identity, vstack
    except Exception as exc:
        raise ImportError(
            "V6.4 requires scipy.optimize.milp. "
            "Please use a SciPy version that provides scipy.optimize.milp."
        ) from exc

    M = len(ambiguous_df)
    N = len(fixed_counts)

    if M == 0:
        x = np.zeros(0, dtype=np.int64)
        counts = fixed_counts.copy()
        return {
            "x": x,
            "counts": counts,
            "stage1_spatial_L1": spatial_l1(counts, available_share),
            "stage1_prior_mean_NLL": 0.0,
            "final_spatial_L1": spatial_l1(counts, available_share),
            "final_prior_mean_NLL": 0.0,
            "stage1_message": "No ambiguous bikes.",
            "stage2_message": "No ambiguous bikes.",
        }

    A_idx = ambiguous_df["A_index"].to_numpy(dtype=np.int64)
    B_idx = ambiguous_df["B_index"].to_numpy(dtype=np.int64)

    # Base assignment: every ambiguous bike chooses A.
    base = fixed_counts.astype(np.int64).copy()
    np.add.at(base, A_idx, 1)

    # C maps x flips from A to B.
    row = np.concatenate([A_idx, B_idx])
    col = np.concatenate([
        np.arange(M, dtype=np.int64),
        np.arange(M, dtype=np.int64),
    ])
    data = np.concatenate([
        -np.ones(M, dtype=float),
        np.ones(M, dtype=float),
    ])

    C = coo_matrix(
        (data, (row, col)),
        shape=(N, M),
    ).tocsr()

    # ------------------------------------------------------------------
    # Spatial absolute-value constraints.
    # ------------------------------------------------------------------
    I = identity(N, dtype=float, format="csr")
    Cshare = C.astype(float) / float(station_total)

    # count/total - target <= d
    A_sp1 = hstack([Cshare, -I], format="csr")
    ub_sp1 = (
        available_share
        - base.astype(float) / float(station_total)
    )

    # target - count/total <= d
    A_sp2 = hstack([-Cshare, -I], format="csr")
    ub_sp2 = (
        base.astype(float) / float(station_total)
        - available_share
    )

    neg_inf_N = np.full(N, -np.inf, dtype=float)

    # ------------------------------------------------------------------
    # NLL constraint.
    # ------------------------------------------------------------------
    p = ambiguous_df[
        "relocation_probability_before_t0"
    ].to_numpy(dtype=float)

    p_log = np.clip(p, LOG_EPS, 1.0 - LOG_EPS)

    # NLL at x=0 (all A), averaged over bikes.
    nll_base = float(np.mean(-np.log(1.0 - p_log)))

    # Switching x_e from 0 to 1 changes per-bike NLL by:
    #   -log(p) - [-log(1-p)] = log((1-p)/p)
    nll_delta_mean = (
        np.log((1.0 - p_log) / p_log)
        / float(M)
    )

    nll_rhs = float(nll_threshold - nll_base)

    A_nll_x = coo_matrix(
        nll_delta_mean.reshape(1, -1)
    ).tocsr()
    zero_d_one = coo_matrix((1, N)).tocsr()
    A_nll = hstack(
        [A_nll_x, zero_d_one],
        format="csr",
    )

    # ------------------------------------------------------------------
    # Check that the prior envelope is feasible before optimization.
    # MAP assignment minimizes NLL bike-by-bike.
    # ------------------------------------------------------------------
    x_map = (p >= 0.5).astype(np.int64)
    map_nll = prior_mean_nll(ambiguous_df, x_map)

    print(f"NLL threshold: {nll_threshold:.12f}")
    print(f"Minimum-NLL/MAP assignment NLL: {map_nll:.12f}")
    print(f"Expected choose-B count sum(p): {float(np.sum(p)):.3f}")
    print(f"MAP choose-B count: {int(np.sum(x_map)):,}")

    if map_nll > nll_threshold + 1e-10:
        raise RuntimeError(
            "The chosen NLL envelope is infeasible even for the "
            "bike-wise MAP assignment. "
            f"MAP NLL={map_nll:.12f}, threshold={nll_threshold:.12f}"
        )

    # ------------------------------------------------------------------
    # Common variable bounds/integrality.
    # ------------------------------------------------------------------
    lower_var = np.zeros(M + N, dtype=float)
    upper_var = np.concatenate([
        np.ones(M, dtype=float),
        np.full(N, np.inf, dtype=float),
    ])
    bounds = Bounds(lower_var, upper_var)

    integrality = np.zeros(M + N, dtype=np.int8)
    integrality[:M] = 1

    # ------------------------------------------------------------------
    # STAGE 1: globally minimize spatial L1.
    # ------------------------------------------------------------------
    A_stage1 = vstack(
        [A_sp1, A_sp2, A_nll],
        format="csr",
    )
    lb_stage1 = np.concatenate([
        neg_inf_N,
        neg_inf_N,
        np.array([-np.inf], dtype=float),
    ])
    ub_stage1 = np.concatenate([
        ub_sp1,
        ub_sp2,
        np.array([nll_rhs], dtype=float),
    ])

    constraints1 = LinearConstraint(
        A_stage1,
        lb_stage1,
        ub_stage1,
    )

    c1 = np.zeros(M + N, dtype=float)
    c1[M:] = 1.0

    result1 = milp(
        c=c1,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints1,
        options={"disp": False},
    )

    if result1.x is None or not result1.success:
        raise RuntimeError(
            "Stage-1 MILP failed: "
            f"{result1.message}"
        )

    x1 = np.rint(result1.x[:M]).astype(np.int64)

    if not np.all((x1 == 0) | (x1 == 1)):
        raise RuntimeError("Stage-1 MILP returned a non-binary A/B decision.")

    counts1 = counts_from_choices(
        fixed_counts,
        ambiguous_df,
        x1,
    )

    stage1_l1 = spatial_l1(
        counts1,
        available_share,
    )
    stage1_nll = prior_mean_nll(
        ambiguous_df,
        x1,
    )

    if stage1_nll > nll_threshold + 1e-7:
        raise RuntimeError(
            "Stage-1 solution violates the NLL envelope after recomputation: "
            f"{stage1_nll:.12f} > {nll_threshold:.12f}"
        )

    print()
    print("STAGE 1 COMPLETE")
    print(f"  message: {result1.message}")
    print(f"  spatial L1 optimum: {stage1_l1:.12f}")
    print(f"  prior mean NLL: {stage1_nll:.12f}")
    print(f"  choose B: {int(np.sum(x1 == 1)):,}")
    print(f"  choose A: {int(np.sum(x1 == 0)):,}")

    # ------------------------------------------------------------------
    # STAGE 2: among Stage-1 spatial optima, minimize NLL.
    # ------------------------------------------------------------------
    # sum d_i <= stage1_l1 + tolerance
    zero_x_one = coo_matrix((1, M)).tocsr()
    one_d = coo_matrix(
        np.ones((1, N), dtype=float)
    ).tocsr()
    A_l1_cap = hstack(
        [zero_x_one, one_d],
        format="csr",
    )

    spatial_cap = float(stage1_l1 + LEXICOGRAPHIC_SPATIAL_TOL)

    A_stage2 = vstack(
        [A_sp1, A_sp2, A_nll, A_l1_cap],
        format="csr",
    )
    lb_stage2 = np.concatenate([
        neg_inf_N,
        neg_inf_N,
        np.array([-np.inf, -np.inf], dtype=float),
    ])
    ub_stage2 = np.concatenate([
        ub_sp1,
        ub_sp2,
        np.array([nll_rhs, spatial_cap], dtype=float),
    ])

    constraints2 = LinearConstraint(
        A_stage2,
        lb_stage2,
        ub_stage2,
    )

    # NLL objective. Constant nll_base can be omitted.
    c2 = np.zeros(M + N, dtype=float)
    c2[:M] = nll_delta_mean

    result2 = milp(
        c=c2,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints2,
        options={"disp": False},
    )

    if result2.x is None or not result2.success:
        raise RuntimeError(
            "Stage-2 lexicographic MILP failed: "
            f"{result2.message}"
        )

    x2 = np.rint(result2.x[:M]).astype(np.int64)

    if not np.all((x2 == 0) | (x2 == 1)):
        raise RuntimeError("Stage-2 MILP returned a non-binary A/B decision.")

    counts2 = counts_from_choices(
        fixed_counts,
        ambiguous_df,
        x2,
    )

    final_l1 = spatial_l1(
        counts2,
        available_share,
    )
    final_nll = prior_mean_nll(
        ambiguous_df,
        x2,
    )

    if final_l1 > spatial_cap + 1e-7:
        raise RuntimeError(
            "Stage-2 solution exceeds the Stage-1 spatial optimum cap: "
            f"{final_l1:.12f} > {spatial_cap:.12f}"
        )

    if final_nll > nll_threshold + 1e-7:
        raise RuntimeError(
            "Stage-2 solution violates the NLL envelope: "
            f"{final_nll:.12f} > {nll_threshold:.12f}"
        )

    print()
    print("STAGE 2 COMPLETE")
    print(f"  message: {result2.message}")
    print(f"  spatial L1: {final_l1:.12f}")
    print(f"  prior mean NLL: {final_nll:.12f}")
    print(f"  NLL threshold slack: {nll_threshold - final_nll:.12f}")
    print(f"  choose B: {int(np.sum(x2 == 1)):,}")
    print(f"  choose A: {int(np.sum(x2 == 0)):,}")

    # Exact mass check.
    if int(np.sum(counts2)) != int(station_total):
        raise RuntimeError("V6.4 station-bike mass changed during optimization.")

    return {
        "x": x2,
        "counts": counts2,
        "stage1_x": x1,
        "stage1_counts": counts1,
        "stage1_spatial_L1": stage1_l1,
        "stage1_prior_mean_NLL": stage1_nll,
        "final_spatial_L1": final_l1,
        "final_prior_mean_NLL": final_nll,
        "map_prior_mean_NLL": map_nll,
        "map_choose_B": int(np.sum(x_map)),
        "stage1_message": str(result1.message),
        "stage2_message": str(result2.message),
    }


# =============================================================================
# SAMPLE ACTUAL RELOCATION TIME CONDITIONAL ON OPTIMIZED SIDE
# =============================================================================

def sample_conditional_relocation_times(
    ambiguous_df: pd.DataFrame,
    x: np.ndarray,
    random_seed: int,
) -> pd.DataFrame:
    """
    The optimized A/B side determines whether the latent relocation occurred
    before or after t0.

    To create one concrete relocation timestamp for downstream trajectory
    construction, sample uniformly inside the chosen side:

        x=1 (B at t0):
            T_R ~ Uniform(last_June_stop, t0)

        x=0 (A at t0):
            T_R ~ Uniform(t0, first_July_start)

    total_docks and available_bikes do not affect this within-side timestamp.
    """
    banner("SAMPLE CONDITIONAL RELOCATION TIMESTAMPS")

    out = ambiguous_df.copy().reset_index(drop=True)
    M = len(out)

    x = np.asarray(x, dtype=np.int64)

    if M != len(x):
        raise ValueError("Decision vector length does not match ambiguous table.")

    if M == 0:
        out["choice_binary_x"] = np.array([], dtype=np.int64)
        out["conditional_uniform_draw_u"] = np.array([], dtype=float)
        out["sampled_relocation_time"] = pd.to_datetime([])
        out["chosen_state_at_t0"] = pd.Series(dtype="string")
        out["relocation_side"] = pd.Series(dtype="string")
        return out

    start = pd.to_datetime(
        out["last_pre_stop_time"],
        errors="coerce",
    )
    end = pd.to_datetime(
        out["first_july_start_time"],
        errors="coerce",
    )

    if start.isna().any() or end.isna().any():
        raise RuntimeError("Ambiguous relocation window contains missing timestamps.")

    start_ns = start.astype("int64").to_numpy(dtype=np.int64)
    end_ns = end.astype("int64").to_numpy(dtype=np.int64)
    t0_ns = int(T0.value)

    rng = np.random.default_rng(random_seed)
    u = rng.random(M)

    sampled_ns = np.empty(M, dtype=np.int64)

    for e in range(M):
        if x[e] == 1:
            # Strictly before t0 in the continuous model.
            lo = int(start_ns[e])
            hi = int(min(t0_ns, end_ns[e]))

            if hi <= lo:
                raise RuntimeError(
                    "V6.4 chose B for a bike with no positive-length "
                    "pre-t0 relocation interval. "
                    f"bike_id={out.iloc[e]['bike_id']}"
                )

            span = hi - lo
            offset = int(np.floor(u[e] * float(span)))
            sampled_ns[e] = min(lo + offset, hi - 1)

        else:
            # Strictly after t0.
            lo = int(max(t0_ns + 1, start_ns[e]))
            hi = int(end_ns[e])

            if hi <= lo:
                raise RuntimeError(
                    "V6.4 chose A for a bike with no positive-length "
                    "post-t0 relocation interval. "
                    f"bike_id={out.iloc[e]['bike_id']}"
                )

            span = hi - lo
            offset = int(np.floor(u[e] * float(span)))
            sampled_ns[e] = min(lo + offset, hi - 1)

    sampled_time = pd.to_datetime(sampled_ns)

    out["choice_binary_x"] = x
    out["conditional_uniform_draw_u"] = u
    out["sampled_relocation_time"] = sampled_time
    out["chosen_state_at_t0"] = np.where(
        x == 1,
        out["candidate_B_first_july_origin"].astype(str),
        out["candidate_A_june_destination"].astype(str),
    )
    out["relocation_side"] = np.where(
        x == 1,
        "pre_t0",
        "post_t0_pre_first",
    )

    # Side/time consistency.
    pre_mask = x == 1
    post_mask = x == 0

    if np.any(pre_mask & (sampled_ns >= t0_ns)):
        raise RuntimeError("Pre-t0 optimized relocation got a non-pre-t0 timestamp.")

    if np.any(post_mask & (sampled_ns <= t0_ns)):
        raise RuntimeError("Post-t0 optimized relocation got a non-post-t0 timestamp.")

    print(f"Conditional-time random seed: {random_seed}")
    print(f"Pre-t0 conditional timestamps: {int(np.sum(pre_mask)):,}")
    print(f"Post-t0 conditional timestamps: {int(np.sum(post_mask)):,}")
    print(f"Identity: {int(np.sum(pre_mask)):,} + {int(np.sum(post_mask)):,} = {M:,}")

    return out


# =============================================================================
# BUILD FINAL V6.4 OUTPUTS
# =============================================================================

def build_v64_outputs(
    evidence: pd.DataFrame,
    station_ids: List[str],
    reference: pd.DataFrame,
    fixed_counts: np.ndarray,
    ambiguous_decisions: pd.DataFrame,
    x: np.ndarray,
    final_counts: np.ndarray,
    observed_capacity: np.ndarray,
    finite_cap: np.ndarray,
):
    banner("BUILD FINAL V6.4 t0 OUTPUTS")

    chosen_map: Dict[str, str] = {}
    sampled_time_map: Dict[str, pd.Timestamp] = {}
    conditional_u_map: Dict[str, float] = {}

    for row in ambiguous_decisions.itertuples(index=False):
        bike = str(row.bike_id)
        chosen_map[bike] = str(row.chosen_state_at_t0)
        sampled_time_map[bike] = pd.Timestamp(row.sampled_relocation_time)
        conditional_u_map[bike] = float(row.conditional_uniform_draw_u)

    ev = evidence.copy()

    states = []
    assignment_source = []
    sampled_relocation_times = []
    conditional_draws = []

    for row in ev.itertuples(index=False):
        bike = str(row.bike_id)
        etype = str(row.evidence_type)

        if etype == "active_user_trip_at_t0":
            state = "0"
            src = "active_user_trip_at_t0"
            rt = pd.NaT
            uu = np.nan

        elif etype == "june_destination_vs_first_origin_ambiguous":
            state = chosen_map[bike]
            src = "v6_4_prior_envelope_spatial_calibration"
            rt = sampled_time_map[bike]
            uu = conditional_u_map[bike]

        else:
            state = str(row.candidate_A)
            src = "fixed_single_candidate"
            rt = pd.NaT
            uu = np.nan

        states.append(state)
        assignment_source.append(src)
        sampled_relocation_times.append(rt)
        conditional_draws.append(uu)

    ev["t0_state"] = states
    ev["t0_assignment_source"] = assignment_source
    ev["sampled_relocation_time"] = sampled_relocation_times
    ev["conditional_uniform_draw_u"] = conditional_draws
    ev["t0_is_user_trip_node"] = ev["t0_state"] == "0"
    ev["t0_station_id"] = ev["t0_state"].where(
        ev["t0_state"] != "0",
        pd.NA,
    )

    # Hard no-third-station invariant.
    for row in ev[~ev["t0_is_user_trip_node"]].itertuples(index=False):
        state = str(row.t0_state)
        A = str(row.candidate_A)
        B = str(row.candidate_B)

        if state not in {A, B}:
            raise RuntimeError(
                f"Bike {row.bike_id} assigned to unsupported third station {state}."
            )

    amb = ev[
        ev["evidence_type"]
        == "june_destination_vs_first_origin_ambiguous"
    ].copy()

    pre_internal = amb[
        amb["t0_station_id"].astype("string")
        == amb["first_july_origin"].astype("string")
    ][
        [
            "bike_id",
            "last_pre_trip_id",
            "last_pre_stop_time",
            "last_pre_destination",
            "first_july_trip_id",
            "first_july_start_time",
            "first_july_origin",
            "t0_station_id",
            "relocation_probability_before_t0",
            "conditional_uniform_draw_u",
            "sampled_relocation_time",
        ]
    ].copy()

    pre_internal["source_station_id"] = pre_internal["last_pre_destination"]
    pre_internal["destination_station_id"] = pre_internal["first_july_origin"]
    pre_internal["relocation_side"] = "pre_t0"

    post_internal = amb[
        amb["t0_station_id"].astype("string")
        == amb["last_pre_destination"].astype("string")
    ][
        [
            "bike_id",
            "last_pre_trip_id",
            "last_pre_stop_time",
            "last_pre_destination",
            "first_july_trip_id",
            "first_july_start_time",
            "first_july_origin",
            "t0_station_id",
            "relocation_probability_before_t0",
            "conditional_uniform_draw_u",
            "sampled_relocation_time",
        ]
    ].copy()

    post_internal["source_station_id"] = post_internal["last_pre_destination"]
    post_internal["destination_station_id"] = post_internal["first_july_origin"]
    post_internal["window_start"] = T0
    post_internal["window_end"] = post_internal["first_july_start_time"]
    post_internal["relocation_side"] = "post_t0_pre_first"

    if len(pre_internal) and not (
        pd.to_datetime(pre_internal["sampled_relocation_time"]) < T0
    ).all():
        raise RuntimeError("Pre-t0 relocation table contains an invalid timestamp.")

    if len(post_internal) and not (
        pd.to_datetime(post_internal["sampled_relocation_time"]) > T0
    ).all():
        raise RuntimeError("Post-t0 relocation table contains an invalid timestamp.")

    boundary_pre = ev[
        ev["evidence_type"]
        == "june_destination_outside_july_domain"
    ][
        [
            "bike_id",
            "last_pre_trip_id",
            "last_pre_stop_time",
            "last_pre_destination",
            "first_july_trip_id",
            "first_july_start_time",
            "first_july_origin",
            "t0_station_id",
        ]
    ].copy()

    boundary_pre["source_station_id"] = boundary_pre["last_pre_destination"]
    boundary_pre["destination_station_id"] = boundary_pre["first_july_origin"]
    boundary_pre["assumption"] = (
        "outside_to_inside_relocation_occurred_before_t0"
    )

    # Station table.
    ref = (
        reference
        .set_index("station_id")
        .reindex(station_ids)
        .reset_index()
    )

    station_total = int(final_counts.sum())
    available_counts = ref[
        "available_bikes_t0_reference"
    ].to_numpy(dtype=float)
    available_total = float(np.sum(available_counts))

    station_df = ref.copy()
    station_df["reconstructed_july_cohort_stock_t0"] = final_counts.astype(int)
    station_df["reconstructed_july_cohort_share"] = (
        final_counts.astype(float) / station_total
    )
    station_df["available_reference_share"] = (
        available_counts / available_total
    )
    station_df["share_difference"] = (
        station_df["reconstructed_july_cohort_share"]
        - station_df["available_reference_share"]
    )
    station_df["absolute_share_difference"] = np.abs(
        station_df["share_difference"]
    )
    station_df["count_difference_reconstructed_minus_available"] = (
        station_df["reconstructed_july_cohort_stock_t0"]
        - station_df["available_bikes_t0_reference"]
    )
    station_df["trajectory_fixed_bikes_before_ambiguous"] = (
        fixed_counts.astype(int)
    )
    station_df["observed_total_docks_audit"] = np.where(
        finite_cap,
        observed_capacity.astype(float),
        np.nan,
    )

    exceeds = np.zeros(len(station_ids), dtype=bool)
    excess_amount = np.zeros(len(station_ids), dtype=np.int64)

    exceeds[finite_cap] = (
        final_counts[finite_cap]
        > observed_capacity[finite_cap]
    )
    excess_amount[finite_cap] = np.maximum(
        final_counts[finite_cap]
        - observed_capacity[finite_cap],
        0,
    )

    station_df["exceeds_observed_total_docks_audit"] = exceeds
    station_df["bikes_above_observed_total_docks_audit"] = excess_amount
    station_df["capacity_used_as_assignment_constraint"] = False
    station_df["available_bikes_used_as_spatial_target"] = True
    station_df["available_bikes_used_as_hard_lower_bound"] = False

    # Exact per-bike count cross-check.
    station_to_idx = {
        sid: j
        for j, sid in enumerate(station_ids)
    }

    check = np.zeros(
        len(station_ids),
        dtype=np.int64,
    )

    for sid, n in (
        ev["t0_station_id"]
        .dropna()
        .astype(str)
        .value_counts()
        .items()
    ):
        check[station_to_idx[sid]] = int(n)

    if not np.array_equal(check, final_counts):
        raise RuntimeError(
            "Per-bike V6.4 states do not reproduce optimized station counts."
        )

    return (
        ev,
        station_df,
        pre_internal,
        post_internal,
        boundary_pre,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    banner(
        "DIVVY 2017 t0 BIKE LOCATION INFERENCE V6.4 "
        "- PRIOR-ENVELOPE SPATIAL CALIBRATION"
    )

    for p in [
        JULY_MAIN_PATH,
        TRIP_CONTEXT_PATH,
        JUNE_LAST_PRE_PATH,
        INVENTORY_MAIN_PATH,
    ]:
        require_file(p)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Output: {OUT_ROOT}")
    print("RAW INPUT FILES ARE READ ONLY.")
    print("NO proportional scaling.")
    print("NO available_bikes lower-bound forcing.")
    print("available_bikes is a normalized spatial target.")
    print("NO total_docks assignment constraint.")
    print("NO capacity-floor repair.")
    print("NO third-station assignment.")
    print("NO arbitrary spatial/temporal objective weight.")
    print("Spatial L1 is minimized under an empirical temporal-prior NLL envelope.")

    # ------------------------------------------------------------------
    # Evidence/candidate pipeline.
    # ------------------------------------------------------------------
    july, july_bikes, station_ids, first_july = load_july_main()
    last_pre = load_last_pre_t0(july_bikes)
    active_t0 = load_active_at_t0(july_bikes)

    reference, anchor_time = build_t0_available_reference(
        station_ids
    )

    evidence = build_bike_evidence(
        july_bikes,
        station_ids,
        first_july,
        last_pre,
        active_t0,
    )

    (
        fixed_counts,
        ambiguous_df,
        fixed_df,
        active_count,
        station_total,
    ) = build_assignment_problem(
        evidence,
        station_ids,
    )

    observed_capacity, finite_cap = build_capacity_arrays(
        reference,
        station_ids,
    )

    available_counts = (
        reference
        .set_index("station_id")
        .reindex(station_ids)[
            "available_bikes_t0_reference"
        ]
        .to_numpy(dtype=float)
    )

    available_total = float(np.sum(available_counts))

    if available_total <= 0:
        raise RuntimeError("Available-bike reference total is nonpositive.")

    available_share = (
        available_counts / available_total
    )

    # ------------------------------------------------------------------
    # Temporal-prior admissibility envelope.
    # ------------------------------------------------------------------
    nll_threshold, nll_threshold_source = load_nll_threshold()

    # ------------------------------------------------------------------
    # Global constrained spatial calibration.
    # ------------------------------------------------------------------
    solve = solve_prior_envelope_spatial_milp(
        fixed_counts=fixed_counts,
        ambiguous_df=ambiguous_df,
        available_share=available_share,
        station_total=station_total,
        nll_threshold=nll_threshold,
    )

    x = solve["x"]
    final_counts = solve["counts"]

    # ------------------------------------------------------------------
    # Sample one concrete relocation time conditional on the selected side.
    # ------------------------------------------------------------------
    ambiguous_decisions = sample_conditional_relocation_times(
        ambiguous_df,
        x,
        CONDITIONAL_TIME_RANDOM_SEED,
    )

    # Attach per-bike prior loss for audit.
    if len(ambiguous_decisions):
        p_raw = ambiguous_decisions[
            "relocation_probability_before_t0"
        ].to_numpy(dtype=float)
        p_log = np.clip(
            p_raw,
            LOG_EPS,
            1.0 - LOG_EPS,
        )

        per_bike_nll = -np.where(
            x == 1,
            np.log(p_log),
            np.log(1.0 - p_log),
        )

        ambiguous_decisions[
            "chosen_side_prior_nll"
        ] = per_bike_nll

        ambiguous_decisions[
            "within_side_timestamp_seed"
        ] = CONDITIONAL_TIME_RANDOM_SEED

    # ------------------------------------------------------------------
    # Final bike/station outputs.
    # ------------------------------------------------------------------
    (
        bike_locations,
        station_inventory,
        pre_internal,
        post_internal,
        boundary_pre,
    ) = build_v64_outputs(
        evidence=evidence,
        station_ids=station_ids,
        reference=reference,
        fixed_counts=fixed_counts,
        ambiguous_decisions=ambiguous_decisions,
        x=x,
        final_counts=final_counts,
        observed_capacity=observed_capacity,
        finite_cap=finite_cap,
    )

    ambiguous_count = len(ambiguous_df)
    internal_total = (
        len(pre_internal)
        + len(post_internal)
    )

    if internal_total != ambiguous_count:
        raise RuntimeError(
            "Internal relocation identity failed: "
            f"{len(pre_internal)} + {len(post_internal)} "
            f"!= {ambiguous_count}"
        )

    cohort_size = len(july_bikes)
    total_check = (
        int(np.sum(final_counts))
        + int(active_count)
    )

    if total_check != cohort_size:
        raise RuntimeError("V6.4 t0 mass conservation failed.")

    # ------------------------------------------------------------------
    # Final metrics.
    # ------------------------------------------------------------------
    final_share = (
        final_counts.astype(float)
        / float(np.sum(final_counts))
    )

    spatial_L1 = spatial_l1(
        final_counts,
        available_share,
    )
    spatial_TV = 0.5 * spatial_L1
    share_mae = float(
        np.mean(
            np.abs(
                final_share
                - available_share
            )
        )
    )
    count_corr = safe_corr(
        final_counts,
        available_counts,
    )

    final_nll = prior_mean_nll(
        ambiguous_df,
        x,
    )
    temporal_cost = temporal_disagreement(
        ambiguous_df,
        x,
    )
    brier = brier_score(
        ambiguous_df,
        x,
    )

    p = ambiguous_df[
        "relocation_probability_before_t0"
    ].to_numpy(dtype=float) if ambiguous_count else np.zeros(0, dtype=float)

    expected_pre = float(np.sum(p))
    realized_pre = int(np.sum(x == 1))
    realized_post = int(np.sum(x == 0))

    if final_nll > nll_threshold + 1e-7:
        raise RuntimeError(
            "Final V6.4 assignment exceeds the temporal-prior envelope."
        )

    dock_exceed_mask = (
        finite_cap
        & (
            final_counts
            > observed_capacity
        )
    )

    dock_excess = np.zeros(
        len(station_ids),
        dtype=np.int64,
    )

    dock_excess[finite_cap] = np.maximum(
        final_counts[finite_cap]
        - observed_capacity[finite_cap],
        0,
    )

    # ------------------------------------------------------------------
    # Optimization audit.
    # ------------------------------------------------------------------
    optimization_audit = pd.DataFrame(
        [
            {
                "solution": "stage1_spatial_optimum_under_nll_envelope",
                "spatial_L1_share_distance": solve[
                    "stage1_spatial_L1"
                ],
                "prior_mean_NLL": solve[
                    "stage1_prior_mean_NLL"
                ],
                "NLL_threshold": nll_threshold,
                "NLL_threshold_slack": (
                    nll_threshold
                    - solve["stage1_prior_mean_NLL"]
                ),
                "NLL_quantile": NLL_QUANTILE,
                "count_correlation_with_available": np.nan,
                "choose_B_pre_t0": int(
                    np.sum(
                        solve["stage1_x"] == 1
                    )
                ),
                "choose_A_post_t0": int(
                    np.sum(
                        solve["stage1_x"] == 0
                    )
                ),
            },
            {
                "solution": "v6_4_lexicographic_final",
                "spatial_L1_share_distance": spatial_L1,
                "prior_mean_NLL": final_nll,
                "NLL_threshold": nll_threshold,
                "NLL_threshold_slack": (
                    nll_threshold - final_nll
                ),
                "NLL_quantile": NLL_QUANTILE,
                "count_correlation_with_available": count_corr,
                "choose_B_pre_t0": realized_pre,
                "choose_A_post_t0": realized_post,
            },
        ]
    )

    # ------------------------------------------------------------------
    # Print final audit.
    # ------------------------------------------------------------------
    banner("FINAL V6.4 t0 AUDIT")

    print(f"Fixed July cohort: {cohort_size:,}")
    print(f"t0 active node0 bikes: {active_count:,}")
    print(f"t0 station bikes: {station_total:,}")
    print(f"t0 total: {total_check:,}")
    print()
    print(f"Ambiguous A/B bikes: {ambiguous_count:,}")
    print(f"Choose B / relocation before t0: {realized_pre:,}")
    print(f"Choose A / relocation after t0: {realized_post:,}")
    print(
        "Internal relocation identity: "
        f"{len(pre_internal):,} + "
        f"{len(post_internal):,} = "
        f"{internal_total:,}"
    )
    print(
        "Expected pre-t0 relocations sum(p): "
        f"{expected_pre:.3f}"
    )
    print(
        "Boundary outside -> inside pre-t0 assumptions: "
        f"{len(boundary_pre):,}"
    )
    print()
    print("TEMPORAL-PRIOR ENVELOPE")
    print(f"Threshold source: {nll_threshold_source}")
    print(f"NLL threshold: {nll_threshold:.12f}")
    print(f"Final prior mean NLL: {final_nll:.12f}")
    print(
        "NLL threshold slack: "
        f"{nll_threshold - final_nll:.12f}"
    )
    print(
        "MAP/minimum achievable prior mean NLL: "
        f"{solve['map_prior_mean_NLL']:.12f}"
    )
    print()
    print("SPATIAL CALIBRATION")
    print(
        "Stage-1 minimum spatial L1 under NLL envelope: "
        f"{solve['stage1_spatial_L1']:.12f}"
    )
    print(
        "Final lexicographic spatial L1: "
        f"{spatial_L1:.12f}"
    )
    print(
        "Final spatial TV: "
        f"{spatial_TV:.12f}"
    )
    print(
        "Station-count correlation with available: "
        f"{count_corr:.12f}"
    )
    print(
        "Mean absolute share difference: "
        f"{share_mae:.12f}"
    )
    print()
    print("TEMPORAL AUDIT")
    print(
        "Temporal disagreement: "
        f"{temporal_cost:.12f}"
    )
    print(
        "Brier score: "
        f"{brier:.12f}"
    )
    print()
    print("ORIGINAL total_docks AUDIT ONLY")
    print(
        "Stations above original total_docks: "
        f"{int(np.sum(dock_exceed_mask)):,}"
    )
    print(
        "Total excess bike positions above original total_docks: "
        f"{int(np.sum(dock_excess)):,}"
    )
    print("total_docks was NOT used to alter the assignment.")

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    banner("WRITE V6.4 OUTPUTS")

    p_bikes = OUT_ROOT / "t0_bike_locations_v6_4.csv"
    p_station = OUT_ROOT / "t0_station_inventory_v6_4.csv"
    p_active = OUT_ROOT / "t0_active_user_trip_bikes_v6_4.csv"
    p_decisions = OUT_ROOT / "ambiguous_bike_decisions_v6_4.csv"
    p_pre = OUT_ROOT / "sampled_pre_t0_internal_relocations_v6_4.csv"
    p_post = OUT_ROOT / "sampled_post_t0_pre_first_internal_relocations_v6_4.csv"
    p_boundary = OUT_ROOT / "implied_boundary_pre_t0_relocations_v6_4.csv"
    p_ref = OUT_ROOT / "t0_available_spatial_reference_v6_4.csv"
    p_audit = OUT_ROOT / "optimization_audit_v6_4.csv"
    p_txt = OUT_ROOT / "summary_v6_4.txt"
    p_json = OUT_ROOT / "summary_v6_4.json"

    bike_locations.to_csv(
        p_bikes,
        index=False,
    )

    station_inventory.to_csv(
        p_station,
        index=False,
    )

    bike_locations[
        bike_locations[
            "t0_is_user_trip_node"
        ]
    ].to_csv(
        p_active,
        index=False,
    )

    ambiguous_decisions.to_csv(
        p_decisions,
        index=False,
    )

    pre_internal.to_csv(
        p_pre,
        index=False,
    )

    post_internal.to_csv(
        p_post,
        index=False,
    )

    boundary_pre.to_csv(
        p_boundary,
        index=False,
    )

    reference.to_csv(
        p_ref,
        index=False,
    )

    optimization_audit.to_csv(
        p_audit,
        index=False,
    )

    evidence_counts = (
        bike_locations[
            "evidence_type"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "version": "V6.4_prior_envelope_spatial_calibration",
        "t0": T0,
        "inventory_anchor_time": anchor_time,
        "cohort": {
            "fixed_july_bikes": cohort_size,
            "active_node0": active_count,
            "station_bikes": station_total,
            "total": total_check,
        },
        "evidence_counts": evidence_counts,
        "temporal_prior_envelope": {
            "quantile": NLL_QUANTILE,
            "threshold": nll_threshold,
            "threshold_source": nll_threshold_source,
            "map_minimum_prior_mean_NLL": solve[
                "map_prior_mean_NLL"
            ],
            "stage1_prior_mean_NLL": solve[
                "stage1_prior_mean_NLL"
            ],
            "final_prior_mean_NLL": final_nll,
            "final_threshold_slack": (
                nll_threshold - final_nll
            ),
        },
        "optimization": {
            "method": (
                "lexicographic MILP: first minimize spatial L1 under "
                "prior-NLL envelope; then minimize NLL while retaining "
                "the Stage-1 spatial optimum"
            ),
            "stage1_spatial_L1": solve[
                "stage1_spatial_L1"
            ],
            "final_spatial_L1": spatial_L1,
            "final_spatial_TV": spatial_TV,
            "count_correlation_with_available": count_corr,
            "mean_abs_share_difference": share_mae,
            "arbitrary_spatial_temporal_weight_used": False,
        },
        "ambiguous_assignment": {
            "ambiguous_bikes": ambiguous_count,
            "choose_A_june_destination": realized_post,
            "choose_B_first_july_origin": realized_pre,
            "expected_pre_t0_relocations_sum_p": expected_pre,
            "pre_t0_internal_relocations": len(pre_internal),
            "post_t0_pre_first_internal_relocations": len(post_internal),
            "internal_relocation_total": internal_total,
            "boundary_pre_t0_relocations": len(boundary_pre),
            "conditional_timestamp_random_seed": (
                CONDITIONAL_TIME_RANDOM_SEED
            ),
        },
        "temporal_audit": {
            "prior_mean_NLL": final_nll,
            "temporal_disagreement": temporal_cost,
            "brier_score_choiceB_vs_p_before": brier,
        },
        "available_reference": {
            "available_reference_total": int(
                np.sum(available_counts)
            ),
            "reconstructed_station_total": station_total,
            "count_total_difference": int(
                station_total
                - np.sum(available_counts)
            ),
            "used_as_hard_lower_bound": False,
            "used_as_normalized_spatial_target": True,
        },
        "capacity_audit": {
            "used_as_assignment_constraint": False,
            "capacity_floor_used": False,
            "stations_above_original_total_docks": int(
                np.sum(dock_exceed_mask)
            ),
            "total_excess_bike_positions_above_original_total_docks": int(
                np.sum(dock_excess)
            ),
        },
        "invariants": {
            "no_proportional_scaling": True,
            "no_available_lower_bound": True,
            "no_third_station_assignment": True,
            "no_capacity_repair": True,
            "no_arbitrary_spatial_temporal_weight": True,
            "prior_NLL_envelope_satisfied": bool(
                final_nll <= nll_threshold + 1e-7
            ),
            "internal_ambiguous_relocation_identity_holds": bool(
                internal_total == ambiguous_count
            ),
            "mass_conservation_at_t0": bool(
                total_check == cohort_size
            ),
        },
    }

    with p_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
            default=jsonable,
        )

    lines = [
        "DIVVY 2017 t0 BIKE LOCATION INFERENCE V6.4",
        "PRIOR-ENVELOPE-CONSTRAINED SPATIAL CALIBRATION",
        "=" * 90,
        "",
        f"t0: {T0}",
        f"inventory_anchor_time: {anchor_time}",
        "",
        "CORE METHOD",
        "Stage 1: minimize spatial L1 to available_bikes shares.",
        "Constraint: complete A/B prior mean NLL <= empirical Monte-Carlo envelope.",
        "Stage 2: preserve Stage-1 minimum spatial L1 and minimize NLL.",
        "No arbitrary spatial/temporal trade-off weight.",
        "",
        "TEMPORAL PRIOR ENVELOPE",
        f"quantile: {NLL_QUANTILE}",
        f"threshold: {nll_threshold:.12f}",
        f"threshold_source: {nll_threshold_source}",
        f"map_minimum_prior_mean_NLL: {solve['map_prior_mean_NLL']:.12f}",
        f"stage1_prior_mean_NLL: {solve['stage1_prior_mean_NLL']:.12f}",
        f"final_prior_mean_NLL: {final_nll:.12f}",
        f"final_NLL_slack: {nll_threshold - final_nll:.12f}",
        "",
        "SPATIAL RESULT",
        f"stage1_spatial_L1: {solve['stage1_spatial_L1']:.12f}",
        f"final_spatial_L1: {spatial_L1:.12f}",
        f"final_spatial_TV: {spatial_TV:.12f}",
        f"count_correlation_with_available: {count_corr:.12f}",
        f"mean_abs_share_difference: {share_mae:.12f}",
        "",
        "COHORT",
        f"fixed_july_bikes: {cohort_size}",
        f"active_node0: {active_count}",
        f"station_bikes: {station_total}",
        f"total: {total_check}",
        "",
        "AMBIGUOUS A/B",
        f"ambiguous_bikes: {ambiguous_count}",
        f"choose_A_june_destination: {realized_post}",
        f"choose_B_first_july_origin: {realized_pre}",
        f"expected_pre_t0_relocations_sum_p: {expected_pre:.6f}",
        f"pre_t0_internal_relocations: {len(pre_internal)}",
        f"post_t0_pre_first_internal_relocations: {len(post_internal)}",
        f"pre+post: {internal_total}",
        f"boundary_pre_t0_relocations: {len(boundary_pre)}",
        "",
        "TEMPORAL AUDIT",
        f"temporal_disagreement: {temporal_cost:.12f}",
        f"brier_score: {brier:.12f}",
        "",
        "AVAILABLE REFERENCE",
        f"available_reference_total: {int(np.sum(available_counts))}",
        f"reconstructed_station_total: {station_total}",
        (
            "reconstructed_minus_available_total: "
            f"{int(station_total - np.sum(available_counts))}"
        ),
        "available_bikes is a normalized spatial target, not a hard lower bound.",
        "",
        "CAPACITY AUDIT ONLY",
        "total_docks was not used as an assignment constraint.",
        "No capacity floor or capacity repair.",
        (
            "stations_above_original_total_docks: "
            f"{int(np.sum(dock_exceed_mask))}"
        ),
        (
            "total_excess_bike_positions_above_original_total_docks: "
            f"{int(np.sum(dock_excess))}"
        ),
        "",
        "INVARIANTS",
        "No proportional scaling.",
        "No third-station assignment.",
        "No arbitrary spatial/temporal objective weight.",
        (
            "NLL envelope satisfied: "
            f"{final_nll:.12f} <= {nll_threshold:.12f}"
        ),
        (
            "internal relocation identity: "
            f"{len(pre_internal)} + {len(post_internal)} "
            f"= {ambiguous_count}"
        ),
        (
            "mass conservation: "
            f"{station_total} + {active_count} = {cohort_size}"
        ),
    ]

    p_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    for pth in [
        p_bikes,
        p_station,
        p_active,
        p_decisions,
        p_pre,
        p_post,
        p_boundary,
        p_ref,
        p_audit,
        p_txt,
        p_json,
    ]:
        print(f"Saved: {pth}")

    banner("FINAL V6.4 SNAPSHOT")

    print(
        f"station/node0/total = "
        f"{station_total:,}/"
        f"{active_count:,}/"
        f"{total_check:,}"
    )
    print(f"ambiguous A/B = {ambiguous_count:,}")
    print(
        f"choose A/B = "
        f"{realized_post:,}/"
        f"{realized_pre:,}"
    )
    print(
        f"internal relocation pre/post = "
        f"{len(pre_internal):,}/"
        f"{len(post_internal):,}"
    )
    print(
        f"NLL threshold/final = "
        f"{nll_threshold:.6f}/"
        f"{final_nll:.6f}"
    )
    print(
        f"spatial L1 = "
        f"{spatial_L1:.6f}"
    )
    print(
        f"TV = "
        f"{spatial_TV:.6f}"
    )
    print(
        "count correlation with available = "
        f"{count_corr:.6f}"
    )
    print(
        "above original total_docks "
        "(stations / excess bikes) = "
        f"{int(np.sum(dock_exceed_mask)):,}/"
        f"{int(np.sum(dock_excess)):,}"
    )
    print("DONE.")


if __name__ == "__main__":
    main()
