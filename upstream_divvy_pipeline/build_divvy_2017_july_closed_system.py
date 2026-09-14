# -*- coding: utf-8 -*-
"""
build_divvy_2017_july_closed_system.py

Purpose
-------
Construct one physically closed bike-level realization for the complete
July-2017 Divvy system, using V6.4 as the ONLY source of the t0 bike state.

The fixed cohort is exactly the July bike-ID cohort used by V6.4. No bike may
enter or leave this cohort during July.

State space
-----------
For each bike b and time t:

    state_b(t) = "0"          : user is riding the bike
                 station_id   : bike is at that station

Hard user-trip physics
----------------------
Every accepted July user trip is represented exactly as:

    origin_station --(start_time)--> 0 --(stop_time)--> destination_station

Latent relocations
------------------
1. V6.4 post-t0 / pre-first-trip relocations are reused EXACTLY, including
   their already-sampled timestamps. They are never resampled here.

2. For two consecutive accepted July trips of the same bike, if the previous
   destination differs from the next origin, a non-user relocation is mandatory.
   Its timestamp is sampled once, reproducibly, strictly inside the admissible
   gap (previous_stop, next_start).

3. For a bike already in a user trip at t0, if that crossing trip ends at a
   station different from the first accepted July-trip origin, a relocation is
   sampled inside (crossing_trip_stop, first_July_start).

Physical invariants
-------------------
At every reconstructed state boundary:

    sum_i B_i(t) + B_0(t) = N_JULY_BIKES

and all station stocks and node-0 stock must be nonnegative.

available_bikes and total_docks
-------------------------------
These are audit/comparison information only. They NEVER add/remove bikes,
change a bike trajectory, or force a third-station assignment.

10-minute representation
-----------------------
The state grid contains both month boundaries:

    2017-07-01 00:00:00, ..., 2017-08-01 00:00:00

so there are 4,465 state boundaries and 4,464 ten-minute intervals.
An event is reflected in the first state boundary >= its event timestamp.

Inputs
------
Raw:
D:/physic_predict_bike/divvy_2017_physics_raw/
    trips/divvy_trips_main_window.csv
    trips/divvy_trips_context.csv
    station_inventory/divvy_station_inventory_main_window.csv

V6.4:
D:/physic_predict_bike/divvy_2017_physics_closed_active_v6_4_prior_envelope_t0/
    t0_bike_locations_v6_4.csv
    sampled_post_t0_pre_first_internal_relocations_v6_4.csv
    t0_available_spatial_reference_v6_4.csv

Outputs
-------
D:/physic_predict_bike/divvy_2017_physics_closed_active_v7_10min_from_v64/

    active_stations_v7.csv
    accepted_july_trips_v7.csv.gz
    same_bike_overlap_anomalies_excluded_v7.csv
    initial_bike_states_from_v64.csv
    bike_events_v7.csv.gz
    relocations_v7.csv.gz
    invalid_relocation_gaps_v7.csv
    bike_event_state_errors_v7.csv
    system_totals_v7.csv
    reconstructed_bike_inventory_10min_wide_v7.csv.gz
    available_bikes_10min_comparison_wide_v7.csv.gz
    station_state_comparison_10min_v7.csv.gz
    user_od_flows_10min_sparse_v7.csv.gz
    relocation_od_flows_10min_sparse_v7.csv.gz
    all_od_flows_10min_sparse_v7.csv.gz
    closed_system_v7_10min_arrays.npz
    summary_v7.json
    summary_v7.txt

Raw inputs and V6.4 outputs are READ ONLY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_ROOT = Path(r"D:/physic_predict_bike/divvy_2017_physics_raw")

TRIP_CONTEXT_PATH = RAW_ROOT / "trips" / "divvy_trips_context.csv"
TRIP_MAIN_PATH = RAW_ROOT / "trips" / "divvy_trips_main_window.csv"
INVENTORY_MAIN_PATH = (
    RAW_ROOT / "station_inventory" / "divvy_station_inventory_main_window.csv"
)

V64_ROOT = Path(
    r"D:/physic_predict_bike/"
    r"divvy_2017_physics_closed_active_v6_4_prior_envelope_t0"
)

V64_T0_BIKES_PATH = V64_ROOT / "t0_bike_locations_v6_4.csv"
V64_POST_RELOCATIONS_PATH = (
    V64_ROOT / "sampled_post_t0_pre_first_internal_relocations_v6_4.csv"
)
V64_T0_AVAILABLE_REFERENCE_PATH = (
    V64_ROOT / "t0_available_spatial_reference_v6_4.csv"
)

OUT_ROOT = Path(
    r"D:/physic_predict_bike/divvy_2017_physics_closed_active_v7_10min_from_v64"
)

MAIN_START = pd.Timestamp("2017-07-01 00:00:00")
MAIN_END_EXCLUSIVE = pd.Timestamp("2017-08-01 00:00:00")
GRID_FREQ = "10min"

# Only NEW latent relocation times inside July are sampled with this seed.
# V6.4 post-t0/pre-first relocation timestamps are reused exactly.
RANDOM_SEED = 20260909

# Inventory comparison lifecycle rules only. They never change bike-ID physics.
LIFECYCLE_RULES = {
    "624": "zero_before_first_observation",
    "193": "zero_after_last_observation",
    "392": "zero_after_last_observation",
}

WRITE_LONG_STATE_COMPARISON = True
WRITE_WIDE_MATRICES = True
WRITE_NPZ = True
WRITE_SPARSE_FLOWS = True

# Fail immediately on physical inconsistency instead of silently repairing.
FAIL_ON_INVALID_RELOCATION_GAP = True
FAIL_ON_EVENT_STATE_MISMATCH = True
FAIL_ON_NEGATIVE_STOCK = True
FAIL_ON_MASS_ERROR = True


# =============================================================================
# HELPERS
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


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


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """Robustly parse CSV boolean values."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    x = s.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    out = x.map(mapping)
    if out.isna().any():
        bad = s[out.isna()].head(20).tolist()
        raise ValueError(f"Cannot parse boolean values. Examples={bad}")
    return out.astype(bool)


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


def random_time_strictly_between(
    rng: np.random.Generator,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
) -> pd.Timestamp:
    """Draw lower < t < upper on the nanosecond lattice."""
    lower = pd.Timestamp(lower)
    upper = pd.Timestamp(upper)

    lo = int(lower.value) + 1
    hi = int(upper.value) - 1

    if lo > hi:
        raise ValueError(
            f"No strictly interior relocation time: lower={lower}, upper={upper}"
        )

    return pd.Timestamp(int(rng.integers(lo, hi + 1)))


def event_to_interval_start(ts: pd.Series) -> pd.Series:
    """
    Map event times to the interval whose END boundary contains the event.

    Physical state convention:
        I_0 = [t0, t1]
        I_k = (t_k, t_{k+1}] for k >= 1

    Thus an event exactly on an interior grid boundary belongs to the
    preceding transition, matching the state update convention that the
    state at t_{k+1} already includes events occurring at t_{k+1}.

    Events exactly at t0 are assigned to the first reporting interval. This
    matters only for OD reporting; t0 departures are already embedded in the
    V6.4 initial state and are not replayed as state-transition events.
    """
    t = pd.to_datetime(ts, errors="coerce")
    out = t.dt.ceil(GRID_FREQ) - pd.Timedelta(GRID_FREQ)
    out = out.mask(t == MAIN_START, MAIN_START)
    return out


# =============================================================================
# LOAD TRIPS
# =============================================================================

def load_trips(path: Path, label: str) -> pd.DataFrame:
    banner(f"LOAD {label}")

    header = pd.read_csv(path, nrows=0).columns.tolist()

    aliases = {
        "trip_id": ["trip_id", "tripid"],
        "bike_id": ["bike_id", "bikeid"],
        "start_time": ["start_time", "started_at"],
        "stop_time": ["stop_time", "end_time", "ended_at"],
        "from_station_id": ["from_station_id", "start_station_id"],
        "to_station_id": ["to_station_id", "end_station_id"],
        "from_station_name": ["from_station_name", "start_station_name"],
        "to_station_name": ["to_station_name", "end_station_name"],
    }

    resolved = {}
    for canonical, candidates in aliases.items():
        for c in candidates:
            if c in header:
                resolved[canonical] = c
                break

    required = [
        "trip_id",
        "bike_id",
        "start_time",
        "stop_time",
        "from_station_id",
        "to_station_id",
    ]

    missing = [c for c in required if c not in resolved]
    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}; available={header}"
        )

    df = pd.read_csv(
        path,
        usecols=list(dict.fromkeys(resolved.values())),
        low_memory=False,
    )
    df = df.rename(
        columns={actual: canonical for canonical, actual in resolved.items()}
    )

    for c in ["trip_id", "bike_id", "from_station_id", "to_station_id"]:
        df[c] = normalize_id_series(df[c])

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce")

    df["valid_time"] = (
        df["start_time"].notna()
        & df["stop_time"].notna()
        & (df["stop_time"] >= df["start_time"])
    )

    dup_mask = df["trip_id"].notna() & df["trip_id"].duplicated(keep=False)

    if dup_mask.any():
        dup = df[dup_mask]
        physical_cols = [
            "bike_id",
            "start_time",
            "stop_time",
            "from_station_id",
            "to_station_id",
        ]
        conflicts = (
            dup.groupby("trip_id")[physical_cols]
            .nunique(dropna=False)
            .max(axis=1)
            .gt(1)
        )
        if conflicts.any():
            raise RuntimeError(
                "Conflicting duplicate trip_id rows. "
                f"Examples={conflicts[conflicts].index.tolist()[:10]}"
            )
        df = df.drop_duplicates("trip_id", keep="first").copy()

    if df["bike_id"].isna().any():
        raise RuntimeError(f"{label}: missing bike_id exists.")

    print(f"Rows: {len(df):,}")
    print(f"Unique bikes: {df['bike_id'].nunique():,}")
    print(f"Invalid trip times: {(~df['valid_time']).sum():,}")

    return df


# =============================================================================
# JULY COHORT + STATION DOMAIN
# =============================================================================

def build_july_cohort_and_station_domain(
    trips_main: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    banner("BUILD JULY COHORT + ACTIVE STATION DOMAIN")

    main = trips_main[
        trips_main["valid_time"]
        & (trips_main["start_time"] >= MAIN_START)
        & (trips_main["start_time"] < MAIN_END_EXCLUSIVE)
        & trips_main["from_station_id"].notna()
        & trips_main["to_station_id"].notna()
    ].copy()

    july_bikes = sorted(
        main["bike_id"].dropna().astype(str).unique().tolist(),
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )

    station_ids = sorted(
        set(main["from_station_id"].astype(str))
        | set(main["to_station_id"].astype(str)),
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )

    dep = main["from_station_id"].value_counts()
    arr = main["to_station_id"].value_counts()

    name_map: Dict[str, str] = {}
    if "from_station_name" in main.columns:
        for sid, name in zip(main["from_station_id"], main["from_station_name"]):
            if pd.notna(sid) and pd.notna(name):
                name_map.setdefault(str(sid), str(name))

    if "to_station_name" in main.columns:
        for sid, name in zip(main["to_station_id"], main["to_station_name"]):
            if pd.notna(sid) and pd.notna(name):
                name_map.setdefault(str(sid), str(name))

    domain_df = pd.DataFrame(
        {
            "station_id": station_ids,
            "station_name": [name_map.get(s, "") for s in station_ids],
            "july_departures": [int(dep.get(s, 0)) for s in station_ids],
            "july_arrivals": [int(arr.get(s, 0)) for s in station_ids],
        }
    )
    domain_df["july_endpoint_events"] = (
        domain_df["july_departures"] + domain_df["july_arrivals"]
    )

    print(f"July fixed bike cohort from raw main trips: {len(july_bikes):,}")
    print(f"Active station domain: {len(station_ids):,}")
    print(f"July main-window trips before overlap cleaning: {len(main):,}")

    return main, july_bikes, station_ids, domain_df


# =============================================================================
# SAME-BIKE OVERLAP CLEANING
# =============================================================================

def remove_same_bike_trip_overlaps(
    july_main: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    banner("CLEAN SAME-BIKE OVERLAP ANOMALIES")

    accepted_idx: List[int] = []
    anomaly_rows: List[dict] = []

    ordered = july_main.sort_values(
        ["bike_id", "start_time", "stop_time", "trip_id"]
    )

    for bike_id, g in ordered.groupby("bike_id", sort=False):
        last_stop: Optional[pd.Timestamp] = None
        last_trip_id: Optional[str] = None

        for idx, row in g.iterrows():
            st = pd.Timestamp(row["start_time"])
            en = pd.Timestamp(row["stop_time"])

            if last_stop is not None and st < last_stop:
                anomaly_rows.append(
                    {
                        "bike_id": str(bike_id),
                        "excluded_trip_id": str(row["trip_id"]),
                        "excluded_start_time": st,
                        "excluded_stop_time": en,
                        "previous_accepted_trip_id": last_trip_id,
                        "previous_accepted_stop_time": last_stop,
                        "reason": "same_bike_user_trip_overlap",
                    }
                )
                continue

            accepted_idx.append(idx)
            last_stop = en
            last_trip_id = str(row["trip_id"])

    clean = july_main.loc[accepted_idx].copy()
    clean = clean.sort_values(
        ["bike_id", "start_time", "stop_time", "trip_id"]
    ).reset_index(drop=True)

    anomalies = pd.DataFrame(anomaly_rows)

    print(f"Accepted July trips: {len(clean):,}")
    print(f"Excluded overlapping trips: {len(anomalies):,}")

    return clean, anomalies


# =============================================================================
# LOAD V6.4 INITIAL STATE
# =============================================================================

def load_v64_initial_state(
    july_bikes: List[str],
    station_ids: List[str],
    july_trips_clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    banner("LOAD V6.4 t0 INITIAL STATE")

    v = pd.read_csv(V64_T0_BIKES_PATH, low_memory=False)

    required = [
        "bike_id",
        "t0_state",
        "t0_is_user_trip_node",
        "t0_station_id",
        "evidence_type",
        "first_july_trip_id",
        "first_july_start_time",
        "first_july_origin",
    ]
    missing = [c for c in required if c not in v.columns]
    if missing:
        raise ValueError(f"V6.4 t0 bike file missing columns: {missing}")

    v["t0_is_user_trip_node"] = normalize_bool_series(
        v["t0_is_user_trip_node"]
    )

    for c in [
        "bike_id",
        "t0_state",
        "t0_station_id",
        "first_july_trip_id",
        "first_july_origin",
    ]:
        v[c] = normalize_id_series(v[c])

    for c in [
        "first_july_start_time",
        "first_july_stop_time",
        "last_pre_stop_time",
        "active_t0_start_time",
        "active_t0_stop_time",
        "sampled_relocation_time",
    ]:
        if c in v.columns:
            v[c] = pd.to_datetime(v[c], errors="coerce")

    if v["bike_id"].duplicated().any():
        raise RuntimeError("V6.4 t0 bike file has duplicate bike_id rows.")

    v = v.set_index("bike_id", drop=False)

    july_set = set(july_bikes)
    v64_set = set(v.index.astype(str))

    missing_in_v64 = sorted(july_set - v64_set)
    extra_in_v64 = sorted(v64_set - july_set)

    if missing_in_v64 or extra_in_v64:
        raise RuntimeError(
            "V6.4 cohort differs from current July raw cohort. "
            f"missing_in_v64={missing_in_v64[:20]}, "
            f"extra_in_v64={extra_in_v64[:20]}"
        )

    v = v.reindex(july_bikes).reset_index(drop=True)

    station_set = set(station_ids)
    bad_state_rows = []

    for row in v.itertuples(index=False):
        state = str(row.t0_state)
        is_node0 = bool(row.t0_is_user_trip_node)

        if is_node0:
            if state != "0":
                bad_state_rows.append((row.bike_id, state, "node0_flag_but_state_not_0"))
        else:
            if state not in station_set:
                bad_state_rows.append((row.bike_id, state, "station_state_outside_domain"))

    if bad_state_rows:
        raise RuntimeError(
            "Invalid V6.4 t0 states. Examples=" + repr(bad_state_rows[:20])
        )

    # Cross-check V6.4 first July trip against the accepted July trip sequence.
    first_clean = (
        july_trips_clean.sort_values(
            ["bike_id", "start_time", "stop_time", "trip_id"]
        )
        .groupby("bike_id", as_index=False, sort=False)
        .head(1)
        .set_index("bike_id")
    )

    mismatch_rows = []
    for row in v.itertuples(index=False):
        bike = str(row.bike_id)
        if bike not in first_clean.index:
            mismatch_rows.append((bike, "no_accepted_july_trip"))
            continue

        fr = first_clean.loc[bike]
        if isinstance(fr, pd.DataFrame):
            fr = fr.iloc[0]

        if str(row.first_july_trip_id) != str(fr["trip_id"]):
            mismatch_rows.append(
                (
                    bike,
                    str(row.first_july_trip_id),
                    str(fr["trip_id"]),
                    "v64_first_trip_differs_after_overlap_cleaning",
                )
            )

    if mismatch_rows:
        raise RuntimeError(
            "V6.4 first-trip evidence is inconsistent with the accepted V7 "
            "July trip sequence. Do not silently change the initial condition. "
            f"Examples={mismatch_rows[:20]}"
        )

    initial_state_map = {
        str(r.bike_id): str(r.t0_state)
        for r in v.itertuples(index=False)
    }

    # Load V6.4 post-t0 / pre-first relocations.
    post = pd.read_csv(V64_POST_RELOCATIONS_PATH, low_memory=False)

    if len(post):
        required_post = [
            "bike_id",
            "source_station_id",
            "destination_station_id",
            "sampled_relocation_time",
            "first_july_start_time",
        ]
        missing = [c for c in required_post if c not in post.columns]
        if missing:
            raise ValueError(
                f"V6.4 post-t0 relocation file missing columns: {missing}"
            )

        for c in [
            "bike_id",
            "source_station_id",
            "destination_station_id",
            "first_july_trip_id",
        ]:
            if c in post.columns:
                post[c] = normalize_id_series(post[c])

        for c in [
            "sampled_relocation_time",
            "first_july_start_time",
            "window_start",
            "window_end",
            "last_pre_stop_time",
        ]:
            if c in post.columns:
                post[c] = pd.to_datetime(post[c], errors="coerce")

        if post["bike_id"].duplicated().any():
            raise RuntimeError(
                "V6.4 post-t0 relocation file has duplicate bike_id rows."
            )

        post_bikes = set(post["bike_id"].astype(str))
        if not post_bikes.issubset(july_set):
            raise RuntimeError("V6.4 post relocation contains non-cohort bike IDs.")

        bad_time = (
            post["sampled_relocation_time"].isna()
            | (post["sampled_relocation_time"] <= MAIN_START)
            | (post["sampled_relocation_time"] >= post["first_july_start_time"])
        )
        if bad_time.any():
            raise RuntimeError(
                "Invalid V6.4 post-t0 relocation timestamp. Examples="
                + repr(
                    post.loc[
                        bad_time,
                        [
                            "bike_id",
                            "sampled_relocation_time",
                            "first_july_start_time",
                        ],
                    ].head(20).to_dict("records")
                )
            )

        # Exact source must equal V6.4 t0 station, destination must equal first origin.
        v_index = v.set_index("bike_id")
        bad_link = []
        for row in post.itertuples(index=False):
            bike = str(row.bike_id)
            vr = v_index.loc[bike]
            if str(vr["t0_state"]) != str(row.source_station_id):
                bad_link.append((bike, "source_not_t0_state"))
            if str(vr["first_july_origin"]) != str(row.destination_station_id):
                bad_link.append((bike, "destination_not_first_origin"))
        if bad_link:
            raise RuntimeError(
                "V6.4 post relocation does not match V6.4 t0/first-origin evidence. "
                f"Examples={bad_link[:20]}"
            )

    print(f"Loaded V6.4 bike states: {len(v):,}")
    print(f"V6.4 t0 node0 bikes: {int(v['t0_is_user_trip_node'].astype(bool).sum()):,}")
    print(f"V6.4 t0 station bikes: {int((~v['t0_is_user_trip_node'].astype(bool)).sum()):,}")
    print(f"V6.4 post-t0/pre-first relocations reused: {len(post):,}")

    return v, initial_state_map, post


# =============================================================================
# ACTIVE-AT-t0 CONTEXT MAP
# =============================================================================

def build_active_t0_map(
    trips_context: pd.DataFrame,
    july_bikes: List[str],
) -> Dict[str, dict]:
    banner("BUILD ACTIVE-AT-t0 CONTEXT MAP")

    july_set = set(july_bikes)

    a = trips_context[
        trips_context["valid_time"]
        & trips_context["bike_id"].astype(str).isin(july_set)
        & (trips_context["start_time"] <= MAIN_START)
        & (trips_context["stop_time"] > MAIN_START)
    ].copy()

    active_map: Dict[str, dict] = {}

    for bike, g in a.groupby("bike_id", sort=False):
        if len(g) > 1:
            print(
                f"WARNING: bike {bike} has {len(g)} active-at-t0 context rows; "
                "using latest-starting row, matching V6.4 defensive policy."
            )

        row = g.sort_values(
            ["start_time", "stop_time", "trip_id"]
        ).iloc[-1]

        active_map[str(bike)] = {
            "trip_id": str(row["trip_id"]),
            "start_time": pd.Timestamp(row["start_time"]),
            "stop_time": pd.Timestamp(row["stop_time"]),
            "from_station_id": str(row["from_station_id"]),
            "to_station_id": str(row["to_station_id"]),
        }

    print(f"Active-at-t0 context bikes: {len(active_map):,}")
    return active_map


# =============================================================================
# BUILD V7 EVENTS
# =============================================================================

def build_v7_events(
    july_trips_clean: pd.DataFrame,
    july_bikes: List[str],
    station_ids: List[str],
    initial_state_map: Dict[str, str],
    v64_initial_df: pd.DataFrame,
    v64_post_relocations: pd.DataFrame,
    active_map: Dict[str, dict],
    rng: np.random.Generator,
):
    banner("BUILD V7 PHYSICAL EVENT STREAM")

    station_set = set(station_ids)

    trips_by_bike = {
        str(b): g.sort_values(
            ["start_time", "stop_time", "trip_id"]
        ).copy()
        for b, g in july_trips_clean.groupby("bike_id", sort=False)
    }

    v64_by_bike = v64_initial_df.set_index("bike_id")
    post_by_bike = (
        v64_post_relocations.set_index("bike_id")
        if len(v64_post_relocations)
        else pd.DataFrame()
    )

    event_rows: List[dict] = []
    relocation_rows: List[dict] = []
    invalid_gap_rows: List[dict] = []

    relocation_counter = 0
    global_insert_counter = 0
    bike_seq_counter: Dict[str, int] = {b: 0 for b in july_bikes}

    def add_event(
        bike_id: str,
        event_time: pd.Timestamp,
        event_type: str,
        source_state: str,
        destination_state: str,
        trip_id: Optional[str] = None,
        relocation_id: Optional[str] = None,
        event_source: str = "",
    ) -> None:
        nonlocal global_insert_counter

        bike_seq_counter[bike_id] += 1
        global_insert_counter += 1

        event_rows.append(
            {
                "bike_id": bike_id,
                "event_time": pd.Timestamp(event_time),
                "event_type": event_type,
                "source_state": source_state,
                "destination_state": destination_state,
                "trip_id": trip_id,
                "relocation_id": relocation_id,
                "event_source": event_source,
                "bike_event_seq_insert": bike_seq_counter[bike_id],
                "global_insert_seq": global_insert_counter,
            }
        )

    def record_invalid_gap(
        bike_id: str,
        src: str,
        dst: str,
        lower: pd.Timestamp,
        upper: pd.Timestamp,
        scope: str,
        prev_trip_id: Optional[str],
        next_trip_id: Optional[str],
        reason: str,
    ) -> None:
        invalid_gap_rows.append(
            {
                "bike_id": bike_id,
                "source_station_id": src,
                "destination_station_id": dst,
                "window_start_exclusive": lower,
                "window_end_exclusive": upper,
                "scope": scope,
                "prev_trip_id": prev_trip_id,
                "next_trip_id": next_trip_id,
                "reason": reason,
            }
        )

    def add_random_relocation(
        bike_id: str,
        src: str,
        dst: str,
        lower: pd.Timestamp,
        upper: pd.Timestamp,
        scope: str,
        prev_trip_id: Optional[str],
        next_trip_id: Optional[str],
    ) -> None:
        nonlocal relocation_counter

        if src == dst:
            return

        lower = pd.Timestamp(lower)
        upper = pd.Timestamp(upper)

        # The physical gap itself must be positive before any July clipping.
        if not (lower < upper):
            record_invalid_gap(
                bike_id, src, dst, lower, upper, scope,
                prev_trip_id, next_trip_id, "no_positive_relocation_gap"
            )
            return

        # Only create a July event if there is a strict interior point inside
        # the modeled month. For all intended V7 calls the full gap is in July.
        lo = max(lower, MAIN_START)
        hi = min(upper, MAIN_END_EXCLUSIVE)

        if not (lo < hi):
            record_invalid_gap(
                bike_id, src, dst, lo, hi, scope,
                prev_trip_id, next_trip_id, "gap_outside_modeled_month"
            )
            return

        try:
            rt = random_time_strictly_between(rng, lo, hi)
        except ValueError:
            record_invalid_gap(
                bike_id, src, dst, lo, hi, scope,
                prev_trip_id, next_trip_id, "no_strictly_interior_nanosecond"
            )
            return

        relocation_counter += 1
        rid = f"R{relocation_counter:08d}"

        relocation_rows.append(
            {
                "relocation_id": rid,
                "bike_id": bike_id,
                "source_station_id": src,
                "destination_station_id": dst,
                "window_start_exclusive": lo,
                "window_end_exclusive": hi,
                "relocation_time": rt,
                "scope": scope,
                "timing_source": "v7_reproducible_uniform_draw",
                "random_seed": RANDOM_SEED,
                "prev_trip_id": prev_trip_id,
                "next_trip_id": next_trip_id,
            }
        )

        add_event(
            bike_id=bike_id,
            event_time=rt,
            event_type="relocation",
            source_state=src,
            destination_state=dst,
            relocation_id=rid,
            event_source=scope,
        )

    def add_v64_fixed_relocation(
        bike_id: str,
        row: pd.Series,
    ) -> None:
        nonlocal relocation_counter

        src = str(row["source_station_id"])
        dst = str(row["destination_station_id"])
        rt = pd.Timestamp(row["sampled_relocation_time"])

        lower = MAIN_START
        if "window_start" in row.index and pd.notna(row["window_start"]):
            lower = pd.Timestamp(row["window_start"])

        upper = pd.Timestamp(row["first_july_start_time"])
        if "window_end" in row.index and pd.notna(row["window_end"]):
            upper = pd.Timestamp(row["window_end"])

        if not (lower < rt < upper):
            raise RuntimeError(
                "V6.4 fixed relocation time lies outside its stored admissible "
                f"window: bike={bike_id}, lower={lower}, time={rt}, upper={upper}"
            )

        relocation_counter += 1
        rid = f"R{relocation_counter:08d}"

        relocation_rows.append(
            {
                "relocation_id": rid,
                "bike_id": bike_id,
                "source_station_id": src,
                "destination_station_id": dst,
                "window_start_exclusive": lower,
                "window_end_exclusive": upper,
                "relocation_time": rt,
                "scope": "v64_post_t0_pre_first",
                "timing_source": "v6_4_reused_exact_sampled_timestamp",
                "random_seed": (
                    row["within_side_timestamp_seed"]
                    if "within_side_timestamp_seed" in row.index
                    else np.nan
                ),
                "prev_trip_id": (
                    row["last_pre_trip_id"]
                    if "last_pre_trip_id" in row.index
                    else None
                ),
                "next_trip_id": (
                    row["first_july_trip_id"]
                    if "first_july_trip_id" in row.index
                    else None
                ),
            }
        )

        add_event(
            bike_id=bike_id,
            event_time=rt,
            event_type="relocation",
            source_state=src,
            destination_state=dst,
            relocation_id=rid,
            event_source="v64_post_t0_pre_first",
        )

    # ------------------------------------------------------------------
    # Per-bike event generation
    # ------------------------------------------------------------------
    for bike in july_bikes:
        g = trips_by_bike.get(bike)
        if g is None or len(g) == 0:
            raise RuntimeError(f"No accepted July trip for cohort bike {bike}")

        rows = list(g.itertuples(index=False))
        first = rows[0]
        first_origin = str(first.from_station_id)
        first_start = pd.Timestamp(first.start_time)

        init_state = str(initial_state_map[bike])
        vr = v64_by_bike.loc[bike]

        if str(vr["first_july_origin"]) != first_origin:
            raise RuntimeError(
                f"First origin mismatch for bike {bike}: "
                f"V6.4={vr['first_july_origin']} vs accepted={first_origin}"
            )

        # --------------------------------------------------------------
        # Initial t0 -> first accepted July trip linkage
        # --------------------------------------------------------------
        if init_state == "0":
            if bike not in active_map:
                raise RuntimeError(
                    f"V6.4 says bike {bike} is node0 at t0 but no active context trip exists."
                )

            a = active_map[bike]
            active_trip_id = str(a["trip_id"])
            active_dest = str(a["to_station_id"])

            if active_dest not in station_set:
                raise RuntimeError(
                    f"Active-at-t0 bike {bike} arrives outside station domain: {active_dest}"
                )

            clean_trip_ids = {str(r.trip_id) for r in rows}
            active_trip_is_july_main = active_trip_id in clean_trip_ids

            # If the active trip began before t0, its departure is pre-t0 and
            # only its arrival may be inside July.
            if not active_trip_is_july_main:
                active_stop = pd.Timestamp(a["stop_time"])

                if MAIN_START < active_stop <= MAIN_END_EXCLUSIVE:
                    add_event(
                        bike_id=bike,
                        event_time=active_stop,
                        event_type="user_arrival",
                        source_state="0",
                        destination_state=active_dest,
                        trip_id=active_trip_id,
                        event_source="active_t0_crossing_trip_arrival",
                    )

                if active_stop > first_start:
                    record_invalid_gap(
                        bike,
                        active_dest,
                        first_origin,
                        active_stop,
                        first_start,
                        "active_t0_to_first_july",
                        active_trip_id,
                        str(first.trip_id),
                        "first_july_trip_overlaps_active_t0_crossing_trip",
                    )
                elif active_dest != first_origin:
                    add_random_relocation(
                        bike_id=bike,
                        src=active_dest,
                        dst=first_origin,
                        lower=active_stop,
                        upper=first_start,
                        scope="active_t0_to_first_july",
                        prev_trip_id=active_trip_id,
                        next_trip_id=str(first.trip_id),
                    )

        else:
            if init_state not in station_set:
                raise RuntimeError(
                    f"Non-node0 V6.4 initial state outside domain: bike={bike}, state={init_state}"
                )

            if init_state != first_origin:
                if len(v64_post_relocations) == 0 or bike not in post_by_bike.index:
                    raise RuntimeError(
                        "A non-active bike starts away from its first July origin, "
                        "but V6.4 provides no post-t0/pre-first relocation: "
                        f"bike={bike}, t0={init_state}, first_origin={first_origin}"
                    )

                pr = post_by_bike.loc[bike]
                if isinstance(pr, pd.DataFrame):
                    raise RuntimeError(f"Duplicate V6.4 post relocation rows for bike {bike}")
                add_v64_fixed_relocation(bike, pr)
            else:
                if len(v64_post_relocations) and bike in post_by_bike.index:
                    raise RuntimeError(
                        "V6.4 post relocation exists even though t0 state already equals "
                        f"first origin: bike={bike}, station={init_state}"
                    )

        # --------------------------------------------------------------
        # Hard July user-trip events
        # --------------------------------------------------------------
        for row in rows:
            st = pd.Timestamp(row.start_time)
            en = pd.Timestamp(row.stop_time)
            src = str(row.from_station_id)
            dst = str(row.to_station_id)
            trip_id = str(row.trip_id)

            if src not in station_set or dst not in station_set:
                raise RuntimeError(
                    f"Accepted July trip endpoint outside station domain: trip={trip_id}"
                )

            # At exactly t0, V6.4 initial state already represents the departure.
            if MAIN_START < st < MAIN_END_EXCLUSIVE:
                add_event(
                    bike_id=bike,
                    event_time=st,
                    event_type="user_departure",
                    source_state=src,
                    destination_state="0",
                    trip_id=trip_id,
                    event_source="observed_july_user_trip",
                )

            # A July-starting trip may arrive exactly at Aug-1 boundary.
            if MAIN_START < en <= MAIN_END_EXCLUSIVE:
                add_event(
                    bike_id=bike,
                    event_time=en,
                    event_type="user_arrival",
                    source_state="0",
                    destination_state=dst,
                    trip_id=trip_id,
                    event_source="observed_july_user_trip",
                )

        # --------------------------------------------------------------
        # Mandatory relocations between consecutive accepted July trips
        # --------------------------------------------------------------
        for prev, nxt in zip(rows[:-1], rows[1:]):
            prev_dest = str(prev.to_station_id)
            next_origin = str(nxt.from_station_id)

            if prev_dest == next_origin:
                continue

            add_random_relocation(
                bike_id=bike,
                src=prev_dest,
                dst=next_origin,
                lower=pd.Timestamp(prev.stop_time),
                upper=pd.Timestamp(nxt.start_time),
                scope="between_july_trips",
                prev_trip_id=str(prev.trip_id),
                next_trip_id=str(nxt.trip_id),
            )

    events = pd.DataFrame(event_rows)
    relocations = pd.DataFrame(relocation_rows)
    invalid_gaps = pd.DataFrame(invalid_gap_rows)

    if len(events):
        # event_time is the physical primary key. global_insert_seq resolves
        # exact-time ties in the order the bike sequence was constructed.
        events = events.sort_values(
            ["event_time", "global_insert_seq"]
        ).reset_index(drop=True)

        # Recompute strict chronological per-bike sequence after sorting.
        events["bike_event_seq"] = (
            events.groupby("bike_id", sort=False).cumcount() + 1
        )

    if len(relocations):
        relocations = relocations.sort_values(
            ["relocation_time", "relocation_id"]
        ).reset_index(drop=True)

    print(f"Total events: {len(events):,}")
    print(f"Relocations: {len(relocations):,}")
    if len(relocations):
        print(relocations["scope"].value_counts().to_string())
    print(f"Invalid relocation gaps: {len(invalid_gaps):,}")

    return events, relocations, invalid_gaps


# =============================================================================
# BIKE EVENT STATE AUDIT
# =============================================================================

def audit_bike_event_states(
    july_bikes: List[str],
    initial_state_map: Dict[str, str],
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    banner("AUDIT BIKE-LEVEL EVENT STATE CONSISTENCY")

    state = {bike: str(initial_state_map[bike]) for bike in july_bikes}
    errors = []

    if len(events):
        ordered = events.sort_values(
            ["event_time", "global_insert_seq"]
        )
    else:
        ordered = events

    for row in ordered.itertuples(index=False):
        bike = str(row.bike_id)
        expected = str(row.source_state)
        actual = str(state[bike])

        if actual != expected:
            errors.append(
                {
                    "bike_id": bike,
                    "event_time": row.event_time,
                    "event_type": row.event_type,
                    "expected_source_state": expected,
                    "actual_state_before_event": actual,
                    "destination_state": row.destination_state,
                    "trip_id": row.trip_id,
                    "relocation_id": row.relocation_id,
                    "event_source": row.event_source,
                }
            )

        # Continue replay to expose all mismatches; no silent physical repair.
        state[bike] = str(row.destination_state)

    error_df = pd.DataFrame(errors)
    print(f"State-transition mismatches: {len(error_df):,}")

    return error_df, state


# =============================================================================
# FIXED 10-MINUTE STATE GRID
# =============================================================================

def build_state_times() -> pd.DatetimeIndex:
    times = pd.date_range(
        MAIN_START,
        MAIN_END_EXCLUSIVE,
        freq=GRID_FREQ,
        inclusive="both",
    )

    grid_seconds = float(pd.Timedelta(GRID_FREQ).total_seconds())
    if grid_seconds <= 0:
        raise RuntimeError(f"GRID_FREQ must be positive: {GRID_FREQ}")

    expected_intervals = int(
        (MAIN_END_EXCLUSIVE - MAIN_START).total_seconds() // grid_seconds
    )

    if len(times) != expected_intervals + 1:
        raise RuntimeError(
            f"Unexpected grid length: got {len(times)}, expected {expected_intervals + 1}"
        )

    return times


def build_initial_station_vector(
    v64_initial_df: pd.DataFrame,
    station_ids: List[str],
) -> Tuple[np.ndarray, int]:
    station_to_idx = {sid: j for j, sid in enumerate(station_ids)}

    B = np.zeros(len(station_ids), dtype=np.int64)
    B0 = 0

    for row in v64_initial_df.itertuples(index=False):
        state = str(row.t0_state)
        if state == "0":
            B0 += 1
        else:
            B[station_to_idx[state]] += 1

    return B, B0


def map_events_to_grid_transitions(
    events: pd.DataFrame,
    state_times: pd.DatetimeIndex,
    station_ids: List[str],
):
    K = len(state_times) - 1
    N = len(station_ids)

    station_to_idx = {sid: j for j, sid in enumerate(station_ids)}

    station_delta = np.zeros((K, N), dtype=np.int64)
    node0_delta = np.zeros(K, dtype=np.int64)

    relocation_out = np.zeros((K, N), dtype=np.int64)
    relocation_in = np.zeros((K, N), dtype=np.int64)
    user_departures = np.zeros((K, N), dtype=np.int64)
    user_arrivals = np.zeros((K, N), dtype=np.int64)
    event_counts = np.zeros(K, dtype=np.int64)

    c = state_times.to_numpy(dtype="datetime64[ns]")

    for row in events.itertuples(index=False):
        et = np.datetime64(pd.Timestamp(row.event_time), "ns")

        # State at c[idx] includes all events with event_time <= c[idx].
        idx = int(np.searchsorted(c, et, side="left"))

        if idx <= 0:
            raise RuntimeError(
                "V7 event exists at/before t0 even though t0 state is already fixed: "
                f"{row}"
            )

        if idx >= len(c):
            raise RuntimeError(
                f"V7 event lies after modeled month-end boundary: {row}"
            )

        k = idx - 1
        event_counts[k] += 1

        src = str(row.source_state)
        dst = str(row.destination_state)
        etype = str(row.event_type)

        if src == "0":
            node0_delta[k] -= 1
        else:
            station_delta[k, station_to_idx[src]] -= 1

        if dst == "0":
            node0_delta[k] += 1
        else:
            station_delta[k, station_to_idx[dst]] += 1

        if etype == "relocation":
            if src == "0" or dst == "0":
                raise RuntimeError("Relocation illegally touches node0.")
            relocation_out[k, station_to_idx[src]] += 1
            relocation_in[k, station_to_idx[dst]] += 1

        elif etype == "user_departure":
            if dst != "0":
                raise RuntimeError("user_departure destination must be node0.")
            user_departures[k, station_to_idx[src]] += 1

        elif etype == "user_arrival":
            if src != "0":
                raise RuntimeError("user_arrival source must be node0.")
            user_arrivals[k, station_to_idx[dst]] += 1

        else:
            raise ValueError(f"Unknown event type: {etype}")

    return {
        "station_delta": station_delta,
        "node0_delta": node0_delta,
        "relocation_out": relocation_out,
        "relocation_in": relocation_in,
        "user_departures": user_departures,
        "user_arrivals": user_arrivals,
        "event_counts": event_counts,
    }


def reconstruct_inventory_from_events(
    initial_station: np.ndarray,
    initial_node0: int,
    mapped: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    K, N = mapped["station_delta"].shape

    B = np.zeros((K + 1, N), dtype=np.int64)
    B0 = np.zeros(K + 1, dtype=np.int64)

    B[0] = initial_station
    B0[0] = int(initial_node0)

    for k in range(K):
        B[k + 1] = B[k] + mapped["station_delta"][k]
        B0[k + 1] = B0[k] + mapped["node0_delta"][k]

    return B, B0


# =============================================================================
# AVAILABLE_BIKES / TOTAL_DOCKS COMPARISON ONLY
# =============================================================================

def load_inventory_for_comparison(
    path: Path,
    station_ids: List[str],
) -> dict:
    banner("LOAD INVENTORY FOR COMPARISON ONLY")

    header = pd.read_csv(path, nrows=0).columns.tolist()

    aliases = {
        "station_id": ["id", "station_id"],
        "timestamp": ["timestamp"],
        "available_bikes": ["available_bikes"],
        "total_docks": ["total_docks"],
    }

    resolved = {}
    for canonical, candidates in aliases.items():
        for c in candidates:
            if c in header:
                resolved[canonical] = c
                break

    required = ["station_id", "timestamp", "available_bikes"]
    missing = [c for c in required if c not in resolved]
    if missing:
        raise ValueError(f"Inventory missing columns {missing}; available={header}")

    inv = pd.read_csv(
        path,
        usecols=list(dict.fromkeys(resolved.values())),
        low_memory=False,
    ).rename(columns={actual: canonical for canonical, actual in resolved.items()})

    inv["station_id"] = normalize_id_series(inv["station_id"])
    inv["timestamp"] = pd.to_datetime(inv["timestamp"], errors="coerce")
    inv["available_bikes"] = pd.to_numeric(inv["available_bikes"], errors="coerce")

    if "total_docks" in inv.columns:
        inv["total_docks"] = pd.to_numeric(inv["total_docks"], errors="coerce")

    station_set = set(station_ids)
    inv = inv[
        inv["station_id"].isin(station_set)
        & inv["timestamp"].notna()
        & (inv["timestamp"] >= MAIN_START)
        & (inv["timestamp"] < MAIN_END_EXCLUSIVE)
    ].copy()

    dup = inv.duplicated(["timestamp", "station_id"], keep=False)
    if dup.any():
        d = inv[dup]
        conflicts = (
            d.groupby(["timestamp", "station_id"])["available_bikes"]
            .nunique(dropna=False)
            .gt(1)
        )
        if conflicts.any():
            raise RuntimeError(
                "Conflicting duplicate inventory cells. Examples="
                + repr(conflicts[conflicts].index.tolist()[:10])
            )

    inv = (
        inv.sort_values(["timestamp", "station_id"])
        .drop_duplicates(["timestamp", "station_id"], keep="last")
        .copy()
    )

    return {"inventory_long": inv}


def build_available_grid_comparison(
    inv: dict,
    state_times: pd.DatetimeIndex,
    station_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a comparison-only grid-aligned matrix by time interpolation of observed
    available_bikes. V6.4's own back-propagated t0 reference is injected at t0.
    No value here changes bike-ID dynamics.
    """
    banner("BUILD GRID-ALIGNED AVAILABLE_BIKES COMPARISON")

    long = inv["inventory_long"]

    wide = (
        long.pivot(
            index="timestamp",
            columns="station_id",
            values="available_bikes",
        )
        .reindex(columns=station_ids)
        .sort_index()
    )

    raw_observed = wide.copy()

    # Lifecycle outer zeros on comparison series only.
    for sid, rule in LIFECYCLE_RULES.items():
        if sid not in wide.columns:
            continue
        obs = wide[sid].dropna()
        if obs.empty:
            continue
        first_t = obs.index.min()
        last_t = obs.index.max()

        if rule == "zero_before_first_observation":
            idx = wide.index < first_t
            wide.loc[idx & wide[sid].isna(), sid] = 0.0
        elif rule == "zero_after_last_observation":
            idx = wide.index > last_t
            wide.loc[idx & wide[sid].isna(), sid] = 0.0
        else:
            raise ValueError(f"Unknown lifecycle rule: {rule}")

    # Interpolate on union grid, only inside observed/lifecycle-supported span.
    union_index = wide.index.union(state_times).sort_values()
    comp_union = wide.reindex(union_index).interpolate(
        method="time",
        axis=0,
        limit_area="inside",
    )

    comp = comp_union.reindex(state_times).to_numpy(dtype=np.float64)

    # Exact raw observations only where inventory timestamps equal grid boundaries.
    raw_grid = raw_observed.reindex(state_times).to_numpy(dtype=np.float64)

    source = np.full(comp.shape, "time_interpolated_or_missing", dtype=object)
    source[np.isfinite(comp)] = "time_interpolated_comparison"
    source[np.isfinite(raw_grid)] = "raw_observed_exact_grid_time"

    # Inject exact V6.4 t0 spatial reference counts if available.
    if V64_T0_AVAILABLE_REFERENCE_PATH.exists():
        ref = pd.read_csv(V64_T0_AVAILABLE_REFERENCE_PATH, low_memory=False)
        if {"station_id", "available_bikes_t0_reference"}.issubset(ref.columns):
            ref["station_id"] = normalize_id_series(ref["station_id"])
            ref["available_bikes_t0_reference"] = pd.to_numeric(
                ref["available_bikes_t0_reference"], errors="coerce"
            )
            t0_vec = (
                ref.set_index("station_id")
                .reindex(station_ids)["available_bikes_t0_reference"]
                .to_numpy(dtype=float)
            )
            comp[0] = t0_vec
            source[0] = "v6_4_t0_available_reference"

    print(f"grid comparison finite cells: {int(np.isfinite(comp).sum()):,}/{comp.size:,}")

    return raw_grid, comp, source


def build_capacity_grid(
    inv: dict,
    state_times: pd.DatetimeIndex,
    station_ids: List[str],
) -> Optional[np.ndarray]:
    long = inv["inventory_long"]
    if "total_docks" not in long.columns:
        return None

    td = (
        long.pivot(
            index="timestamp",
            columns="station_id",
            values="total_docks",
        )
        .reindex(columns=station_ids)
        .sort_index()
    )

    union = td.index.union(state_times).sort_values()
    td_union = td.reindex(union).ffill().bfill()
    return td_union.reindex(state_times).to_numpy(dtype=float)


# =============================================================================
# OUTPUT BUILDERS
# =============================================================================

def build_system_totals(
    state_times: pd.DatetimeIndex,
    B: np.ndarray,
    B0: np.ndarray,
    available_comp: np.ndarray,
    cohort_size: int,
) -> pd.DataFrame:
    station_sum = B.sum(axis=1)
    total = station_sum + B0
    available_sum = np.nansum(available_comp, axis=1)
    available_n = np.isfinite(available_comp).sum(axis=1)

    return pd.DataFrame(
        {
            "timestamp": state_times,
            "reconstructed_station_bikes": station_sum,
            "reconstructed_user_in_transit": B0,
            "reconstructed_total": total,
            "fixed_july_bike_cohort": cohort_size,
            "total_minus_fixed_cohort": total - cohort_size,
            "available_comparison_station_sum": available_sum,
            "available_comparison_observed_or_interpolated_station_count": available_n,
            "reconstructed_station_minus_available_comparison": (
                station_sum - available_sum
            ),
        }
    )


def build_long_state_comparison(
    state_times: pd.DatetimeIndex,
    station_ids: List[str],
    B: np.ndarray,
    available_raw_grid: np.ndarray,
    available_comp: np.ndarray,
    available_source: np.ndarray,
) -> pd.DataFrame:
    T = len(state_times)
    N = len(station_ids)

    return pd.DataFrame(
        {
            "timestamp": np.repeat(
                state_times.to_numpy(dtype="datetime64[ns]"), N
            ),
            "station_id": np.tile(np.asarray(station_ids, dtype=object), T),
            "bike_id_reconstructed_stock": B.reshape(-1),
            "available_bikes_raw_exact_grid_time": available_raw_grid.reshape(-1),
            "available_bikes_for_comparison": available_comp.reshape(-1),
            "available_inventory_source": available_source.reshape(-1),
            "reconstructed_minus_available": (
                B.astype(float) - available_comp
            ).reshape(-1),
        }
    )


def build_sparse_user_od_flows(july_trips_clean: pd.DataFrame) -> pd.DataFrame:
    x = july_trips_clean.copy()
    x["interval_start"] = event_to_interval_start(x["start_time"])
    x["flow_type"] = "user_trip"
    x["interval_end"] = x["interval_start"] + pd.Timedelta(GRID_FREQ)

    out = (
        x.groupby(
            ["interval_start", "interval_end", "from_station_id", "to_station_id", "flow_type"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "from_station_id": "origin_station_id",
                "to_station_id": "destination_station_id",
                "size": "count",
            }
        )
    )
    return out


def build_sparse_relocation_od_flows(relocations: pd.DataFrame) -> pd.DataFrame:
    if len(relocations) == 0:
        return pd.DataFrame(
            columns=[
                "interval_start",
                "interval_end",
                "origin_station_id",
                "destination_station_id",
                "flow_type",
                "count",
            ]
        )

    x = relocations.copy()
    x["interval_start"] = event_to_interval_start(x["relocation_time"])
    x["flow_type"] = "relocation"
    x["interval_end"] = x["interval_start"] + pd.Timedelta(GRID_FREQ)

    out = (
        x.groupby(
            [
                "interval_start",
                "interval_end",
                "source_station_id",
                "destination_station_id",
                "flow_type",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "source_station_id": "origin_station_id",
                "size": "count",
            }
        )
    )
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    banner("DIVVY 2017 JULY CLOSED SYSTEM V7 10-MIN FROM V6.4")

    for p in [
        TRIP_CONTEXT_PATH,
        TRIP_MAIN_PATH,
        INVENTORY_MAIN_PATH,
        V64_T0_BIKES_PATH,
        V64_POST_RELOCATIONS_PATH,
    ]:
        require_file(p)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Raw root: {RAW_ROOT}")
    print(f"V6.4 root: {V64_ROOT}")
    print(f"Output root: {OUT_ROOT}")
    print(f"New-relocation random seed: {RANDOM_SEED}")
    print("V6.4 is the ONLY t0 state source.")
    print("V6.4 post-t0/pre-first relocation timestamps are reused exactly.")
    print("available_bikes and total_docks are audit/comparison only.")
    print("NO bike insertion/removal. NO hidden node. NO third-station repair.")

    rng = np.random.default_rng(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Load trips and establish current raw July domain.
    # ------------------------------------------------------------------
    trips_context = load_trips(TRIP_CONTEXT_PATH, "TRIP CONTEXT")
    trips_main = load_trips(TRIP_MAIN_PATH, "TRIP MAIN")

    july_main_raw, july_bikes, station_ids, domain_df = (
        build_july_cohort_and_station_domain(trips_main)
    )

    july_main, overlap_anomalies = remove_same_bike_trip_overlaps(july_main_raw)

    # ------------------------------------------------------------------
    # Load fixed V6.4 initial state and active crossing-trip evidence.
    # ------------------------------------------------------------------
    v64_initial_df, initial_state_map, v64_post_relocations = (
        load_v64_initial_state(
            july_bikes,
            station_ids,
            july_main,
        )
    )

    active_map = build_active_t0_map(trips_context, july_bikes)

    v64_node0_bikes = set(
        v64_initial_df.loc[
            v64_initial_df["t0_state"].astype(str) == "0", "bike_id"
        ].astype(str)
    )
    active_context_bikes = set(active_map)

    if v64_node0_bikes != active_context_bikes:
        raise RuntimeError(
            "V6.4 node0 set differs from active-at-t0 context set. "
            f"only_v64={sorted(v64_node0_bikes-active_context_bikes)[:20]}, "
            f"only_context={sorted(active_context_bikes-v64_node0_bikes)[:20]}"
        )

    # ------------------------------------------------------------------
    # Build event stream.
    # ------------------------------------------------------------------
    events_df, relocations_df, invalid_gaps_df = build_v7_events(
        july_trips_clean=july_main,
        july_bikes=july_bikes,
        station_ids=station_ids,
        initial_state_map=initial_state_map,
        v64_initial_df=v64_initial_df,
        v64_post_relocations=v64_post_relocations,
        active_map=active_map,
        rng=rng,
    )

    invalid_gap_failure = bool(
        FAIL_ON_INVALID_RELOCATION_GAP and len(invalid_gaps_df) > 0
    )

    event_state_errors_df, final_bike_state_map = audit_bike_event_states(
        july_bikes,
        initial_state_map,
        events_df,
    )

    event_state_failure = bool(
        FAIL_ON_EVENT_STATE_MISMATCH and len(event_state_errors_df) > 0
    )

    # ------------------------------------------------------------------
    # Reconstruct fixed 10-minute state grid.
    # ------------------------------------------------------------------
    state_times = build_state_times()
    initial_station, initial_node0 = build_initial_station_vector(
        v64_initial_df,
        station_ids,
    )

    mapped = map_events_to_grid_transitions(
        events_df,
        state_times,
        station_ids,
    )

    B, B0 = reconstruct_inventory_from_events(
        initial_station,
        initial_node0,
        mapped,
    )

    cohort_size = len(july_bikes)
    reconstructed_total = B.sum(axis=1) + B0
    total_error = reconstructed_total - cohort_size

    negative_station_cells = int((B < 0).sum())
    negative_node0_cells = int((B0 < 0).sum())
    min_station_stock = int(B.min())
    min_node0_stock = int(B0.min())
    max_abs_mass_error = int(np.max(np.abs(total_error)))

    banner("V7 PHYSICS AUDIT")
    print(f"Fixed July bike cohort: {cohort_size:,}")
    print(f"State boundaries: {len(state_times):,}")
    print(f"10-minute intervals: {len(state_times)-1:,}")
    print(
        "Reconstructed total min/median/max: "
        f"{reconstructed_total.min()}/"
        f"{np.median(reconstructed_total):.0f}/"
        f"{reconstructed_total.max()}"
    )
    print(f"max |total - fixed cohort|: {max_abs_mass_error}")
    print(f"Minimum station stock: {min_station_stock}")
    print(f"Negative station cells: {negative_station_cells:,}")
    print(f"Minimum node0 stock: {min_node0_stock}")
    print(f"Negative node0 cells: {negative_node0_cells:,}")

    mass_failure = bool(
        FAIL_ON_MASS_ERROR and max_abs_mass_error != 0
    )
    negative_stock_failure = bool(
        FAIL_ON_NEGATIVE_STOCK
        and (negative_station_cells != 0 or negative_node0_cells != 0)
    )

    # ------------------------------------------------------------------
    # Inventory comparison/audit only.
    # ------------------------------------------------------------------
    inv = load_inventory_for_comparison(INVENTORY_MAIN_PATH, station_ids)
    available_raw_grid, available_comp, available_source = (
        build_available_grid_comparison(
            inv,
            state_times,
            station_ids,
        )
    )

    capacity_grid = build_capacity_grid(
        inv,
        state_times,
        station_ids,
    )

    capacity_exceed_cells = 0
    max_capacity_excess = np.nan

    if capacity_grid is not None:
        cap_mask = np.isfinite(capacity_grid)
        exceed = cap_mask & (B > capacity_grid)
        capacity_exceed_cells = int(exceed.sum())
        if exceed.any():
            max_capacity_excess = float(
                np.max(np.where(exceed, B - capacity_grid, 0.0))
            )

    system_totals_df = build_system_totals(
        state_times,
        B,
        B0,
        available_comp,
        cohort_size,
    )

    # ------------------------------------------------------------------
    # Sparse OD flows.
    # ------------------------------------------------------------------
    user_od = build_sparse_user_od_flows(july_main)
    relocation_od = build_sparse_relocation_od_flows(relocations_df)
    all_od = pd.concat([user_od, relocation_od], ignore_index=True)
    if len(all_od):
        all_od = all_od.sort_values(
            [
                "interval_start",
                "interval_end",
                "flow_type",
                "origin_station_id",
                "destination_station_id",
            ]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    banner("WRITE V7 OUTPUTS")

    p_domain = OUT_ROOT / "active_stations_v7.csv"
    p_trips = OUT_ROOT / "accepted_july_trips_v7.csv.gz"
    p_overlap = OUT_ROOT / "same_bike_overlap_anomalies_excluded_v7.csv"
    p_initial = OUT_ROOT / "initial_bike_states_from_v64.csv"
    p_events = OUT_ROOT / "bike_events_v7.csv.gz"
    p_reloc = OUT_ROOT / "relocations_v7.csv.gz"
    p_invalid = OUT_ROOT / "invalid_relocation_gaps_v7.csv"
    p_event_errors = OUT_ROOT / "bike_event_state_errors_v7.csv"
    p_system = OUT_ROOT / "system_totals_v7.csv"
    p_recon_wide = OUT_ROOT / "reconstructed_bike_inventory_10min_wide_v7.csv.gz"
    p_avail_wide = OUT_ROOT / "available_bikes_10min_comparison_wide_v7.csv.gz"
    p_long = OUT_ROOT / "station_state_comparison_10min_v7.csv.gz"
    p_user_od = OUT_ROOT / "user_od_flows_10min_sparse_v7.csv.gz"
    p_reloc_od = OUT_ROOT / "relocation_od_flows_10min_sparse_v7.csv.gz"
    p_all_od = OUT_ROOT / "all_od_flows_10min_sparse_v7.csv.gz"
    p_npz = OUT_ROOT / "closed_system_v7_10min_arrays.npz"
    p_json = OUT_ROOT / "summary_v7.json"
    p_txt = OUT_ROOT / "summary_v7.txt"

    domain_df.to_csv(p_domain, index=False)
    july_main.to_csv(p_trips, index=False, compression="gzip")
    overlap_anomalies.to_csv(p_overlap, index=False)
    v64_initial_df.to_csv(p_initial, index=False)
    events_df.to_csv(p_events, index=False, compression="gzip")
    relocations_df.to_csv(p_reloc, index=False, compression="gzip")
    invalid_gaps_df.to_csv(p_invalid, index=False)
    event_state_errors_df.to_csv(p_event_errors, index=False)
    system_totals_df.to_csv(p_system, index=False)

    if WRITE_WIDE_MATRICES:
        recon_wide = pd.DataFrame(B, columns=station_ids)
        recon_wide.insert(0, "timestamp", state_times)
        recon_wide.insert(1, "user_in_transit_node0", B0)
        recon_wide.to_csv(p_recon_wide, index=False, compression="gzip")

        avail_wide = pd.DataFrame(available_comp, columns=station_ids)
        avail_wide.insert(0, "timestamp", state_times)
        avail_wide.to_csv(p_avail_wide, index=False, compression="gzip")

    if WRITE_LONG_STATE_COMPARISON:
        long_df = build_long_state_comparison(
            state_times,
            station_ids,
            B,
            available_raw_grid,
            available_comp,
            available_source,
        )
        long_df.to_csv(p_long, index=False, compression="gzip")

    if WRITE_SPARSE_FLOWS:
        user_od.to_csv(p_user_od, index=False, compression="gzip")
        relocation_od.to_csv(p_reloc_od, index=False, compression="gzip")
        all_od.to_csv(p_all_od, index=False, compression="gzip")

    if WRITE_NPZ:
        np.savez_compressed(
            p_npz,
            station_ids=np.asarray(station_ids, dtype=object),
            july_bike_ids=np.asarray(july_bikes, dtype=object),
            state_times_ns=state_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            reconstructed_bike_inventory=B.astype(np.int32),
            reconstructed_user_in_transit=B0.astype(np.int32),
            available_bikes_comparison=available_comp.astype(np.float32),
            station_delta=mapped["station_delta"].astype(np.int32),
            node0_delta=mapped["node0_delta"].astype(np.int32),
            user_departures=mapped["user_departures"].astype(np.int32),
            user_arrivals=mapped["user_arrivals"].astype(np.int32),
            relocation_out=mapped["relocation_out"].astype(np.int32),
            relocation_in=mapped["relocation_in"].astype(np.int32),
            event_counts=mapped["event_counts"].astype(np.int32),
        )

    relocation_scope_counts = (
        relocations_df["scope"].value_counts().to_dict()
        if len(relocations_df)
        else {}
    )

    summary = {
        "version": "V7_10min_from_V6_4",
        "study_window": {
            "start": MAIN_START,
            "end_exclusive": MAIN_END_EXCLUSIVE,
            "grid_frequency": GRID_FREQ,
            "state_boundary_count": len(state_times),
            "interval_count": len(state_times) - 1,
            "physical_transition_interval_definition": (
                "I0=[t0,t1], Ik=(tk,t{k+1}] for k>=1; state at each boundary "
                "includes events exactly at that boundary"
            ),
            "user_od_time_assignment": (
                "Trip start time mapped to the same right-closed reporting bins; "
                "a trip starting exactly at t0 is assigned to the first reporting interval"
            ),
        },
        "fixed_cohort": {
            "july_bike_count": cohort_size,
            "definition": (
                "Exact set of bike IDs appearing in raw July main-window trips; "
                "verified equal to the V6.4 t0 cohort."
            ),
        },
        "initial_condition": {
            "source": str(V64_T0_BIKES_PATH),
            "v64_is_only_t0_state_source": True,
            "initial_station_bikes": int(B[0].sum()),
            "initial_node0_bikes": int(B0[0]),
            "initial_total": int(reconstructed_total[0]),
            "v64_post_t0_pre_first_relocations_reused": len(v64_post_relocations),
        },
        "trips": {
            "raw_july_trip_rows": len(july_main_raw),
            "accepted_july_trip_rows": len(july_main),
            "excluded_same_bike_overlap_rows": len(overlap_anomalies),
        },
        "events": {
            "total_event_count": len(events_df),
            "user_departures": int((events_df["event_type"] == "user_departure").sum()),
            "user_arrivals": int((events_df["event_type"] == "user_arrival").sum()),
            "relocations": len(relocations_df),
            "relocation_scope_counts": relocation_scope_counts,
            "new_relocation_random_seed": RANDOM_SEED,
            "invalid_relocation_gap_count": len(invalid_gaps_df),
            "bike_event_state_mismatch_count": len(event_state_errors_df),
        },
        "physics": {
            "reconstructed_total_min": int(reconstructed_total.min()),
            "reconstructed_total_max": int(reconstructed_total.max()),
            "max_abs_total_minus_fixed_cohort": max_abs_mass_error,
            "minimum_station_stock": min_station_stock,
            "negative_station_cells": negative_station_cells,
            "minimum_node0_stock": min_node0_stock,
            "negative_node0_cells": negative_node0_cells,
            "mass_conservation_holds": bool(max_abs_mass_error == 0),
            "nonnegative_stock_holds": bool(
                negative_station_cells == 0 and negative_node0_cells == 0
            ),
            "bike_event_state_consistency_holds": bool(
                len(event_state_errors_df) == 0
            ),
        },
        "inventory_audit_only": {
            "available_bikes_changes_trajectory": False,
            "total_docks_changes_trajectory": False,
            "capacity_exceed_cells": capacity_exceed_cells,
            "max_capacity_excess": max_capacity_excess,
        },
        "outputs": {
            "event_level_is_physical_master_record": True,
            "state_grid_contains_aug1_boundary": True,
            "sparse_od_user_flow_rows": len(user_od),
            "sparse_od_relocation_flow_rows": len(relocation_od),
        },
    }

    with p_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=jsonable)

    lines = [
        "DIVVY 2017 JULY CLOSED SYSTEM V7 10-MIN FROM V6.4",
        "=" * 90,
        "",
        f"study_start: {MAIN_START}",
        f"study_end_exclusive: {MAIN_END_EXCLUSIVE}",
        f"grid_frequency: {GRID_FREQ}",
        f"state_boundaries: {len(state_times)}",
        f"five_minute_intervals: {len(state_times)-1}",
        "",
        "INITIAL CONDITION",
        "V6.4 is the only t0 bike-state source.",
        f"fixed cohort: {cohort_size}",
        f"initial station bikes: {int(B[0].sum())}",
        f"initial node0 bikes: {int(B0[0])}",
        f"initial total: {int(reconstructed_total[0])}",
        f"V6.4 post-t0/pre-first relocations reused exactly: {len(v64_post_relocations)}",
        "",
        "TRIPS",
        f"raw July trips: {len(july_main_raw)}",
        f"accepted July trips: {len(july_main)}",
        f"excluded same-bike overlap trips: {len(overlap_anomalies)}",
        "",
        "EVENTS",
        f"total events: {len(events_df)}",
        f"relocations: {len(relocations_df)}",
        f"invalid relocation gaps: {len(invalid_gaps_df)}",
        f"bike-event state mismatches: {len(event_state_errors_df)}",
        f"new relocation timing seed: {RANDOM_SEED}",
        "",
        "PHYSICS",
        f"reconstructed total min: {int(reconstructed_total.min())}",
        f"reconstructed total max: {int(reconstructed_total.max())}",
        f"max abs(total-fixed cohort): {max_abs_mass_error}",
        f"minimum station stock: {min_station_stock}",
        f"negative station cells: {negative_station_cells}",
        f"minimum node0 stock: {min_node0_stock}",
        f"negative node0 cells: {negative_node0_cells}",
        "",
        "AUDIT-ONLY EXTERNAL INVENTORY",
        "available_bikes does not change the bike-ID trajectory.",
        "total_docks does not change the bike-ID trajectory.",
        f"capacity exceed cells: {capacity_exceed_cells}",
        f"max capacity excess: {max_capacity_excess}",
        "",
        "INVARIANTS",
        "No bike insertion/removal.",
        "No hidden/depot node.",
        "No third-station repair.",
        "User trips use observed endpoints and timestamps.",
        "V6.4 post-t0/pre-first relocation timestamps are never resampled.",
    ]

    p_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for p in [
        p_domain,
        p_trips,
        p_overlap,
        p_initial,
        p_events,
        p_reloc,
        p_invalid,
        p_event_errors,
        p_system,
    ]:
        print(f"Saved: {p}")

    if WRITE_WIDE_MATRICES:
        print(f"Saved: {p_recon_wide}")
        print(f"Saved: {p_avail_wide}")
    if WRITE_LONG_STATE_COMPARISON:
        print(f"Saved: {p_long}")
    if WRITE_SPARSE_FLOWS:
        print(f"Saved: {p_user_od}")
        print(f"Saved: {p_reloc_od}")
        print(f"Saved: {p_all_od}")
    if WRITE_NPZ:
        print(f"Saved: {p_npz}")
    print(f"Saved: {p_json}")
    print(f"Saved: {p_txt}")

    banner("FINAL V7 SNAPSHOT")
    print(f"Fixed July bikes = {cohort_size:,}")
    print(f"Stations = {len(station_ids):,}")
    print(
        f"Initial station/node0/total = "
        f"{B[0].sum():,}/{B0[0]:,}/{reconstructed_total[0]:,}"
    )
    print(
        f"Reconstructed total min/max = "
        f"{reconstructed_total.min():,}/{reconstructed_total.max():,}"
    )
    print(f"max |total-fixed cohort| = {max_abs_mass_error}")
    print(f"Minimum station stock = {min_station_stock}")
    print(f"Negative station cells = {negative_station_cells:,}")
    print(f"Minimum node0 stock = {min_node0_stock}")
    print(f"Negative node0 cells = {negative_node0_cells:,}")
    print(f"Relocations = {len(relocations_df):,}")
    print(f"Invalid relocation gaps = {len(invalid_gaps_df):,}")
    print(f"Bike-event mismatches = {len(event_state_errors_df):,}")

    hard_failures = []
    if invalid_gap_failure:
        hard_failures.append(
            f"invalid relocation gaps={len(invalid_gaps_df)}"
        )
    if event_state_failure:
        hard_failures.append(
            f"bike-event state mismatches={len(event_state_errors_df)}"
        )
    if mass_failure:
        hard_failures.append(
            f"max mass error={max_abs_mass_error}"
        )
    if negative_stock_failure:
        hard_failures.append(
            "negative stock cells="
            f"{negative_station_cells + negative_node0_cells}"
        )

    if hard_failures:
        raise RuntimeError(
            "V7 outputs were written for diagnosis, but the physical audit "
            "FAILED: " + "; ".join(hard_failures)
        )

    print("DONE.")


if __name__ == "__main__":
    main()
