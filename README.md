# PDSTGCN-1 — Divvy 2017 physics-consistent training project

This project is the Divvy adaptation of `StongKing/ASTGCN-r-pytorch-huaqi-dynamic-df` for the physics-guided dynamic graph forecasting pipeline described in the manuscript.

## What changed for Divvy

The old Jersey City experiment used 5-minute data and a 12-step input / 12-step target.  Divvy V7 is a **10-minute** closed-system reconstruction, so the task is now:

- 6 historical states = 1 hour;
- predict 6 future states = 1 hour;
- `points_per_hour = 6`;
- `len_input = 6`;
- `num_for_predict = 6`;
- valid sample loss at the two boundaries is `6 + 6 - 1 = 11`, replacing the old implicit `12 + 12 - 1 = 23` logic.

The model graph contains the V7 active station set **plus node 0**, the manuscript's mesoscopic in-transit node.  `data_generate.py` reads the actual number of V7 stations and the actual fixed fleet size and rewrites the config automatically; it does not rely on the template values in `DIVVY_astgcn.conf`.

## Data provenance and physical semantics

`data_generate.py` expects the outputs of the supplied V6.4 -> V7 reconstruction. It never reconstructs or guesses bike locations again.

Signal channel 0 is:

- node 0: `reconstructed_user_in_transit`;
- real stations: `reconstructed_bike_inventory`.

Channels 1 and 2 are time-of-day and day-of-week encodings.

The predefined dynamic graph is constructed from physical V7 transitions during the same historical hour as each input sample:

- station -> 0: observed user departures;
- 0 -> station: observed user arrivals;
- station -> station: inferred operator relocations from the V7 relocation OD file.

The graph is directed and weighted. The model creates forward and backward diffusion matrices from it.

The static ASTGCN graph is generated from **training-period support only** to avoid using validation/test OD topology when training. It is made undirected for the Chebyshev branch.

## Why the dynamic graph is sparse on disk

With roughly 568 model nodes, storing thousands of `(N,N)` `float32` matrices densely would require several GB.  `dynamic_graph_sparse.npz` therefore stores packed `(src,dst,weight)` edges and graph pointers.  `lib/utils.py` materializes only the mini-batch as `(B,N,N)` immediately before the forward pass. The network interface is unchanged.

## Directory layout

```text
PDSTGCN-1/
├─ configurations/
│  └─ DIVVY_astgcn.conf
├─ data/
│  └─ DIVVY/                    # generated here
├─ lib/
│  ├─ metrics.py
│  └─ utils.py
├─ model/
│  └─ ASTGCN_r.py               # PDST-GCN static + two dynamic branches
├─ upstream_divvy_pipeline/     # supplied V6.4/V7 reconstruction scripts
├─ legacy_reference/            # old JC data/dynamic/static generators
├─ data_generate.py
├─ prepareData.py
├─ train_ASTGCN_r.py
├─ loadData.py
└─ requirements.txt
```

## 1. Build the V7 physics-consistent Divvy reconstruction

The default paths in the supplied scripts are under:

```text
D:/physic_predict_bike/
```

The final input expected by `data_generate.py` is primarily:

```text
D:/physic_predict_bike/divvy_2017_physics_closed_active_v7_10min_from_v64/
    closed_system_v7_10min_arrays.npz
    user_od_flows_10min_sparse_v7.csv.gz
    relocation_od_flows_10min_sparse_v7.csv.gz
    active_stations_v7.csv
```

For geographic coordinates it also uses, when available:

```text
D:/physic_predict_bike/divvy_2017_physics_raw/metadata/
    station_metadata_from_historical_window.csv
```

If your local paths differ, edit the **USER SETTINGS** at the top of `data_generate.py`.

## 2. Generate Divvy model data, dynamic graph, and static graph

From the `PDSTGCN-1` directory:

```bash
python data_generate.py
```

Expected products under `data/DIVVY/`:

```text
DIVVY.npz                       # [T,N,3] raw graph signal
DIVVY.csv                       # static graph edge list
static_adjacency.npz            # static connectivity + distance matrices
adj_DIVVY.pkl
adj_DIVVY_distance.pkl
node_mapping.csv
dynamic_graph_sparse.npz        # one sample-aligned historical-hour graph per sample
sample_index.csv
dataset_meta.json
```

The script also rewrites `configurations/DIVVY_astgcn.conf` with the actual `num_of_vertices` and `fleet_size`.

## 3. Materialize chronological 6 -> 6 samples

```bash
python prepareData.py --config configurations/DIVVY_astgcn.conf
```

This generates:

```text
data/DIVVY/DIVVY_r1_d0_w0_astcgn.npz
```

The split is chronological `60% / 20% / 20%`.  Normalization statistics are computed from the training split only.  Targets remain in original bike-count units.

`graph_id` is saved explicitly for every sample.  Consequently, shuffling the training `DataLoader` also shuffles the graph IDs together with `X` and `Y`; the dynamic graph no longer becomes misaligned when training samples are shuffled.

## 4. Train

Install dependencies:

```bash
pip install -r requirements.txt
```

Then train:

```bash
python train_ASTGCN_r.py --config configurations/DIVVY_astgcn.conf
```

For prediction from an existing checkpoint:

```bash
python train_ASTGCN_r.py --config configurations/DIVVY_astgcn.conf --predict-only --epoch 30
```

## Important parameter changes

The supplied Divvy config starts with:

```ini
points_per_hour = 6
num_for_predict = 6
len_input = 6
num_of_hours = 1
in_channels = 3
batch_size = 8
loss_function = mse
metric_method = unmask
```

`batch_size` is reduced from the small 68-node experiment because the attention and dynamic graph tensors scale approximately with `N^2`. Start with 8; increase only after checking GPU memory.

`metric_method = unmask` is intentional: **zero inventory is a valid bike count, not a missing value**.

The old model's hard-coded physical fleet total `494` has been removed. `fleet_size` is read from the V7 conserved total and passed into the physics-guided adjustment module.

## Reproducibility / leakage notes

- train/validation/test are chronological;
- input normalization uses training statistics only;
- every dynamic graph is indexed explicitly together with its sample;
- training shuffling cannot separate a sample from its graph;
- static graph support is restricted to the training period;
- validation/test dynamic graphs use only the historical graph window associated with that forecasting sample;
- V7 bike states and relocations are not re-inferred in this project.

## Relationship to the original GitHub project

This package preserves the original project's training interfaces and PDST-GCN ideas, but removes old PEMS/JC data, IDE caches, historical checkpoints and multi-GB generated artifacts from the downloadable bundle. The old JC data-generation scripts are retained under `legacy_reference/`, while the Divvy V6.4/V7 source scripts supplied for this task are retained under `upstream_divvy_pipeline/`.
