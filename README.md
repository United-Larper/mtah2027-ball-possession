# Ball possession for More Than A Hack 2027 (Initial Challenge)

Team **RickRolled**. This repository labels every frame of the tactical video of
FC Barcelona vs Athletic Club as `H` (home in possession), `A` (away in possession),
`D` (dead ball) or `X` (outside the graded halves), and regenerates our submission file
from the data bundle the organisers provide.

The technical document is [docs/RickRolled.md](docs/RickRolled.md) (submitted as `RickRolled.pdf`).

## Reproduce the submission

Requirements: [uv](https://docs.astral.sh/uv/) and the competition data bundle. Python 3.12 and
all pinned dependencies are installed by uv from `uv.lock`.

1. Put the data bundle in `./data` with **both** folders as delivered, `data/tactical/` and
   `data/broadcast/` (the broadcast tracking and homography are inputs), or point
   `MTAH_DATA_DIR` at the folder that holds them. The videos are not needed.
2. Run:

```bash
uv sync
uv run python scripts/run.py
```

Without uv: use Python 3.12 and the same pinned versions, exported from `uv.lock`:

```bash
python -m pip install -r requirements.txt
python scripts/run.py
```

This writes `outputs/RickRolled.csv` (199,319 rows) and validates it with the organisers'
`validate_submission` logic. The first run takes about a minute because it parses the
tracking and synchronises the broadcast recording; later runs take about 15 seconds.

Expected SHA-256 of `outputs/RickRolled.csv`:

```
d118760450ec0a3c076840d8bd0c2526446dd03a8121d82e3a3c6110f3d4d46d
```

Check it with `sha256sum outputs/RickRolled.csv` (Linux/macOS) or
`Get-FileHash outputs\RickRolled.csv` (Windows PowerShell).

The trained models (`models/*.txt`), the decoder parameters (`models/decoder.json`) and the
auxiliary-evidence tables (`models/aux_tables.json`) are committed, so the run involves no
training and no randomness. We checked that a fresh clone reproduces the file byte for byte.

## How it works

1. **Inputs used.** Tactical tracking (players and ball), broadcast tracking and homography,
   Smart Tagging. Not used: either video, the tactical homography (only to estimate the camera
   position for analysis), audio.
2. **Broadcast synchronisation** (`possession/sync_broadcast.py`). The broadcast recording is
   aligned to the tactical timeline by matching the two ball tracks; one constant offset per half
   is found from the data. Where the tactical ball is missing and the broadcast shows live play
   with the ball, the broadcast ball fills the gap (`possession/target_fused.py`).
3. **Features** (`possession/features.py`). 333 identity-free per-frame features in a frame where
   the home team attacks towards +x: ball kinematics, ball-to-player geometry, touch events per
   team, what the ball was doing before and after a gap, team shape and movement, and temporal
   context.
4. **Models** (`possession/model.py`). Five LightGBM classifiers trained only on other matches:
   alive vs dead; home vs away given alive; and three transition detectors (team change,
   alive to dead, dead to alive). The alive/dead models see only player movement, whether the
   ball is tracked, and where it is, never how it moves (see the technical document for why).
5. **Auxiliary evidence** (`possession/aux_streams.py`, `possession/aux_fit.py`). Broadcast state
   (replay or close-up, live without ball, live with ball), set-piece and goal windows and
   attacking-phase tags are added as down-weighted evidence. Their probability tables are fitted
   without labels, by EM against the classifier's own confident frames.
6. **Decoding** (`possession/decode.py`). A Viterbi pass over H / A / D per half, where the cost of
   switching state at each frame comes from the transition detectors. Frames before each half's
   tagged kick-off are dead ball.

## Retrain from scratch (optional)

The committed models are the reference. To rebuild them:

```bash
bash scripts/fetch_external.sh            # 7 DFL matches (CC BY 4.0) and the Metrica sample data
uv run python scripts/build_external.py --dfl-dir external/dfl
uv run python scripts/build_metrica_sample.py --root external/metrica-sample/data
uv run python scripts/target_stats.py     # label-free statistics of the competition tracking
uv run python scripts/cv_external.py oof  # grouped 5-fold on the 10 external matches
uv run python scripts/cv_external.py tune # decoder parameters from out-of-fold predictions
uv run python scripts/train_final.py
uv run python scripts/run.py --refit-aux
```

Training is seeded and uses LightGBM's deterministic mode. Cross-validation takes about 15
minutes and final training about 10 minutes on a 20-thread laptop CPU; no GPU is used.

## Repository layout

```
scripts/run.py              regenerate the submission
scripts/cv_external.py      cross-validation and decoder tuning on external matches
scripts/train_final.py      final training
scripts/fetch_external.sh   download the external training data
scripts/eda*.py, *_report.py, check_*.py, calib_*.py   aggregate-only analysis
possession/                 loaders, synchronisation, simulator, features, models, decoder
models/                     trained models and fitted tables
evaluation/, games.py       the organisers' scorer and constants, vendored verbatim
docs/                       the technical document
```

## Rules we worked to

- No hand labels and no hand corrections on the competition match; no constants tied to
  specific frames. Kick-off times come from the Smart Tagging, half boundaries from `games.py`.
- The public 5-minute ground truth was used to check alignment and what the labels mean. It was
  not used to train, tune or select anything. Every parameter is fitted on other matches or
  fixed in advance.
- No match data, and nothing that could reconstruct it, is in this repository. `data/`, `cache/`
  and `outputs/` are git-ignored. `possession/assets/target_stats.json` and
  `models/aux_tables.json` hold aggregate statistics only.
- No competition data was sent to any hosted language model. The code was written with an AI
  coding assistant that only ever saw aggregate outputs (scores, counts, summary statistics).

## Data and acknowledgements

- **Competition data:** Metrica Sports, FC Barcelona (Barça Innovation Hub) and Mobile World
  Capital Barcelona, under the competition's terms. Not redistributed here.
- **DFL / IDSSE open dataset:** Bassek, Rein, Weber and Memmert (2025), "An integrated dataset of
  spatiotemporal and event data in elite soccer", Scientific Data, CC BY 4.0, figshare record
  28196177. Seven matches, used for training and validation.
- **Metrica Sports sample data:** https://github.com/metrica-sports/sample-data. Three games, used
  for training and validation with possession labels derived from their event files
  (`possession/external/metrica_sample.py`).
- **Scorer:** vendored from https://github.com/metrica-sports/more-than-a-hack-2027 (MIT); see
  `THIRD_PARTY_NOTICES.md`.

Our own code is released under the MIT licence (`LICENSE`).
