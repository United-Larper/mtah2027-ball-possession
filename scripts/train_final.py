"""Train the final state and transition models on all external matches and save them under models/.

    uv run python scripts/train_final.py

Uses the settings chosen in cross-validation (tree counts from cache/oof/meta.json; decoder
parameters from models/decoder.json, written by `cv_external.py tune`).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import canon, model  # noqa: E402
from possession.external_list import external_ids, load_external  # noqa: E402

SEED = 2027
K_COPIES = 3


def main():
    meta = json.loads((ROOT / "cache" / "oof" / "meta.json").read_text())
    ids = external_ids()
    matches = [canon.canonicalise(m) for m in load_external()]
    t = time.time()
    sets, names = model.make_training_sets(matches, seed=SEED, k_copies=K_COPIES)
    print(f"{len(ids)} matches, {len(sets)} training sets, {sum(len(s.X) for s in sets)} rows, built in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    mdl = model.fit(sets, names, meta["n_state"], meta["n_trans"])
    mdl.save(ROOT / "models")
    print(f"trained in {time.time() - t:.0f}s; saved to models/ ({len(names)} features)")
    dec = ROOT / "models" / "decoder.json"
    if not dec.exists():
        shutil.copy(ROOT / "cache" / "oof" / "decoder_learned.json", dec)
    print("decoder:", dec.read_text().replace("\n", " "))


if __name__ == "__main__":
    main()
