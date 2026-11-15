"""Train the v1 hierarchical models (AD, TEAM, TC, A2D, D2A) on all 10 external matches and save them under models/.

    uv run python scripts/train_final.py

Tree counts come from the cross-validation early stopping (cache/oof_v1/trees.json); decoder parameters are in
models/decoder.json (written by `cv_external.py tune`).
"""
from __future__ import annotations

import json
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
    trees = json.loads((ROOT / "cache" / "oof_v1" / "trees.json").read_text())["trees"]
    ids = external_ids()
    matches = [canon.canonicalise(m) for m in load_external()]
    t = time.time()
    sets, names = model.make_training_sets(matches, seed=SEED, k_copies=K_COPIES)
    print(f"{len(ids)} matches, {len(sets)} training sets, {sum(len(s.X) for s in sets)} rows, built in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    mdl = model.fit(sets, names, trees)
    mdl.save(ROOT / "models")
    for old in ("state.txt", "trans.txt"):             # v0 models (preserved in git)
        (ROOT / "models" / old).unlink(missing_ok=True)
    print(f"trained in {time.time() - t:.0f}s; trees {trees}; features all/AD {len(names)}/{len(mdl.cols['ad'])}; saved to models/")
    print("decoder:", (ROOT / "models" / "decoder.json").read_text().replace("\n", " "))


if __name__ == "__main__":
    main()
