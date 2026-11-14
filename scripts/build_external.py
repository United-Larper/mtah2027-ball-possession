"""Convert every complete DFL match to cache/external/dfl_<id>.npz (idempotent, --force to redo).

    uv run scripts/build_external.py [--dfl-dir DIR] [--force] [--ids J03WMX ...]
Also writes cache/external/dfl_<id>.report.json with the alignment / direction / crosstab diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from possession.external.dfl import load_dfl_match_with_report  # noqa: E402

DEFAULT_DIR = REPO.parent / "external" / "dfl"
IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ids", nargs="*", default=IDS)
    args = ap.parse_args()
    src = Path(args.dfl_dir)
    out = REPO / "cache" / "external"
    out.mkdir(parents=True, exist_ok=True)
    for mid in args.ids:
        pos = src / f"DFL-MAT-{mid}_positions.xml"
        info = src / f"DFL-MAT-{mid}_matchinfo.xml"
        dst = out / f"dfl_{mid}.npz"
        if dst.exists() and not args.force:
            print(f"{mid}: exists, skipped")
            continue
        if not pos.exists() or not info.exists():
            print(f"{mid}: MISSING (positions complete: {pos.exists()}, matchinfo: {info.exists()})")
            continue
        t0 = time.time()
        m, rep = load_dfl_match_with_report(pos, info)
        m.save(dst)
        rep["parse_seconds"] = round(time.time() - t0, 1)
        (out / f"dfl_{mid}.report.json").write_text(json.dumps(rep, indent=1, default=str))
        per = {p: v["T"] for p, v in rep["periods"].items()}
        print(f"{mid}: T={m.T} frames/period={per} cols H/A={m.hx.shape[1]}/{m.ax.shape[1]} "
              f"home_dir={m.home_dir} gaps={[v['gaps_in_union'] for v in rep['periods'].values()]} "
              f"{rep['parse_seconds']}s")


if __name__ == "__main__":
    main()
