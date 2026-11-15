#!/usr/bin/env bash
# Download the external training data into ./external (git-ignored).
#
#   bash scripts/fetch_external.sh
#
# 1. DFL / IDSSE open dataset: Bassek, Rein, Weber, Memmert (2025), Scientific Data,
#    CC BY 4.0, figshare record 28196177. Seven matches, about 2.6 GB.
# 2. Metrica Sports sample data: https://github.com/metrica-sports/sample-data (three games).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DFL="$ROOT/external/dfl"
mkdir -p "$DFL"
cd "$DFL" || exit 1

fetch() { # figshare-file-id output-name
  if [ -s "$2" ]; then echo "have $2"; return; fi
  curl -sS -L --retry 5 --retry-delay 5 -o "$2.part" "https://ndownloader.figshare.com/files/$1" \
    && mv "$2.part" "$2" && echo "ok   $2" || echo "FAIL $2"
}

# match information (small)
fetch 51643475 DFL-MAT-J03WMX_matchinfo.xml
fetch 51643472 DFL-MAT-J03WN1_matchinfo.xml
fetch 51643478 DFL-MAT-J03WOH_matchinfo.xml
fetch 51643481 DFL-MAT-J03WOY_matchinfo.xml
fetch 51643487 DFL-MAT-J03WPY_matchinfo.xml
fetch 51643484 DFL-MAT-J03WQQ_matchinfo.xml
fetch 51643490 DFL-MAT-J03WR9_matchinfo.xml

# position data (350-420 MB each)
fetch 51643514 DFL-MAT-J03WMX_positions.xml
fetch 51643517 DFL-MAT-J03WN1_positions.xml
fetch 51643520 DFL-MAT-J03WOH_positions.xml
fetch 51643523 DFL-MAT-J03WOY_positions.xml
fetch 51643526 DFL-MAT-J03WPY_positions.xml
fetch 51643529 DFL-MAT-J03WQQ_positions.xml
fetch 51643532 DFL-MAT-J03WR9_positions.xml

cd "$ROOT/external" || exit 1
if [ -d metrica-sample/.git ]; then
  echo "have metrica-sample"
else
  git clone --depth 1 https://github.com/metrica-sports/sample-data.git metrica-sample
fi
echo "done"
