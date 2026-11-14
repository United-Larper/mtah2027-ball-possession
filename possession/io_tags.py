"""Smart Tagging (SportsCode XML) loader."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd

from .paths import asset_paths

FPS = 25


def load_tags(fps: int = FPS) -> pd.DataFrame:
    """One row per instance: code, start_frame, end_frame (round(s*fps)), start_s, end_s, tags (dict)."""
    rows = []
    for inst in ET.parse(asset_paths("tactical")["smart_tagging"]).iter("instance"):
        s, e = float(inst.findtext("start")), float(inst.findtext("end"))
        tags = {}
        for lab in inst.findall("label"):
            tags[lab.findtext("group")] = lab.findtext("text")
        rows.append(dict(code=inst.findtext("code"), start_s=s, end_s=e,
                         start_frame=int(round(s * fps)), end_frame=int(round(e * fps)), tags=tags))
    return pd.DataFrame(rows)
