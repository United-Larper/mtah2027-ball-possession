"""Render docs/RickRolled.md to docs/RickRolled.pdf with a headless Chromium-based browser.

    uv run --with markdown python scripts/build_pdf.py

Needs Microsoft Edge or Google Chrome installed. Not part of the submission pipeline.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "RickRolled.md"
HTML = ROOT / "docs" / "RickRolled.html"
PDF = ROOT / "docs" / "RickRolled.pdf"

CSS = """
@page { size: A4; margin: 17mm 16mm 17mm 16mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 4pt 0; }
h2 { font-size: 12.5pt; margin: 15pt 0 5pt 0; border-bottom: 0.6pt solid #bbb; padding-bottom: 2pt; break-after: avoid; }
p { margin: 0 0 7pt 0; }
ul { margin: 0 0 7pt 0; padding-left: 16pt; }
li { margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt 0; font-size: 9.3pt; break-inside: avoid; }
th, td { border: 0.5pt solid #999; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2pt; }
pre { background: #f2f2f2; padding: 6pt 8pt; font-size: 9pt; break-inside: avoid; }
pre code { background: none; padding: 0; }
a { color: #1a1a1a; text-decoration: none; }
"""

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    for name in ("msedge", "chrome", "chromium", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("no Edge or Chrome found")


def main() -> None:
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>RickRolled</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    HTML.unlink()
    print(f"wrote {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
