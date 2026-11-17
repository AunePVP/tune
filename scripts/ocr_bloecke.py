#!/usr/bin/env python3
"""Find where the text lines are on each page, so that the correction tool can show which text belongs to a place on
the page (click on the page -> the text is marked, and the other way round). Runs the local program ``tesseract`` on
each page image and writes ``<Seite>.bloecke.json`` next to it: a list of lines with their position (as fractions of
the image size) and the text Tesseract read there. That text is only used to find the matching place in the draft; it
never becomes part of the target text. Pages that already have the file are skipped.

    python3 ocr_bloecke.py --ordner training

Needs ``tesseract`` with the language ``deu`` (on the Spark: installed). Everything stays on this computer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_vergleich import pages_in  # noqa: E402

SUFFIX = ".bloecke.json"


def parse_tsv(tsv: str) -> list[dict]:
    """Tesseract's TSV as lines: ``{"x","y","w","h"}`` as fractions of the page and ``"t"`` the words of the line."""
    width = height = 0
    lines: dict[tuple, dict] = {}
    for row in tsv.splitlines()[1:]:
        cells = row.split("\t")
        if len(cells) < 12:
            continue
        try:
            level, left, top, w, h = int(cells[0]), int(cells[6]), int(cells[7]), int(cells[8]), int(cells[9])
        except ValueError:
            continue
        if level == 1:
            width, height = w, h
        elif level == 4:
            lines[tuple(cells[2:5])] = {"box": (left, top, w, h), "words": []}
        elif level == 5 and cells[11].strip() and tuple(cells[2:5]) in lines:
            lines[tuple(cells[2:5])]["words"].append(cells[11].strip())
    if not width or not height:
        return []
    out = []
    for line in lines.values():
        text = " ".join(line["words"])
        if sum(c.isalnum() for c in text) < 2:  # specks and rules are no text
            continue
        left, top, w, h = line["box"]
        out.append({"x": round(left / width, 4), "y": round(top / height, 4), "w": round(w / width, 4),
                    "h": round(h / height, 4), "t": text})
    return out


def read_lines(page: Path, languages: str, timeout: float) -> list[dict] | None:
    """The lines of one page image, or None if Tesseract failed or took too long."""
    try:
        res = subprocess.run(["tesseract", str(page), "stdout", "-l", languages, "--psm", "1", "tsv"],
                             capture_output=True, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_tsv(res.stdout) if res.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lage der Textzeilen je Seite ermitteln (für das Korrekturwerkzeug)")
    ap.add_argument("--ordner", required=True, help="Ordner mit den Seitenbildern")
    ap.add_argument("--sprachen", default="deu+eng")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=180, help="Sekunden je Seite (große Pläne dauern)")
    args = ap.parse_args(argv)
    if not shutil.which("tesseract"):
        raise SystemExit("Das Programm tesseract fehlt auf diesem Rechner.")
    todo = [p for p in pages_in(Path(args.ordner)) if not p.with_name(p.stem + SUFFIX).exists()]
    stats = {"seiten": 0, "ohne_text": 0, "fehler": 0}

    def one(page: Path) -> None:
        lines = read_lines(page, args.sprachen, args.timeout)
        if lines is None:
            stats["fehler"] += 1
            return
        stats["seiten"] += 1
        stats["ohne_text"] += not lines
        page.with_name(page.stem + SUFFIX).write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for k, _ in enumerate(pool.map(one, todo), start=1):
            if k % 50 == 0:
                print(f"  {k} von {len(todo)} …", flush=True)
    print(f"{stats['seiten']} Seiten gelesen, davon {stats['ohne_text']} ohne erkannte Zeilen; {stats['fehler']} Fehler "
          "oder Zeitüberschreitungen (beim nächsten Aufruf erneut).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
