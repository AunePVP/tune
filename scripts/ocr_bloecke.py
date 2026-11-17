#!/usr/bin/env python3
"""Find where the text lines are on each page, so that the correction tool can show which text belongs to a place on
the page (click on the page -> the text is marked, and the other way round). Runs the local program ``tesseract`` on
each page image and writes ``<Seite>.bloecke.json`` next to it: a list of lines with their position (as fractions of
the image size) and the text Tesseract read there. That text is only used to find the matching place in the draft; it
never becomes part of the target text. Pages that already have the file are skipped.

    python3 ocr_bloecke.py --ordner training

Needs ``tesseract`` with the language ``deu`` (on the Spark: installed). Everything stays on this computer.

Tesseract often fails on pages scanned sideways. With Pillow available (on the Spark: run this script with
``~/tune-training/venv/bin/python``) a page that reads badly is tried in the other three directions as well, the best
one wins, and the file records the rotation so that the correction tool can show the page upright:
``{"drehung": 90, "zeilen": [...]}`` – positions always refer to the image as it is stored.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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


def score(tsv: str) -> tuple[int, float]:
    """How well a page was read: (letters and digits in words Tesseract is fairly sure of, mean confidence)."""
    sure, confidences = 0, []
    for row in tsv.splitlines()[1:]:
        cells = row.split("\t")
        if len(cells) >= 12 and cells[0] == "5" and cells[11].strip():
            try:
                conf = float(cells[10])
            except ValueError:
                continue
            confidences.append(conf)
            if conf >= 60:
                sure += sum(c.isalnum() for c in cells[11])
    return sure, (sum(confidences) / len(confidences) if confidences else 0.0)


def unrotate(line: dict, degrees: int) -> dict:
    """A line found on the page turned clockwise by ``degrees``, as a position on the page as it is stored."""
    x, y, w, h = line["x"], line["y"], line["w"], line["h"]
    if degrees == 90:
        x, y, w, h = y, 1 - (x + w), h, w
    elif degrees == 180:
        x, y = 1 - (x + w), 1 - (y + h)
    elif degrees == 270:
        x, y, w, h = 1 - (y + h), x, h, w
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4), "t": line["t"]}


def _tesseract(image: Path, languages: str, timeout: float) -> str | None:
    try:
        res = subprocess.run(["tesseract", str(image), "stdout", "-l", languages, "--psm", "1", "tsv"],
                             capture_output=True, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout if res.returncode == 0 else None


def orientation(image: Path, timeout: float) -> int:
    """Degrees the page must be turned clockwise to stand upright, by Tesseract's orientation detection; 0 if unsure."""
    try:
        res = subprocess.run(["tesseract", str(image), "stdout", "--psm", "0"], capture_output=True, text=True,
                             errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    found = dict(line.split(":", 1) for line in res.stdout.splitlines() if ":" in line)
    try:
        degrees, sure = int(found["Rotate"]), float(found["Orientation confidence"])
    except (KeyError, ValueError):
        return 0
    return degrees if degrees in (90, 180, 270) and sure >= 2 else 0


def read_page(page: Path, languages: str, timeout: float) -> dict | None:
    """``{"drehung", "zeilen"}`` of one page image, or None if Tesseract failed or took too long. With Pillow the page
    is read upright: first in the direction the orientation detection names, and if that reads badly in the other
    directions as well; another direction only wins clearly."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None  # large plans are fine, the images are our own
    except ImportError:
        Image = None
    first = orientation(page, timeout)
    order = [first] + [d for d in (0, 90, 270, 180) if d != first] if Image is not None else [0]
    best = None  # (degrees, tsv, sure)
    with tempfile.TemporaryDirectory() as tmp:
        for degrees in order:
            image = page
            if degrees:
                image = Path(tmp) / f"{degrees}.png"
                with Image.open(page) as original:
                    upright = original if original.mode in ("L", "RGB") else original.convert("RGB")  # e.g. CMYK
                    upright.rotate(-degrees, expand=True).save(image)  # clockwise
            tsv = _tesseract(image, languages, timeout)
            if tsv is None:
                if best is None and degrees == order[-1]:
                    return None
                continue
            sure, mean = score(tsv)
            if best is None or sure > max(1.5 * best[2], best[2] + 40):
                best = (degrees, tsv, sure)
            if degrees == first and sure >= 300 and mean >= 80:
                break  # reads well as detected: no need to try the other directions
    if best is None:
        return None
    return {"drehung": best[0], "zeilen": [unrotate(line, best[0]) for line in parse_tsv(best[1])]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lage der Textzeilen je Seite ermitteln (für das Korrekturwerkzeug)")
    ap.add_argument("--ordner", required=True, help="Ordner mit den Seitenbildern")
    ap.add_argument("--sprachen", default="deu+eng")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=180, help="Sekunden je Seite (große Pläne dauern)")
    ap.add_argument("--neu", action="store_true", help="auch Seiten neu lesen, die schon eine Datei haben")
    args = ap.parse_args(argv)
    if not shutil.which("tesseract"):
        raise SystemExit("Das Programm tesseract fehlt auf diesem Rechner.")
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Hinweis: Pillow fehlt – gedrehte Seiten werden nicht in anderen Richtungen versucht. Auf dem Spark "
              "mit ~/tune-training/venv/bin/python starten.")
    todo = [p for p in pages_in(Path(args.ordner)) if args.neu or not p.with_name(p.stem + SUFFIX).exists()]
    stats = {"seiten": 0, "ohne_text": 0, "gedreht": 0, "fehler": 0}

    def one(page: Path) -> None:
        try:
            result = read_page(page, args.sprachen, args.timeout)
        except Exception as exc:  # one unreadable image must not stop the run
            print(f"  Fehler bei einer Seite: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            result = None
        if result is None:
            stats["fehler"] += 1
            return
        stats["seiten"] += 1
        stats["ohne_text"] += not result["zeilen"]
        stats["gedreht"] += result["drehung"] != 0
        page.with_name(page.stem + SUFFIX).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for k, _ in enumerate(pool.map(one, todo), start=1):
            if k % 50 == 0:
                print(f"  {k} von {len(todo)} …", flush=True)
    print(f"{stats['seiten']} Seiten gelesen, davon {stats['ohne_text']} ohne erkannte Zeilen, {stats['gedreht']} gedreht "
          f"gelesen; {stats['fehler']} Fehler "
          "oder Zeitüberschreitungen (beim nächsten Aufruf erneut).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
