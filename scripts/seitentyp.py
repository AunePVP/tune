#!/usr/bin/env python3
"""Page types for a small classifier: an exported page folder (python -m projektakte.ocr_testseiten) goes to a
vision model page by page; it answers with one type only. The result is ``seitentypen.csv`` in the folder
(datei;typ;antwort) – the labels for training a one-layer classifier. Resumable: pages with a type are skipped,
pages without (error, answer not understood) are asked again. Standard library only; ocr_vergleich.py must lie
next to it.

    python3 seitentyp.py --ordner seiten --modell "Qwen|http://localhost:8000/v1|qwen-3.8-flash" \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' --parallel 8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_vergleich import MIME, ask, pages_in, parse_model  # noqa: E402

TYPES = ("Text", "Plan", "Bild", "Leer", "Unsicher")
RESULT = "seitentypen.csv"
PROMPT = """Ordne dieses Seitenbild genau einem Typ zu. Antworte nur mit einer Zeile: TYP: <Typ>

Typen:
- Text: alle normalen Dokumente – Briefe, Berichte, Gutachten, Formulare, Tabellen, Datenblätter, Mails,
  Zulassungen, Berechnungen, auch mit Grafiken oder Diagrammen. Auch eine mit Handy oder Kamera
  fotografierte Textseite ist Text.
- Plan: Pläne – Grundrisse, Schnitte, Ansichten, Lagepläne, Brandschutz- und Fluchtpläne, CAD-Pläne,
  Schemata. Auch ein fotografierter Plan ist Plan.
- Bild: ein mit Handy oder Kamera aufgenommenes Foto (Baustelle, Raum, Bauteil, Gerät, Typenschild),
  das kein Dokument und kein Plan ist.
- Leer: ein Logo, eine leere Seite oder eine Seite ohne Text (Grafik, Hintergrund, Trennblatt).
- Unsicher: wenn keiner der Typen klar passt.

Antworte nur mit: TYP: Text, TYP: Plan, TYP: Bild, TYP: Leer oder TYP: Unsicher"""

_ANSWER = re.compile(r"\b(text|plan|bild|leer|unsicher)\b", re.I)


def parse(answer: str) -> str | None:
    """The type in an answer like "TYP: Plan" (also without "TYP:"); None if not understood."""
    first = (answer or "").strip().splitlines()[0] if (answer or "").strip() else ""
    m = _ANSWER.search(first.split(":", 1)[1] if ":" in first else first)
    return m.group(1).capitalize() if m else None


def _read(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {r["datei"]: r for r in csv.DictReader(f, delimiter=";")}


def _write(path: Path, rows: dict[str, dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["datei", "typ", "antwort"], delimiter=";")
        w.writeheader()
        for name in sorted(rows):
            w.writerow(rows[name])
    tmp.replace(path)


def classify(folder: Path, model: dict, *, parallel: int, max_tokens: int, timeout: float,
             extra: dict | None) -> dict:
    out = folder / RESULT
    rows = _read(out)
    todo = [p for p in pages_in(folder) if not (rows.get(p.name) or {}).get("typ")]
    stats = {"seiten": len(todo), "fehler": 0, "unverstanden": 0, "meldungen": Counter()}
    lock = threading.Lock()

    def one(page: Path) -> None:
        res = ask(model["url"], model["model"], model["prompt"], page.read_bytes(), MIME[page.suffix.lower()],
                  max_tokens, timeout, extra)
        with lock:
            if res["error"]:
                stats["fehler"] += 1
                if not stats["meldungen"][res["error"]]:
                    print(f"  Fehler bei {page.name}: {res['error']}", flush=True)
                stats["meldungen"][res["error"]] += 1
                return
            typ = parse(res["text"])
            if typ is None:
                stats["unverstanden"] += 1
            rows[page.name] = {"datei": page.name, "typ": typ or "", "antwort": res["text"][:200].replace("\n", " ")}

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for k, _ in enumerate(pool.map(one, todo), start=1):
            if k % 50 == 0:
                with lock:
                    _write(out, rows)  # progress survives an interruption
                print(f"  {k} von {len(todo)} …", flush=True)
    _write(out, rows)
    stats["verteilung"] = Counter(r["typ"] or "(unverstanden)" for r in rows.values())
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seitentypen (Text/Plan/Bild/Leer/Unsicher) von einem Bildmodell "
                                             "zuordnen lassen – Daten für einen Klassifizierer")
    ap.add_argument("--ordner", required=True, help="Ordner mit den exportierten Seitenbildern")
    ap.add_argument("--modell", required=True, help="Name|URL|Modell-ID (das Modell muss Bilder verstehen)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--extra-body", default="", help="weitere Felder als JSON, z. B. Denken ausschalten")
    args = ap.parse_args(argv)
    model = parse_model(args.modell)
    if len(args.modell.split("|")) < 4:  # no own prompt given: the type question, not the transcription
        model["prompt"] = PROMPT
    extra = json.loads(args.extra_body) if args.extra_body.strip() else None
    stats = classify(Path(args.ordner), model, parallel=args.parallel, max_tokens=args.max_tokens,
                     timeout=args.timeout, extra=extra)
    print(f"{stats['seiten']} Seite(n) gefragt – {stats['fehler']} Fehler, {stats['unverstanden']} Antwort(en) "
          f"nicht verstanden (beides beim nächsten Aufruf erneut).")
    for message, count in stats["meldungen"].most_common():
        print(f"  {count}× {message}")
    print("Verteilung: " + ", ".join(f"{t} {n}" for t, n in sorted(stats["verteilung"].items())))
    print(f"Ergebnis: {Path(args.ordner) / RESULT}")
    return 1 if stats["fehler"] else 0


if __name__ == "__main__":
    sys.exit(main())
