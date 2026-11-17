#!/usr/bin/env python3
"""Drafts for the training pages: every page image of a folder (python -m projektakte.ocr_testseiten --training) goes to
a model with the target prompt (ocr_vorgabe.PROMPT); the answer is written next to it as ``<Seite>.entwurf.txt`` – the
starting point for the correction (ocr_korrektur.py). Pages with a draft already are skipped (the run can be resumed).
Standard library only; ocr_vergleich.py and ocr_vorgabe.py must lie next to it.

    python3 ocr_entwurf.py --ordner training --modell "Qwen|http://localhost:8003/v1|qwen35-4b" \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' --parallel 8
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_vergleich import MIME, ask, looped, pages_in, parse_model  # noqa: E402
from ocr_vorgabe import ANWEISUNG, PROMPT, join, split  # noqa: E402


def draft(folder: Path, model: dict, *, parallel: int, max_tokens: int, timeout: float, extra: dict | None,
          prompt: str = PROMPT) -> dict:
    todo = [p for p in pages_in(folder) if not p.with_name(p.stem + ".entwurf.txt").exists()]
    stats = {"seiten": len(todo), "fehler": 0, "ohne_typ": 0, "schleife": 0, "meldungen": {}}

    def one(page: Path) -> None:
        res = ask(model["url"], model["model"], prompt, page.read_bytes(), MIME[page.suffix.lower()], max_tokens,
                  timeout, extra)
        if res["error"]:
            stats["fehler"] += 1
            first = stats["meldungen"].setdefault(res["error"], 0) == 0
            stats["meldungen"][res["error"]] += 1
            if first:  # say why at once, not only in the count at the end
                print(f"  Fehler bei {page.name}: {res['error']}", flush=True)
            return
        page_type, text = split(res["text"])
        if page_type is None:
            stats["ohne_typ"] += 1
            page_type = "Text"
        if looped(text) or res.get("finish") == "length":
            stats["schleife"] += 1
            text = ""  # a loop is no help when correcting: start from the empty page
        page.with_name(page.stem + ".entwurf.txt").write_text(join(page_type, text), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for k, _ in enumerate(pool.map(one, todo), start=1):
            if k % 25 == 0:
                print(f"  {k} von {len(todo)} …", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entwürfe für die Trainingsseiten erzeugen")
    ap.add_argument("--ordner", required=True, help="Ordner mit den Seitenbildern")
    ap.add_argument("--modell", required=True, help="Name|URL|Modell-ID (die Anweisung kommt aus ocr_vorgabe.py)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--extra-body", default="", help="weitere Felder als JSON, z. B. Denken ausschalten")
    ap.add_argument("--trainiert", action="store_true",
                    help="das Modell ist ein feingetunter Seitenleser: so fragen wie im Training (nur das Bild)")
    args = ap.parse_args(argv)
    extra = json.loads(args.extra_body) if args.extra_body.strip() else None
    stats = draft(Path(args.ordner), parse_model(args.modell), parallel=args.parallel, max_tokens=args.max_tokens,
                  timeout=args.timeout, extra=extra, prompt=ANWEISUNG if args.trainiert else PROMPT)
    print(f"{stats['seiten']} Entwürfe – {stats['fehler']} Fehler (beim nächsten Aufruf erneut), "
          f"{stats['ohne_typ']} ohne Typzeile (als Text angenommen), {stats['schleife']} Schleifen/abgeschnitten (leer "
          f"gelassen)")
    for message, count in sorted(stats["meldungen"].items(), key=lambda m: -m[1]):
        print(f"  {count}× {message}")
    return 1 if stats["fehler"] else 0


if __name__ == "__main__":
    sys.exit(main())
