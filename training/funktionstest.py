#!/usr/bin/env python3
"""Functional test of the whole chain with a few INVENTED pages (drawn here, no real data): train briefly, continue
on the adapter, merge, optionally quantize, serve with vLLM, ask one page. It shows that the tools work together –
the model learns nothing useful from a dozen pages and a few steps.

    ~/tune-training/venv/bin/python training/funktionstest.py --modell qwen2b
    ~/tune-training/venv/bin/python training/funktionstest.py --modell glm-ocr --fassung fp8

Everything is written to ~/tune-training/funktionstest (runs in their own folder, apart from the real runs).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import tune  # noqa: E402
from ocr_vorgabe import ANWEISUNG, join  # noqa: E402

A4 = (1240, 1754)  # 150 dpi


def schrift(groesse: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=groesse)


def textseite(nr: int, rng: random.Random):
    from PIL import Image, ImageDraw
    bild = Image.new("RGB", A4, "white")
    d = ImageDraw.Draw(bild)
    zeilen = [f"Ingenieurbüro Beispiel GmbH · Musterweg {nr} · 12345 Musterstadt",
              f"Prüfbericht Nr. {1000 + nr}", "",
              f"Objekt: Lagerhalle {rng.choice(['Nord', 'Süd', 'West'])}, Testgasse {rng.randint(1, 99)}",
              f"Die Rauchableitung wurde am {rng.randint(1, 28):02d}.03.2021 geprüft.",
              f"Es wurden {rng.randint(2, 9)} Rauchabzüge ohne Mängel festgestellt.",
              "Die Feuerwehrzufahrt ist freizuhalten.", "", f"Seite {nr} von 12"]
    for i, zeile in enumerate(zeilen):
        d.text((120, 140 + i * 70), zeile, fill="black", font=schrift(36))
    return bild, join("Text", "\n".join(z for z in zeilen if z))


def planseite(nr: int, rng: random.Random):
    from PIL import Image, ImageDraw
    bild = Image.new("RGB", (A4[1], A4[0]), "white")
    d = ImageDraw.Draw(bild)
    for _ in range(14):  # walls and room labels, which must not be written down
        x, y = rng.randint(80, 1100), rng.randint(80, 800)
        d.rectangle((x, y, x + rng.randint(100, 400), y + rng.randint(80, 300)), outline="black", width=3)
        d.text((x + 10, y + 10), f"Büro {nr}.{rng.randint(1, 30):02d}", fill="black", font=schrift(18))
    feld = ["Brandschutzplan", f"Musterschule Testgasse {nr}", "Erdgeschoss", f"Plan-Nr. BS-{100 + nr}",
            "Maßstab 1:100", "Datum 01.02.2020"]
    d.rectangle((1250, 900, 1700, 1200), outline="black", width=4)
    for i, zeile in enumerate(feld):
        d.text((1265, 912 + i * 46), zeile, fill="black", font=schrift(30))
    return bild, join("Plan", "\n".join(feld))


def fotoseite(nr: int, rng: random.Random):
    from PIL import Image
    bild = Image.effect_noise((800, 600), 60).convert("RGB")  # stands in for a photo without text
    return bild.resize((1000 + nr, 750)), join("Foto", "")


def leerseite(nr: int, rng: random.Random):
    from PIL import Image
    return Image.new("RGB", A4, (250 - nr % 5, 250, 248)), join("Leer", "")


def erzeuge(ordner: Path, name: str, anzahl: int, start: int) -> Path:
    """Draws ``anzahl`` pages (two per invented document) and writes them like ocr_korrektur.py --export does."""
    ordner.mkdir(parents=True, exist_ok=True)
    arten = [textseite, planseite, fotoseite, leerseite, textseite, textseite]
    ziel = ordner / name
    with ziel.open("w", encoding="utf-8") as f:
        for i in range(start, start + anzahl):
            bild, antwort = arten[i % len(arten)](i, random.Random(i))
            datei = ordner / f"{i:03d}_Erfunden{i // 2}.pdf_S{i % 2 + 1}.png"
            bild.save(datei)
            f.write(json.dumps({"messages": [{"role": "user", "content": "<image>" + ANWEISUNG},
                                             {"role": "assistant", "content": antwort}],
                                "images": [str(datei)]}, ensure_ascii=False) + "\n")
    return ziel


def tune_py(*args: str) -> None:
    befehl = [sys.executable, str(HERE / "tune.py"), *args]
    print("\n>>> tune.py " + " ".join(args), flush=True)
    if subprocess.run(befehl).returncode:
        raise SystemExit(f"Funktionstest gescheitert bei: tune.py {' '.join(args)}")


def warte_auf_server(url: str, container: str, minuten: float) -> None:
    ende = time.time() + minuten * 60
    while time.time() < ende:
        try:
            with urllib.request.urlopen(url.rsplit("/v1", 1)[0] + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        lebt = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container], capture_output=True, text=True)
        if lebt.stdout.strip() != "true":
            subprocess.run(["docker", "logs", "--tail", "40", container])
            raise SystemExit(f"Der Container {container} läuft nicht mehr (siehe Protokoll oben).")
        time.sleep(5)
    raise SystemExit(f"vLLM hat nach {minuten:g} Minuten nicht geantwortet (docker logs {container}).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Funktionstest mit erfundenen Seiten")
    ap.add_argument("--modell", required=True, choices=list(tune.MODELLE))
    ap.add_argument("--fassung", default="bf16", choices=tune.FASSUNGEN, help="welche Fassung in vLLM gestartet wird")
    ap.add_argument("--schritte", type=int, default=3, help="Lernschritte je Lauf (3)")
    ap.add_argument("--speicher-gb", type=float, default=8.0)
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--image", default=tune.VLLM_IMAGE)
    ap.add_argument("--ohne-vllm", action="store_true", help="nach dem Zusammenführen aufhören")
    args = ap.parse_args()

    wurzel = tune.basis() / "funktionstest"
    os.environ["TUNE_LAEUFE"] = str(wurzel / "laeufe")
    daten1 = erzeuge(wurzel / "seiten-1", "training.jsonl", 12, 0)
    daten2 = erzeuge(wurzel / "seiten-2", "training-2.jsonl", 6, 100)
    print(f"Erfundene Seiten: {daten1} (12), {daten2} (6)")

    gemeinsam = ["--modell", args.modell, "--schritte", str(args.schritte), "--sammeln", "2", "--val-anteil", "0.3",
                 "--name", "funktionstest", "--trotzdem"]
    tune_py("trainieren", *gemeinsam, "--daten", str(daten1))
    erster = int(tune.alle_laeufe()[-1].name[:3])
    tune_py("trainieren", *gemeinsam, "--daten", str(daten2), "--weiter-von", str(erster))
    lauf = tune.alle_laeufe()[-1]
    nr = str(int(lauf.name[:3]))
    tune_py("zusammenfuehren", "--lauf", nr)
    if args.fassung != "bf16":
        tune_py("quantisieren", "--lauf", nr, "--fassung", args.fassung, "--seiten", "4")
    tune_py("laeufe")
    if args.ohne_vllm:
        return 0

    container = f"tune-{tune.dienstname(lauf, args.fassung)}"
    url = f"http://127.0.0.1:{args.port}/v1"
    tune_py("starten", "--lauf", nr, "--fassung", args.fassung, "--speicher-gb", str(args.speicher_gb),
            "--port", str(args.port), "--parallel", "2", "--image", args.image, "--ausfuehren", "--trotzdem")
    try:
        warte_auf_server(url, container, 15)
        seite = next((wurzel / "seiten-1").glob("000_*.png"))
        tune_py("fragen", "--lauf", nr, "--fassung", args.fassung, "--bild", str(seite), "--url", url)
    finally:
        tune_py("stoppen", "--lauf", nr, "--fassung", args.fassung)
    print(f"\nFunktionstest für {args.modell} ({args.fassung}) durchgelaufen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
