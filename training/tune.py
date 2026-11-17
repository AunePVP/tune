#!/usr/bin/env python3
"""Train the page reader with LLaMA-Factory (LoRA), keep every run in its own numbered folder, merge, quantize and
serve it. This file needs the standard library only; the heavy work is done by ``llamafactory-cli`` in the environment
that ``einrichten.sh`` sets up outside the repository (default ``~/tune-training``).

    python3 training/tune.py pruefen
    python3 training/tune.py trainieren --modell qwen2b --daten ~/training.jsonl
    python3 training/tune.py trainieren --modell qwen2b --daten ~/training-2.jsonl --weiter-von 1
    python3 training/tune.py laeufe
    python3 training/tune.py zusammenfuehren --lauf 1
    python3 training/tune.py quantisieren --lauf 1 --fassung fp8
    python3 training/tune.py starten --lauf 1 --fassung fp8 --speicher-gb 10
    python3 training/tune.py fragen --lauf 1 --fassung fp8 --bild seite.png

See training/README.md (German) for what every step and setting does.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from ocr_vorgabe import ANWEISUNG, PROMPT, TYPES, split  # noqa: E402

# Base models. "bedarf_gb": memory a training needs at A4 size with some reserve (measured 17.11.2026 with short
# answers: qwen2b 15 GB, glm-ocr 6 GB; qwen4b is a guess). It feeds the check against the free memory before a run.
MODELLE = {
    "qwen2b": {"hf": "Qwen/Qwen3.5-2B", "template": "qwen3_5_nothink", "trust_remote_code": False, "bedarf_gb": 18},
    "glm-ocr": {"hf": "zai-org/GLM-OCR", "template": "glm_ocr", "trust_remote_code": True, "bedarf_gb": 9},
    "qwen4b": {"hf": "Qwen/Qwen3.5-4B", "template": "qwen3_5_nothink", "trust_remote_code": False, "bedarf_gb": 32},
}
FASSUNGEN = ("bf16", "fp8", "nvfp4")
A4_PIXEL = 1536 * 1536  # 2.36 MP: an A4 page at 150 dpi (1240 x 1754 = 2.18 MP) passes unscaled
VLLM_IMAGE = "vllm/vllm-openai:latest"
# The model is asked with ANWEISUNG (nothing but the image); PROMPT is the written form of the rules the target texts
# follow. Both are recorded with every run.
ANWEISUNG_SHA = hashlib.sha256(ANWEISUNG.encode("utf-8")).hexdigest()
REGELN_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()

# One sentence per setting; written as comments into each run's einstellungen.yaml and explained in the README.
ERKLAERUNG = {
    "model_name_or_path": "Grundmodell (Original in voller Genauigkeit), auf dem trainiert wird.",
    "adapter_name_or_path": "Adapter eines früheren Laufs, auf dem weitertrainiert wird.",
    "trust_remote_code": "Erlaubt Programmcode aus dem Modellordner (GLM-OCR braucht das).",
    "template": "Gesprächsvorlage: wie Anweisung, Bild und Antwort für das Modell verpackt werden (ohne Denken).",
    "image_max_pixels": "Größte Bildfläche; größere Seiten werden proportional verkleinert. A4 mit 150 dpi passt.",
    "image_min_pixels": "Kleinste Bildfläche; kleinere Bilder werden vergrößert.",
    "stage": "Art des Trainings: überwachtes Feintuning (Eingabe und gewünschte Antwort).",
    "do_train": "Es wird trainiert (nicht nur ausgewertet).",
    "finetuning_type": "LoRA: Das Modell bleibt unverändert, nur ein kleiner Zusatz (Adapter) wird gelernt.",
    "lora_rank": "Größe des Adapters: höher kann mehr lernen, braucht mehr Daten und lernt eher auswendig.",
    "lora_alpha": "Stärke, mit der der Adapter wirkt; üblich ist das Doppelte des Rangs.",
    "lora_dropout": "Schaltet beim Training zufällig Teile des Adapters ab; bremst Auswendiglernen.",
    "lora_target": "Welche Schichten einen Adapter bekommen: alle linearen Schichten des Sprachteils.",
    "freeze_vision_tower": "Der Bildteil (das „Auge“) bleibt eingefroren; er kann schon lesen.",
    "freeze_multi_modal_projector": "Die Brücke zwischen Bild- und Sprachteil bleibt eingefroren.",
    "dataset_dir": "Ordner mit den Daten dieses Laufs (Kopie der JSONL-Zeilen, damit der Stand festgehalten ist).",
    "dataset": "Trainingsseiten.",
    "eval_dataset": "Validierungsseiten: ganze Dokumente aus den Trainingsdaten, die nicht gelernt werden.",
    "cutoff_len": "Längste Folge aus Bild, Anweisung und Antwort in Tokens; was länger ist, wird abgeschnitten.",
    "preprocessing_num_workers": "Prozesse für das Aufbereiten der Daten.",
    "output_dir": "Hierhin kommen Adapter, Checkpoints und Verlustkurve.",
    "logging_steps": "Alle wie viele Schritte der Verlust protokolliert wird.",
    "save_strategy": "Wann ein Checkpoint gespeichert wird: nach jeder Epoche.",
    "plot_loss": "Zeichnet die Verlustkurve als Bild.",
    "save_only_model": "Checkpoints enthalten auch den Stand des Optimierers (zum Fortsetzen nach Abbruch).",
    "report_to": "Keine Meldung an Dienste im Netz.",
    "per_device_train_batch_size": "Seiten, die gleichzeitig durch das Modell gehen (mehr braucht mehr Speicher).",
    "gradient_accumulation_steps": "So viele Stapel werden gesammelt, bevor das Modell einen Lernschritt macht.",
    "learning_rate": "Lernrate: wie stark ein Lernschritt den Adapter verändert.",
    "num_train_epochs": "Epochen: wie oft jede Seite gesehen wird.",
    "max_steps": "Obergrenze der Lernschritte (nur für Funktionstests).",
    "lr_scheduler_type": "Die Lernrate fällt im Lauf des Trainings sanft auf null (Kosinus).",
    "warmup_steps": "Anteil der Schritte am Anfang, in dem die Lernrate von null hochfährt.",
    "bf16": "Rechnet in BF16 (volle Genauigkeit des Originals).",
    "seed": "Startwert des Zufalls: gleiche Einstellungen ergeben den gleichen Lauf.",
    "eval_strategy": "Wann die Validierung gemessen wird: nach jeder Epoche.",
    "per_device_eval_batch_size": "Stapelgröße bei der Validierung.",
    "export_dir": "Hierhin kommt das zusammengeführte Modell.",
    "export_size": "Größte Dateigröße der Gewichte in GB.",
    "export_device": "Zusammengeführt wird auf dem Prozessor (braucht keinen Grafikspeicher).",
    "export_legacy_format": "Gewichte im Format safetensors.",
}


class Fehler(SystemExit):
    """A message for the user, no traceback."""

    def __init__(self, text: str):
        super().__init__("FEHLER: " + text)


def basis() -> Path:
    return Path(os.environ.get("TUNE_TRAINING") or Path.home() / "tune-training")


# ---------------------------------------------------------------------------------------------------------------------
# Data: read the export, refuse the test set, split off validation documents
# ---------------------------------------------------------------------------------------------------------------------

def bild_name(pfad: str) -> str:
    """File name of an image path written on Windows or Linux."""
    return re.split(r"[/\\]", pfad)[-1]


def dokument(pfad: str) -> str:
    """The document a page image belongs to: the file name without the export number and the page suffix
    (the same rule as ``document_of`` in ocr_korrektur.py without seiten.json)."""
    stem = bild_name(pfad).rsplit(".", 1)[0]
    return re.sub(r"_S\d+$", "", re.sub(r"^\d+_", "", stem))


def ist_validierung(dok: str, anteil: float) -> bool:
    """A fixed hash per document with its own salt, so it is independent of the test-set split of the export and the
    same for every run; a larger share only adds documents."""
    return int(hashlib.sha256(("validierung:" + dok).encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32 < anteil


def lies_jsonl(datei: Path) -> list[dict]:
    """The rows of an export file, checked: one image, the current PROMPT, an answer in the target form."""
    if datei.name.lower().endswith(".pruef.jsonl"):
        raise Fehler(f"{datei.name} ist ein Prüfsatz. Prüfseiten werden nie trainiert.")
    if not datei.is_file():
        raise Fehler(f"Datei nicht gefunden: {datei}")
    zeilen = []
    with datei.open(encoding="utf-8") as f:
        for nr, zeile in enumerate(f, start=1):
            if not zeile.strip():
                continue
            wo = f"{datei.name}, Zeile {nr}"
            try:
                row = json.loads(zeile)
                frage, antwort = row["messages"]
                bilder = row["images"]
            except (ValueError, KeyError, TypeError):
                raise Fehler(f"{wo}: nicht im Format des Exports (messages + images).") from None
            if frage.get("role") != "user" or antwort.get("role") != "assistant" or len(bilder) != 1:
                raise Fehler(f"{wo}: erwartet werden eine Frage, eine Antwort und genau ein Bild.")
            if frage.get("content") != "<image>" + ANWEISUNG:
                raise Fehler(f"{wo}: Die Anfrage ist nicht die aktuelle Fassung von ANWEISUNG (scripts/ocr_vorgabe.py: "
                             "nur das Bild). Die Datei stammt aus einem älteren Export – neu exportieren.")
            if split(antwort.get("content") or "")[0] is None:
                raise Fehler(f"{wo}: Die Antwort beginnt nicht mit einer Typzeile (TYP: …).")
            zeilen.append({"messages": [frage, antwort], "images": [str(bilder[0])]})
    if not zeilen:
        raise Fehler(f"{datei} enthält keine Seiten.")
    return zeilen


def pruefsatz_neben(datei: Path) -> Path:
    return datei.with_name(datei.name[: -len(".jsonl")] + ".pruef.jsonl") if datei.name.endswith(".jsonl") else datei


def pruef_bilder(dateien: list[Path]) -> tuple[set[str], set[str]]:
    """Image names and documents of test-set files (read leniently, only the image paths)."""
    namen, doks = set(), set()
    for datei in dateien:
        with datei.open(encoding="utf-8") as f:
            for zeile in f:
                if zeile.strip():
                    for bild in json.loads(zeile).get("images", []):
                        namen.add(bild_name(str(bild)))
                        doks.add(dokument(str(bild)))
    return namen, doks


def lade_daten(dateien: list[Path], pruefsaetze: list[Path]) -> tuple[list[dict], dict]:
    """All training rows of the given files (a page given twice counts once, the later file wins). Refuses if a page
    or a document is also in a test set: the ``*.pruef.jsonl`` next to each file and those named explicitly."""
    seiten: dict[str, dict] = {}
    doppelt = 0
    for datei in dateien:
        for row in lies_jsonl(datei):
            doppelt += row["images"][0] in seiten
            seiten[row["images"][0]] = row
    neben = [p for p in map(pruefsatz_neben, dateien) if p.is_file()]
    bekannt = list(dict.fromkeys([*neben, *pruefsaetze]))
    namen, doks = pruef_bilder(bekannt)
    treffer = [b for b in seiten if bild_name(b) in namen or dokument(b) in doks]
    if treffer:
        raise Fehler(f"{len(treffer)} Seiten der Trainingsdaten gehören zu Seiten oder Dokumenten eines Prüfsatzes "
                     f"({', '.join(p.name for p in bekannt)}). Prüfseiten werden nie trainiert. Wurde der Prüf-Anteil "
                     "beim Export verkleinert oder eine Datei verwechselt?")
    return list(seiten.values()), {"doppelt": doppelt, "pruefsaetze": [str(p) for p in bekannt]}


def teile(zeilen: list[dict], anteil: float) -> tuple[list[dict], list[dict]]:
    """(training, validation): whole documents go to validation."""
    val = [r for r in zeilen if ist_validierung(dokument(r["images"][0]), anteil)]
    train = [r for r in zeilen if not ist_validierung(dokument(r["images"][0]), anteil)]
    return train, val


def mische_alt(neu: list[dict], alt: list[dict], anteil: float) -> list[dict]:
    """Old pages to mix into a continued training so that they make up ``anteil`` of the pages; always the same
    selection for the same data (ordered by a hash of the image path)."""
    if not 0 <= anteil < 1:
        raise Fehler("--alt-anteil muss zwischen 0 und 1 liegen (1 ausgeschlossen).")
    schon = {r["images"][0] for r in neu}
    rest = sorted((r for r in alt if r["images"][0] not in schon),
                  key=lambda r: hashlib.sha256(r["images"][0].encode("utf-8")).hexdigest())
    return rest[: round(len(neu) * anteil / (1 - anteil))]


def zaehle(zeilen: list[dict]) -> dict[str, int]:
    counts = {t: 0 for t in TYPES}
    for r in zeilen:
        counts[split(r["messages"][1]["content"])[0]] += 1
    return counts


def bildgroesse(datei: Path) -> tuple[int, int] | None:
    """(width, height) of a PNG or JPEG from its header, without Pillow; None if unknown."""
    try:
        with datei.open("rb") as f:
            kopf = f.read(32)
            if kopf[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", kopf[16:24])
            if kopf[:2] != b"\xff\xd8":
                return None
            f.seek(2)
            while True:
                marker = f.read(4)
                if len(marker) < 4 or marker[0] != 0xFF:
                    return None
                kind, length = marker[1], struct.unpack(">H", marker[2:4])[0]
                if 0xC0 <= kind <= 0xCF and kind not in (0xC4, 0xC8, 0xCC):
                    height, width = struct.unpack(">HH", f.read(5)[1:5])
                    return width, height
                f.seek(length - 2, 1)
    except (OSError, struct.error):
        return None


def pruefe_bilder(zeilen: list[dict], max_pixel: int) -> dict:
    """All images must exist; counts those larger than ``max_pixel`` (they are scaled down for the model)."""
    fehlen = [r["images"][0] for r in zeilen if not Path(r["images"][0]).is_file()]
    if fehlen:
        raise Fehler(f"{len(fehlen)} von {len(zeilen)} Seitenbildern fehlen, z. B. {fehlen[0]}. Die Bildpfade legt "
                     "beim Export --bild-basis fest (Ordner der Seitenbilder auf diesem Rechner).")
    flaechen = [w * h for w, h in filter(None, (bildgroesse(Path(r["images"][0])) for r in zeilen))]
    return {"groesser_als_bildgroesse": sum(f > max_pixel for f in flaechen),
            "groesste_mp": round(max(flaechen, default=0) / 1e6, 2)}


def sha256_datei(datei: Path) -> str:
    h = hashlib.sha256()
    with datei.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Runs: numbered folders that are never overwritten
# ---------------------------------------------------------------------------------------------------------------------

def laeufe_ordner() -> Path:
    # TUNE_LAEUFE lets the functional test keep its runs apart from the real ones.
    return Path(os.environ.get("TUNE_LAEUFE") or basis() / "laeufe")


def alle_laeufe() -> list[Path]:
    root = laeufe_ordner()
    return sorted(p for p in root.iterdir() if p.is_dir() and re.match(r"\d{3}-", p.name)) if root.is_dir() else []


def finde_lauf(angabe: str) -> Path:
    """A run by number ("3", "003") or folder name."""
    for p in alle_laeufe():
        if p.name == angabe or (angabe.isdigit() and int(p.name[:3]) == int(angabe)):
            return p
    raise Fehler(f"Lauf {angabe} nicht gefunden in {laeufe_ordner()} (Übersicht: tune.py laeufe).")


def neuer_lauf(modell: str, zusatz: str = "") -> Path:
    """Creates the next numbered run folder; fails rather than reuse one."""
    nummer = max((int(p.name[:3]) for p in alle_laeufe()), default=0) + 1
    name = f"{nummer:03d}-{modell}" + (f"-{zusatz}" if zusatz else "")
    ordner = laeufe_ordner() / name
    ordner.mkdir(parents=True, exist_ok=False)
    return ordner


def lies_lauf(ordner: Path) -> dict:
    datei = ordner / "lauf.json"
    if not datei.is_file():
        raise Fehler(f"{ordner} hat keine lauf.json (kein Lauf dieses Werkzeugs).")
    return json.loads(datei.read_text(encoding="utf-8"))


def schreibe_lauf(ordner: Path, info: dict) -> None:
    (ordner / "lauf.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def yaml_wert(v) -> str:
    """A scalar as YAML. Floats always carry a dot (PyYAML reads "5e-05" as a string)."""
    if isinstance(v, float):
        text = f"{v:.10g}"
        if "." not in text and "inf" not in text and "nan" not in text:
            mantisse, e, exponent = text.partition("e")
            text = mantisse + ".0" + e + exponent
        return text
    return json.dumps(v, ensure_ascii=False)


def schreibe_yaml(datei: Path, werte: dict, kopf: str) -> None:
    zeilen = [f"# {z}" for z in kopf.splitlines()] + [""]
    for k, v in werte.items():
        if k in ERKLAERUNG:
            zeilen.append(f"# {ERKLAERUNG[k]}")
        zeilen.append(f"{k}: {yaml_wert(v)}")
    datei.write_text("\n".join(zeilen) + "\n", encoding="utf-8")


def einstellungen(modell: str, ordner: Path, args, *, mit_validierung: bool, adapter: Path | None) -> dict:
    """The LLaMA-Factory configuration of a training run."""
    m = MODELLE[modell]
    cfg = {
        "model_name_or_path": modell_ort(modell),
        "trust_remote_code": m["trust_remote_code"],
        "template": m["template"],
        "image_max_pixels": args.bild_pixel,
        "image_min_pixels": 32 * 32,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": args.rang,
        "lora_alpha": 2 * args.rang,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "dataset_dir": str(ordner / "daten"),
        "dataset": "training",
        "cutoff_len": args.laenge,
        "preprocessing_num_workers": 4,
        "output_dir": str(ordner / "adapter"),
        "logging_steps": 5,
        "save_strategy": "epoch",
        "plot_loss": True,
        "save_only_model": False,
        "report_to": "none",
        "per_device_train_batch_size": args.stapel,
        "gradient_accumulation_steps": args.sammeln,
        "learning_rate": float(args.lernrate),
        "num_train_epochs": float(args.epochen),
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0.1,
        "bf16": True,
        "seed": 42,
    }
    if adapter:
        cfg["adapter_name_or_path"] = str(adapter)
    if args.schritte:
        cfg["max_steps"] = args.schritte
    if mit_validierung:
        cfg.update({"eval_dataset": "validierung", "eval_strategy": "epoch", "per_device_eval_batch_size": 1})
    return cfg


def schreibe_daten(ordner: Path, train: list[dict], val: list[dict]) -> None:
    """The run's own copy of the data in the layout LLaMA-Factory reads (sharegpt with OpenAI-style roles)."""
    daten = ordner / "daten"
    daten.mkdir()
    spalten = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
               "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                        "assistant_tag": "assistant", "system_tag": "system"}}
    info = {}
    for name, zeilen in (("training", train), ("validierung", val)):
        if zeilen:
            with (daten / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in zeilen:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            info[name] = {"file_name": f"{name}.jsonl", **spalten}
    (daten / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def lies_jsonl_roh(datei: Path) -> list[dict]:
    if not datei.is_file():
        return []
    with datei.open(encoding="utf-8") as f:
        return [json.loads(z) for z in f if z.strip()]


# ---------------------------------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------------------------------

def frei_gb() -> float | None:
    """Memory available right now (the Spark shares one memory between processor and graphics card)."""
    try:
        for zeile in Path("/proc/meminfo").read_text().splitlines():
            if zeile.startswith("MemAvailable:"):
                return int(zeile.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def gesamt_gb() -> float | None:
    try:
        for zeile in Path("/proc/meminfo").read_text().splitlines():
            if zeile.startswith("MemTotal:"):
                return int(zeile.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def modell_ordner(modell: str) -> Path:
    """Where ``tune.py pruefen`` puts a base model: the tool's own folder, not the shared Hugging Face cache (parts
    of that belong to root on the Spark because Docker containers downloaded them)."""
    return basis() / "modelle" / MODELLE[modell]["hf"].split("/")[-1]


def modell_ort(modell: str) -> str:
    """The base model for LLaMA-Factory: the local folder if it is there, else the Hugging Face name."""
    ordner = modell_ordner(modell)
    return str(ordner) if (ordner / "config.json").is_file() else MODELLE[modell]["hf"]


def umgebung(hf_name: str | None = None) -> dict:
    """Environment of the child processes: no telemetry, and no network at all once the model is in the cache."""
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_DISABLED="true",
               GRADIO_ANALYTICS_ENABLED="False", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    if hf_name and Path(hf_name).is_dir():
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def venv_bin(name: str, venv: str = "venv") -> Path:
    datei = basis() / venv / "bin" / name
    if not datei.exists():
        hinweis = " --quantisierung" if venv == "venv-quant" else ""
        raise Fehler(f"{datei} fehlt. Erst einrichten: bash training/einrichten.sh{hinweis}")
    return datei


def _zuerst_beenden() -> None:
    """Runs in the child process: if the machine runs out of memory, the kernel ends this process first, not the
    other services on the machine (the Spark shares one memory between all of them)."""
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass


def laufe(befehl: list[str], protokoll: Path | None, env: dict) -> int:
    """Run a command, show its output and append it to the run's log."""
    print("$ " + " ".join(shlex.quote(str(b)) for b in befehl), flush=True)
    log = protokoll.open("a", encoding="utf-8") if protokoll else None
    try:
        if log:
            log.write(f"\n$ {' '.join(map(str, befehl))}\n")
        proc = subprocess.Popen(list(map(str, befehl)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, errors="replace", preexec_fn=_zuerst_beenden)
        for zeile in proc.stdout:
            sys.stdout.write(zeile)
            if log:
                log.write(zeile)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        return 130
    finally:
        if log:
            log.close()


def git_stand(ordner: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(ordner), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unbekannt"


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def cmd_trainieren(args) -> int:
    modell = args.modell
    eltern = finde_lauf(args.weiter_von) if args.weiter_von else None
    weiter = eltern is not None
    if args.lernrate is None:
        args.lernrate = 5e-5 if weiter else 1e-4
    if args.epochen is None:
        args.epochen = 2 if weiter else 3

    neu, hinweise = lade_daten([Path(d).expanduser() for d in args.daten],
                               [Path(p).expanduser() for p in args.pruefsatz or []])
    train, val = teile(neu, args.val_anteil)
    alt_dazu = 0
    if weiter:
        e = lies_lauf(eltern)
        if e["modell"] != modell:
            raise Fehler(f"Lauf {eltern.name} wurde mit {e['modell']} trainiert, nicht mit {modell}.")
        if e.get("anweisung_sha256") != ANWEISUNG_SHA:
            raise Fehler("ANWEISUNG hat sich seit dem früheren Lauf geändert. Dann nicht weitertrainieren, sondern "
                         "neu trainieren (ohne --weiter-von).")
        if e.get("regeln_sha256") != REGELN_SHA and not args.regeln_geaendert:
            raise Fehler("Die Regeln (PROMPT) haben sich seit dem früheren Lauf geändert. Sind alle Soll-Texte an die "
                         "neuen Regeln angepasst, besser neu trainieren; war die Änderung nur eine Klarstellung, "
                         "mit --regeln-geaendert weitertrainieren.")
        if e.get("status") != "fertig" or not (eltern / "adapter" / "adapter_config.json").is_file():
            raise Fehler(f"Lauf {eltern.name} hat keinen fertigen Adapter.")
        namen, doks = pruef_bilder([Path(p) for p in hinweise["pruefsaetze"]])
        alt_train = [r for r in lies_jsonl_roh(eltern / "daten" / "training.jsonl")
                     if bild_name(r["images"][0]) not in namen and dokument(r["images"][0]) not in doks]
        dazu = mische_alt(train, alt_train, args.alt_anteil)
        alt_dazu = len(dazu)
        train = train + dazu
        schon = {r["images"][0] for r in val}
        val = val + [r for r in lies_jsonl_roh(eltern / "daten" / "validierung.jsonl") if r["images"][0] not in schon]
    if not train:
        raise Fehler("Keine Trainingsseiten übrig.")
    bilder = pruefe_bilder(train + val, args.bild_pixel)

    frei = frei_gb()
    bedarf = MODELLE[modell]["bedarf_gb"]
    if frei is not None and frei < bedarf and not args.trotzdem:
        raise Fehler(f"Nur {frei:.0f} GB Speicher frei, für {modell} werden grob {bedarf} GB gebraucht. Reicht der "
                     "Speicher nicht, kann der Rechner andere Dienste abwürgen. Speicher frei machen (README, "
                     "„Zu wenig Speicher“) oder mit --trotzdem starten.")

    cli = venv_bin("llamafactory-cli")
    ordner = neuer_lauf(modell, args.name or (f"weiter-{eltern.name[:3]}" if weiter else ""))
    schreibe_daten(ordner, train, val)
    cfg = einstellungen(modell, ordner, args, mit_validierung=bool(val),
                        adapter=eltern / "adapter" if weiter else None)
    weg = f"weiter auf dem Adapter von Lauf {eltern.name}" if weiter else "neu"
    schreibe_yaml(ordner / "einstellungen.yaml", cfg,
                  f"Lauf {ordner.name}: {weg}\nEinstellungen für LLaMA-Factory (llamafactory-cli train <diese Datei>)")
    info = {
        "lauf": ordner.name, "modell": modell, "grundmodell": MODELLE[modell]["hf"],
        "weg": "weiter" if weiter else "neu", "weiter_von": eltern.name if weiter else None,
        "begonnen": datetime.datetime.now().isoformat(timespec="seconds"), "status": "läuft",
        "anweisung_sha256": ANWEISUNG_SHA, "regeln_sha256": REGELN_SHA,
        "daten": [{"datei": str(Path(d).expanduser().resolve()), "sha256": sha256_datei(Path(d).expanduser())}
                  for d in args.daten],
        "pruefsaetze_ausgeschlossen": hinweise["pruefsaetze"],
        "seiten": {"training": zaehle(train), "validierung": zaehle(val), "alte_seiten_beigemischt": alt_dazu,
                   "doppelt_in_den_dateien": hinweise["doppelt"], **bilder},
        "tune_commit": git_stand(HERE), "llamafactory_commit": git_stand(basis() / "LLaMA-Factory"),
    }
    schreibe_lauf(ordner, info)

    print(f"Lauf {ordner.name} ({weg}) in {ordner}")
    print(f"  Training:    {len(train)} Seiten – " + ", ".join(f"{t} {n}" for t, n in zaehle(train).items())
          + (f" (davon {alt_dazu} alte beigemischt)" if weiter else ""))
    print(f"  Validierung: {len(val)} Seiten – " + ", ".join(f"{t} {n}" for t, n in zaehle(val).items()))
    if not val:
        print("  Hinweis: keine Validierungsseiten – ohne sie ist Auswendiglernen am Verlust nicht zu erkennen.")
    if bilder["groesser_als_bildgroesse"]:
        print(f"  Hinweis: {bilder['groesser_als_bildgroesse']} Seiten sind größer als die Bildgröße "
              f"({args.bild_pixel / 1e6:.2f} MP, größte {bilder['groesste_mp']} MP) und werden verkleinert.")
    lang = sum(len(r["messages"][1]["content"]) / 2 > args.laenge - args.bild_pixel / 784 - 100 for r in train)
    if lang:
        print(f"  Hinweis: Bei rund {lang} Seiten könnte die Antwort die Länge {args.laenge} sprengen und würde "
              "abgeschnitten (--laenge erhöhen).")

    code = laufe([cli, "train", ordner / "einstellungen.yaml"], ordner / "protokoll.log", umgebung(modell_ort(modell)))
    fertig = code == 0 and (ordner / "adapter" / "adapter_config.json").is_file()
    info["status"] = "fertig" if fertig else ("abgebrochen" if code == 130 else "fehlgeschlagen")
    info["beendet"] = datetime.datetime.now().isoformat(timespec="seconds")
    info["verlust"] = verlust(ordner)
    schreibe_lauf(ordner, info)
    if not fertig:
        print(f"\nLauf {ordner.name}: {info['status']} (Protokoll: {ordner / 'protokoll.log'})")
        return code or 1
    v = info["verlust"]
    print(f"\nLauf {ordner.name} fertig. Verlust Training {v.get('training')}, Validierung {v.get('validierung')}.")
    print(f"Verlustkurve: {ordner / 'adapter' / 'training_loss.png'}")
    print(f"Weiter: python3 training/tune.py zusammenfuehren --lauf {int(ordner.name[:3])}")
    return 0


def verlust(ordner: Path) -> dict:
    """Last training and validation loss from LLaMA-Factory's trainer_log.jsonl, plus the validation loss per epoch."""
    train, val = None, []
    for row in lies_jsonl_roh(ordner / "adapter" / "trainer_log.jsonl"):
        if row.get("loss") is not None:
            train = row["loss"]
        if row.get("eval_loss") is not None:
            val.append(row["eval_loss"])
    if train is None:  # short runs end before the first logging step; the trainer's summary has the mean loss
        try:
            train = json.loads((ordner / "adapter" / "all_results.json").read_text()).get("train_loss")
        except (OSError, ValueError):
            pass
    return {"training": train, "validierung": val[-1] if val else None, "validierung_je_epoche": val}


def cmd_laeufe(_args) -> int:
    laeufe = alle_laeufe()
    if not laeufe:
        print(f"Noch keine Läufe in {laeufe_ordner()}.")
        return 0
    print(f"{'Lauf':<28} {'Weg':<12} {'Status':<14} {'Seiten':>6} {'Val.':>5} {'Verlust':>8} {'Val.-Verlust':>12}  Fassungen")
    for p in laeufe:
        try:
            i = lies_lauf(p)
        except SystemExit:
            continue
        v = i.get("verlust") or {}
        weg = "neu" if i["weg"] == "neu" else f"weiter {i['weiter_von'][:3]}"
        fass = ", ".join(f for f in FASSUNGEN if (p / f"modell-{f}" / "config.json").is_file()) or "–"
        fmt = lambda x: "–" if x is None else f"{x:.4f}"  # noqa: E731
        print(f"{p.name:<28} {weg:<12} {i.get('status', '?'):<14} {sum(i['seiten']['training'].values()):>6} "
              f"{sum(i['seiten']['validierung'].values()):>5} {fmt(v.get('training')):>8} "
              f"{fmt(v.get('validierung')):>12}  {fass}")
    return 0


def fertiger_lauf(angabe: str) -> tuple[Path, dict]:
    ordner = finde_lauf(angabe)
    info = lies_lauf(ordner)
    if info.get("status") != "fertig":
        raise Fehler(f"Lauf {ordner.name} ist nicht fertig (Status: {info.get('status')}).")
    return ordner, info


def cmd_zusammenfuehren(args) -> int:
    ordner, info = fertiger_lauf(args.lauf)
    ziel = ordner / "modell-bf16"
    if ziel.exists():
        raise Fehler(f"{ziel} gibt es schon.")
    m = MODELLE[info["modell"]]
    cfg = {"model_name_or_path": modell_ort(info["modell"]), "adapter_name_or_path": str(ordner / "adapter"), "template": m["template"],
           "trust_remote_code": m["trust_remote_code"], "export_dir": str(ziel), "export_size": 5,
           "export_device": "cpu", "export_legacy_format": False}
    schreibe_yaml(ordner / "zusammenfuehren.yaml", cfg,
                  f"Lauf {ordner.name}: Adapter in das Grundmodell einrechnen (llamafactory-cli export <diese Datei>)")
    code = laufe([venv_bin("llamafactory-cli"), "export", ordner / "zusammenfuehren.yaml"], ordner / "protokoll.log",
                 umgebung(modell_ort(info["modell"])))
    if code or not (ziel / "config.json").is_file():
        shutil.rmtree(ziel, ignore_errors=True)
        print("Zusammenführen fehlgeschlagen.")
        return code or 1
    schreibe_vorlage(ziel, info["modell"])
    print(f"\nZusammengeführt: {ziel}  (Name im Betrieb: {dienstname(ordner, 'bf16')})")
    n = int(ordner.name[:3])
    print(f"Weiter: python3 training/tune.py quantisieren --lauf {n} --fassung fp8   oder   starten --lauf {n} --fassung bf16")
    return 0


def schreibe_vorlage(modell_ordner: Path, modell: str) -> None:
    """Puts the chat template for operation next to the model (see vorlagen/): it wraps a request exactly the way the
    training did, so the model is asked in operation as it was taught."""
    quelle = HERE / "vorlagen" / (MODELLE[modell]["template"] + ".jinja")
    if quelle.is_file():
        shutil.copyfile(quelle, modell_ordner / "betrieb.jinja")


def cmd_quantisieren(args) -> int:
    ordner, info = fertiger_lauf(args.lauf)
    quelle, ziel = ordner / "modell-bf16", ordner / f"modell-{args.fassung}"
    if not (quelle / "config.json").is_file():
        raise Fehler(f"Erst zusammenführen: python3 training/tune.py zusammenfuehren --lauf {int(ordner.name[:3])}")
    if ziel.exists():
        raise Fehler(f"{ziel} gibt es schon.")
    befehl = [venv_bin("python", "venv-quant"), HERE / "quantisieren.py", "--quelle", quelle, "--ziel", ziel,
              "--fassung", args.fassung]
    vorlage = HERE / "vorlagen" / (MODELLE[info["modell"]]["template"] + ".jinja")
    if vorlage.is_file():
        befehl += ["--vorlage", vorlage]
    if args.fassung == "nvfp4":
        # Calibration pages for NVFP4: the run's own training pages (never the test set).
        befehl += ["--kalibrierung", ordner / "daten" / "training.jsonl", "--bild-pixel",
                   str(bild_pixel_des_laufs(ordner)), "--seiten", str(args.seiten)]
    code = laufe(befehl, ordner / "protokoll.log", umgebung(str(quelle)))
    if code or not (ziel / "config.json").is_file():
        shutil.rmtree(ziel, ignore_errors=True)
        print("Quantisieren fehlgeschlagen.")
        return code or 1
    schreibe_vorlage(ziel, info["modell"])
    print(f"\nQuantisiert: {ziel}  (Name im Betrieb: {dienstname(ordner, args.fassung)})")
    print(f"Weiter: python3 training/tune.py starten --lauf {int(ordner.name[:3])} --fassung {args.fassung}")
    return 0


def bild_pixel_des_laufs(ordner: Path) -> int:
    m = re.search(r"^image_max_pixels: (\d+)", (ordner / "einstellungen.yaml").read_text(encoding="utf-8"), re.M)
    return int(m.group(1)) if m else A4_PIXEL


def bild_kwargs(modell: str, pixel: int) -> dict:
    """How the image limit of the training is told to the model's image processor in operation (the two model
    families name it differently)."""
    if MODELLE[modell]["template"] == "glm_ocr":
        # GLM's processor counts a still image as two frames, so its limit is twice the image area (measured).
        return {"size": {"shortest_edge": 112 * 112, "longest_edge": 2 * pixel}}
    return {"max_pixels": pixel}


def dienstname(ordner: Path, fassung: str) -> str:
    """The name a version is served and measured under, e.g. ``003-qwen2b-fp8``."""
    return f"{ordner.name}-{fassung}"


def vllm_befehl(ordner: Path, fassung: str, *, port: int, speicher_gb: float, gesamt: float, parallel: int,
                image: str = VLLM_IMAGE) -> list[str]:
    """The docker command that serves a version with vLLM. vLLM takes its memory limit as a share of the whole
    memory, so the limit in GB is converted."""
    modell = ordner / f"modell-{fassung}"
    name = dienstname(ordner, fassung)
    pixel = bild_pixel_des_laufs(ordner)
    befehl = ["docker", "run", "-d", "--name", f"tune-{name}", "--gpus", "all", "--network", "host", "--ipc", "host",
              "--oom-score-adj", "1000",  # out of memory hits this container first, not the other services
              "-v", f"{modell}:/modell:ro", "--entrypoint", "vllm", image, "serve", "/modell",
              "--served-model-name", name, "--host", "127.0.0.1", "--port", str(port),
              "--gpu-memory-utilization", f"{speicher_gb / gesamt:.3f}",
              "--max-model-len", "16384", "--max-num-batched-tokens", "16384", "--max-num-seqs", str(parallel),
              "--limit-mm-per-prompt", '{"image":1,"video":0}', "--mm-processor-cache-gb", "0",
              "--mm-processor-kwargs", json.dumps(bild_kwargs(lies_lauf(ordner)["modell"], pixel), separators=(",", ":")),
              "--override-generation-config", '{"temperature":0,"max_new_tokens":4096}']
    if (modell / "betrieb.jinja").is_file():
        befehl += ["--chat-template", "/modell/betrieb.jinja"]
    if MODELLE[lies_lauf(ordner)["modell"]]["trust_remote_code"]:
        befehl += ["--trust-remote-code"]
    return befehl


def cmd_starten(args) -> int:
    ordner, _info = fertiger_lauf(args.lauf)
    if not (ordner / f"modell-{args.fassung}" / "config.json").is_file():
        raise Fehler(f"Die Fassung {args.fassung} von Lauf {ordner.name} gibt es noch nicht "
                     "(zusammenfuehren bzw. quantisieren).")
    gesamt = gesamt_gb() or 128.0
    befehl = vllm_befehl(ordner, args.fassung, port=args.port, speicher_gb=args.speicher_gb, gesamt=gesamt,
                         parallel=args.parallel, image=args.image)
    name = dienstname(ordner, args.fassung)
    zeile = " ".join(shlex.quote(b) for b in befehl)
    if not args.ausfuehren:
        print(f"Startbefehl für {name} (höchstens {args.speicher_gb:g} GB von {gesamt:.0f} GB; mit --ausfuehren "
              f"startet ihn dieses Werkzeug):\n\n{zeile}\n")
        print(f"Erreichbar unter http://127.0.0.1:{args.port}/v1, Modellname {name}. "
              f"Anhalten: docker rm -f tune-{name}")
        return 0
    frei = frei_gb()
    if frei is not None and frei < args.speicher_gb + 2 and not args.trotzdem:
        raise Fehler(f"Nur {frei:.0f} GB Speicher frei, verlangt sind {args.speicher_gb:g} GB. Weniger verlangen "
                     "(--speicher-gb), Speicher frei machen oder --trotzdem.")
    print("$ " + zeile)
    code = subprocess.run(befehl).returncode
    if code == 0:
        print(f"Gestartet als Container tune-{name}. Das Laden dauert etwas: docker logs -f tune-{name}\n"
              f"Fragen: python3 training/tune.py fragen --lauf {int(ordner.name[:3])} --fassung {args.fassung} "
              f"--bild <Seite>\nAnhalten: python3 training/tune.py stoppen --lauf {int(ordner.name[:3])} "
              f"--fassung {args.fassung}")
    return code


def cmd_stoppen(args) -> int:
    ordner = finde_lauf(args.lauf)
    # Only ever touches containers this tool started (name prefix "tune-").
    return subprocess.run(["docker", "rm", "-f", f"tune-{dienstname(ordner, args.fassung)}"]).returncode


def frage(url: str, modell: str, bild: Path, *, max_tokens: int = 4096, timeout: float = 600) -> dict:
    """Ask a served model to read one page as in training (ANWEISUNG: the image only); returns the raw response."""
    mime = "image/png" if bild.suffix.lower() == ".png" else "image/jpeg"
    daten = base64.b64encode(bild.read_bytes()).decode("ascii")
    body = {"model": modell, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{daten}"}},
                *([{"type": "text", "text": ANWEISUNG}] if ANWEISUNG else [])]}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise Fehler(f"Der Server lehnt die Anfrage ab ({e.code}): {e.read().decode('utf-8', 'replace')[:600]}") from None
    except OSError as e:
        raise Fehler(f"Server unter {url} nicht erreichbar: {e}") from None


def cmd_fragen(args) -> int:
    if args.lauf:
        modell = dienstname(finde_lauf(args.lauf), args.fassung)
    elif args.modellname:
        modell = args.modellname
    else:
        raise Fehler("--lauf (mit --fassung) oder --modellname angeben.")
    antwort = frage(args.url, modell, Path(args.bild).expanduser())
    wahl = antwort["choices"][0]
    text = wahl["message"]["content"] or ""
    typ, _inhalt = split(text)
    print(f"[{modell}: Typ {typ or 'ohne Typzeile'}, {antwort.get('usage', {}).get('completion_tokens', '?')} "
          f"Antwort-Tokens, Ende: {wahl.get('finish_reason')}]")
    print(text)
    return 0


def cmd_pruefen(args) -> int:
    """Check the environment: packages, graphics card, templates; optionally load the models."""
    python = venv_bin("python")
    print(f"Umgebung: {basis()}   LLaMA-Factory {git_stand(basis() / 'LLaMA-Factory')}")
    frei, gesamt = frei_gb(), gesamt_gb()
    if frei is not None:
        print(f"Speicher: {frei:.0f} GB von {gesamt:.0f} GB verfügbar")
    befehl = [python, HERE / "venv_hilfe.py", "pruefen"]
    for m in args.modell or []:
        befehl += ["--modell", f"{m}|{MODELLE[m]['hf']}|{MODELLE[m]['template']}|{int(MODELLE[m]['trust_remote_code'])}"
                               f"|{modell_ordner(m)}"]
    if args.laden:
        befehl.append("--laden")
    return laufe(befehl, None, dict(umgebung(), TRANSFORMERS_VERBOSITY="error", LLAMAFACTORY_VERBOSITY="WARN"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seitenleser trainieren (LLaMA-Factory, LoRA) – siehe training/README.md")
    sub = ap.add_subparsers(dest="befehl", required=True)

    p = sub.add_parser("pruefen", help="Umgebung prüfen (Pakete, Grafikkarte, Vorlagen)")
    p.add_argument("--modell", action="append", choices=list(MODELLE), help="auch dieses Modell prüfen (mehrfach möglich); "
                   "lädt es beim ersten Mal herunter")
    p.add_argument("--laden", action="store_true", help="die Modelle ganz auf die Grafikkarte laden (braucht Speicher)")
    p.set_defaults(func=cmd_pruefen)

    p = sub.add_parser("trainieren", help="einen Lauf trainieren (neu oder weiter auf einem früheren Adapter)")
    p.add_argument("--modell", required=True, choices=list(MODELLE))
    p.add_argument("--daten", required=True, nargs="+", help="eine oder mehrere JSONL-Dateien aus "
                   "ocr_korrektur.py --export (nie eine *.pruef.jsonl)")
    p.add_argument("--pruefsatz", nargs="*", help="weitere Prüfsatz-Dateien, deren Seiten und Dokumente ausgeschlossen "
                   "sein müssen (die *.pruef.jsonl neben jeder Datendatei wird von selbst geprüft)")
    p.add_argument("--weiter-von", metavar="LAUF", help="auf dem Adapter dieses Laufs weitertrainieren; --daten sind "
                   "dann die neuen Seiten, alte werden beigemischt")
    p.add_argument("--alt-anteil", type=float, default=0.5, help="beim Weitertrainieren: Anteil alter Seiten aus dem "
                   "früheren Lauf an den Trainingsseiten (0.5)")
    p.add_argument("--epochen", type=float, help="wie oft jede Seite gesehen wird (neu: 3, weiter: 2)")
    p.add_argument("--lernrate", type=float, help="Stärke der Lernschritte (neu: 1e-4, weiter: 5e-5)")
    p.add_argument("--rang", type=int, default=8, help="Größe des LoRA-Adapters (8); beim Weitertrainieren gilt der "
                   "Rang des früheren Laufs")
    p.add_argument("--stapel", type=int, default=1, help="Seiten je Stapel (1)")
    p.add_argument("--sammeln", type=int, default=8, help="Stapel je Lernschritt (8)")
    p.add_argument("--bild-pixel", type=int, default=A4_PIXEL, help=f"größte Bildfläche in Pixeln ({A4_PIXEL}: A4 mit "
                   "150 dpi unverkleinert)")
    p.add_argument("--laenge", type=int, default=8192, help="größte Länge aus Bild, Anweisung und Antwort in Tokens (8192)")
    p.add_argument("--val-anteil", type=float, default=0.1, help="Anteil der Dokumente für die Validierung (0.1; 0 = keine)")
    p.add_argument("--schritte", type=int, help="nach so vielen Lernschritten aufhören (nur für Funktionstests)")
    p.add_argument("--name", help="Zusatz zum Namen des Laufordners")
    p.add_argument("--trotzdem", action="store_true", help="auch bei knappem Speicher starten")
    p.add_argument("--regeln-geaendert", action="store_true", help="weitertrainieren, obwohl PROMPT (die Regeln) seit "
                   "dem früheren Lauf geändert wurde")
    p.set_defaults(func=cmd_trainieren)

    p = sub.add_parser("laeufe", help="Übersicht aller Läufe")
    p.set_defaults(func=cmd_laeufe)

    p = sub.add_parser("zusammenfuehren", help="Adapter eines Laufs in das Grundmodell einrechnen (modell-bf16)")
    p.add_argument("--lauf", required=True)
    p.set_defaults(func=cmd_zusammenfuehren)

    p = sub.add_parser("quantisieren", help="das zusammengeführte Modell auf FP8 oder NVFP4 bringen")
    p.add_argument("--lauf", required=True)
    p.add_argument("--fassung", required=True, choices=FASSUNGEN[1:])
    p.add_argument("--seiten", type=int, default=64, help="Kalibrierseiten für NVFP4 (64)")
    p.set_defaults(func=cmd_quantisieren)

    for name, hilfe in (("starten", "Startbefehl für vLLM mit begrenztem Speicher zeigen oder ausführen"),
                        ("stoppen", "den vLLM-Container einer Fassung anhalten und entfernen")):
        p = sub.add_parser(name, help=hilfe)
        p.add_argument("--lauf", required=True)
        p.add_argument("--fassung", required=True, choices=FASSUNGEN)
        if name == "starten":
            p.add_argument("--port", type=int, default=8004)
            p.add_argument("--speicher-gb", type=float, default=10.0, help="Obergrenze des Speichers für vLLM in GB (10)")
            p.add_argument("--parallel", type=int, default=24, help="gleichzeitige Anfragen (24)")
            p.add_argument("--image", default=VLLM_IMAGE, help=f"Docker-Image von vLLM ({VLLM_IMAGE})")
            p.add_argument("--ausfuehren", action="store_true", help="den Befehl ausführen statt nur zeigen")
            p.add_argument("--trotzdem", action="store_true", help="auch bei knappem Speicher starten")
        p.set_defaults(func=cmd_starten if name == "starten" else cmd_stoppen)

    p = sub.add_parser("fragen", help="eine Seite an ein laufendes Modell schicken (wie im Training: nur das Bild)")
    p.add_argument("--bild", required=True)
    p.add_argument("--lauf")
    p.add_argument("--fassung", choices=FASSUNGEN, default="bf16")
    p.add_argument("--modellname", help="Modellname am Server, falls nicht über --lauf")
    p.add_argument("--url", default="http://127.0.0.1:8004/v1")
    p.set_defaults(func=cmd_fragen)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
