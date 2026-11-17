#!/usr/bin/env python3
"""Quantize a merged model for operation in vLLM with llm-compressor (run by ``tune.py quantisieren`` with the Python
of ``venv-quant``). Only the linear layers of the language part are quantized; the vision part and the output layer
stay in full precision.

  fp8    weights and activations in 8-bit floating point, activation scales found at run time: needs no data.
  nvfp4  weights and activations in NVIDIA's 4-bit floating point: the activation scales are calibrated on pages,
         here on training pages of the run (image + PROMPT), wrapped as in operation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from ocr_vorgabe import PROMPT  # noqa: E402

# Kept in full precision: output layer, vision tower and the bridge between vision and language part.
AUSGENOMMEN = ["re:.*lm_head", "re:.*visual.*", "re:.*vision.*", "re:.*embed_tokens$"]


def kalibrierdaten(datei: Path, processor, vorlage: str | None, max_pixel: int, anzahl: int):
    """The first ``anzahl`` pages of a JSONL as model inputs (image scaled down to ``max_pixel`` like in training)."""
    from datasets import Dataset
    from PIL import Image

    zeilen = []
    with datei.open(encoding="utf-8") as f:
        for zeile in f:
            if zeile.strip():
                zeilen.append(json.loads(zeile)["images"][0])
            if len(zeilen) >= anzahl:
                break
    anfrage = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    kwargs = {"chat_template": vorlage} if vorlage else {}
    text = processor.apply_chat_template(anfrage, add_generation_prompt=True, tokenize=False, **kwargs)

    def eingabe(row):
        bild = Image.open(row["bild"]).convert("RGB")
        if bild.width * bild.height > max_pixel:
            faktor = math.sqrt(max_pixel / (bild.width * bild.height))
            bild = bild.resize((max(28, int(bild.width * faktor)), max(28, int(bild.height * faktor))))
        return processor(text=[text], images=[bild], padding=False)

    return Dataset.from_dict({"bild": zeilen}).map(eingabe, remove_columns=["bild"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quelle", required=True)
    ap.add_argument("--ziel", required=True)
    ap.add_argument("--fassung", required=True, choices=["fp8", "nvfp4"])
    ap.add_argument("--kalibrierung", help="JSONL mit Seiten für die Kalibrierung (nvfp4)")
    ap.add_argument("--vorlage", help="Jinja-Vorlage des Betriebs")
    ap.add_argument("--bild-pixel", type=int, default=1536 * 1536)
    ap.add_argument("--seiten", type=int, default=64)
    args = ap.parse_args()

    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(args.quelle, dtype="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.quelle, trust_remote_code=True)
    schema = {"fp8": "FP8_DYNAMIC", "nvfp4": "NVFP4"}[args.fassung]
    recipe = QuantizationModifier(targets="Linear", scheme=schema, ignore=AUSGENOMMEN)

    if args.fassung == "fp8":
        oneshot(model=model, recipe=recipe)
    else:
        if not args.kalibrierung:
            raise SystemExit("FEHLER: nvfp4 braucht --kalibrierung.")
        vorlage = Path(args.vorlage).read_text(encoding="utf-8") if args.vorlage else None
        daten = kalibrierdaten(Path(args.kalibrierung), processor, vorlage, args.bild_pixel, args.seiten)
        print(f"Kalibrierung mit {len(daten)} Seiten")

        def collator(batch):
            assert len(batch) == 1
            return {k: torch.tensor(v) for k, v in batch[0].items()}

        oneshot(model=model, processor=processor, recipe=recipe, dataset=daten, data_collator=collator,
                max_seq_length=8192, num_calibration_samples=len(daten))

    quantisiert = sum(hasattr(m, "quantization_scheme") for m in model.modules())
    linear = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    print(f"{quantisiert} von {linear} linearen Schichten quantisiert ({schema}); Bildteil und Ausgabeschicht bleiben.")
    if not quantisiert:
        raise SystemExit("FEHLER: keine Schicht quantisiert.")
    model.save_pretrained(args.ziel, save_compressed=True)
    processor.save_pretrained(args.ziel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
