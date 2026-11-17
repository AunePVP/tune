#!/usr/bin/env python3
"""Checks that need the training environment (run by ``tune.py pruefen`` with the venv's Python): package versions,
the graphics card, and for each model whether it loads and whether the chat template for operation (vorlagen/*.jinja)
wraps a request exactly as LLaMA-Factory does in training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from ocr_vorgabe import ANWEISUNG  # noqa: E402


def zusammengefasst(ids: list[int], bild_token: int) -> list[int]:
    """Collapse the run of image placeholder tokens (one per image patch) to a single one."""
    out: list[int] = []
    for i in ids:
        if not (i == bild_token and out and out[-1] == bild_token):
            out.append(i)
    return out


def pruefe_modell(kurz: str, hf: str, vorlage: str, trust: bool, ordner: str, laden: bool) -> bool:
    import torch
    from huggingface_hub import snapshot_download
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer
    from PIL import Image

    print(f"\n{kurz}: {hf}")
    if not (Path(ordner) / "config.json").is_file():
        print(f"  wird heruntergeladen nach {ordner} …")
        snapshot_download(hf, local_dir=ordner)
    print(f"  liegt in {ordner}")
    model_args, data_args, finetuning_args, _ = get_infer_args(
        {"model_name_or_path": ordner, "template": vorlage, "trust_remote_code": trust})
    module = load_tokenizer(model_args)
    tokenizer, processor = module["tokenizer"], module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # What training feeds the model for one page ...
    messages = [{"role": "user", "content": "<image>" + ANWEISUNG}, {"role": "assistant", "content": "TYP: Leer"}]
    bild = Image.new("RGB", (64, 64), "white")
    messages = template.mm_plugin.process_messages(messages, [bild], [], [], processor)
    prompt_ids, antwort_ids = template.encode_oneturn(tokenizer, messages)
    bild_token = tokenizer.convert_tokens_to_ids(template.mm_plugin.image_token)
    # ... and what the operation template makes of the same request.
    jinja = (HERE / "vorlagen" / f"{vorlage}.jinja").read_text(encoding="utf-8")
    anfrage = [{"role": "user", "content": [*([{"type": "text", "text": ANWEISUNG}] if ANWEISUNG else []), {"type": "image"}]}]
    text = tokenizer.apply_chat_template(anfrage, chat_template=jinja, add_generation_prompt=True, tokenize=False)
    betrieb_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    # The request in operation must be the exact beginning of what training showed the model (some templates count
    # a line break before the answer as part of the answer, so the whole sequence is compared, not the prompt alone).
    training_ids = zusammengefasst(prompt_ids + antwort_ids, bild_token)
    betrieb_ids = zusammengefasst(betrieb_ids, bild_token)
    gleich = len(betrieb_ids) >= len(zusammengefasst(prompt_ids, bild_token)) and \
        training_ids[: len(betrieb_ids)] == betrieb_ids
    print(f"  Vorlage {vorlage}: Anfrage im Betrieb (vorlagen/{vorlage}.jinja) "
          + ("stimmt mit dem Training überein" if gleich else "WEICHT VOM TRAINING AB"))
    print(f"  Im Training folgt darauf die Antwort: {tokenizer.decode(training_ids[len(betrieb_ids):])!r}")
    if not gleich:
        print("    Training:", repr(tokenizer.decode(zusammengefasst(prompt_ids, bild_token))[:160]), "…",
              repr(tokenizer.decode(prompt_ids[-12:])))
        print("    Betrieb: ", repr(text[:160]), "…", repr(text[-60:]))
    if laden:
        from llamafactory.model import load_model
        vorher = torch.cuda.memory_allocated()
        model = load_model(tokenizer, model_args, finetuning_args)
        n = sum(p.numel() for p in model.parameters())
        print(f"  geladen: {n / 1e9:.2f} Mrd. Parameter, {(torch.cuda.memory_allocated() - vorher) / 1024 ** 3:.1f} GB "
              f"auf {next(model.parameters()).device}")
        del model
        torch.cuda.empty_cache()
    return gleich


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("befehl", choices=["pruefen"])
    ap.add_argument("--modell", action="append", default=[], help="kurz|hf|vorlage|trust(0/1)|ordner")
    ap.add_argument("--laden", action="store_true")
    args = ap.parse_args()

    import peft
    import torch
    import transformers
    from llamafactory.extras.env import VERSION

    print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, "
          f"peft {peft.__version__}, LLaMA-Factory {VERSION}")
    if not torch.cuda.is_available():
        print("FEHLER: PyTorch sieht keine Grafikkarte.")
        return 1
    print(f"Grafikkarte: {torch.cuda.get_device_name(0)}, BF16: {'ja' if torch.cuda.is_bf16_supported() else 'nein'}")
    ok = True
    for angabe in args.modell:
        kurz, hf, vorlage, trust, ordner = angabe.split("|")
        ok = pruefe_modell(kurz, hf, vorlage, trust == "1", ordner, args.laden) and ok
    print("\nAlles in Ordnung." if ok else "\nEs gibt Abweichungen (siehe oben).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
