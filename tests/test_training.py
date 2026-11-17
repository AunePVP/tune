"""Tests of the training tool (training/tune.py) that need no graphics card: the data checks, the numbered run
folders and the generated settings. ``llamafactory-cli`` is replaced by a small script that behaves like a finished
training. All pages are invented."""
from __future__ import annotations

import json
import re
import stat
import sys
from pathlib import Path

import pytest

from conftest import png

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import tune  # noqa: E402
from ocr_vorgabe import ANWEISUNG, PROMPT  # noqa: E402

FAKE_CLI = """#!/usr/bin/env python3
import json, pathlib, re, sys
cfg = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
get = lambda key: json.loads(re.search(rf"^{key}: (.+)$", cfg, re.M).group(1))
if sys.argv[1] == "train":
    out = pathlib.Path(get("output_dir"))
    out.mkdir(parents=True)
    (out / "adapter_config.json").write_text("{}")
    (out / "trainer_log.jsonl").write_text('{"loss": 1.5}\\n{"eval_loss": 1.25}\\n{"loss": 0.5}\\n{"eval_loss": 0.75}\\n')
else:
    out = pathlib.Path(get("export_dir"))
    out.mkdir(parents=True)
    (out / "config.json").write_text("{}")
print("ok", sys.argv[1])
"""


@pytest.fixture()
def umgebung(tmp_path, monkeypatch):
    """A training environment in a temporary folder with a fake llamafactory-cli and plenty of free memory."""
    root = tmp_path / "tune-training"
    cli = root / "venv" / "bin" / "llamafactory-cli"
    cli.parent.mkdir(parents=True)
    cli.write_text(FAKE_CLI, encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("TUNE_TRAINING", str(root))
    monkeypatch.setattr(tune, "frei_gb", lambda: 100.0)
    return root


def export(folder: Path, name: str, pages: dict[str, str], prompt: str = ANWEISUNG) -> Path:
    """An export file like ocr_korrektur.py writes it; ``pages`` maps image file names to target answers."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for image, answer in pages.items():
        (folder / image).write_bytes(png())
        rows.append({"messages": [{"role": "user", "content": "<image>" + prompt},
                                  {"role": "assistant", "content": answer}], "images": [str(folder / image)]})
    target = folder / name
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return target


def seiten(anzahl: int, start: int = 0) -> dict[str, str]:
    """Invented pages: two pages per document, the four types in turn."""
    antworten = ["TYP: Text\nPrüfbericht Nr. 4711", "TYP: Plan\nGrundriss EG M 1:100", "TYP: Foto", "TYP: Leer"]
    return {f"{i:03d}_Akte{i // 2}.pdf_S{i % 2 + 1}.png": antworten[i % 4] for i in range(start, start + anzahl)}


def einstellung(datei: Path, key: str):
    return json.loads(re.search(rf"^{key}: (.+)$", datei.read_text(encoding="utf-8"), re.M).group(1))


def test_pruefsatz_wird_nie_trainiert(tmp_path, umgebung):
    daten = export(tmp_path / "d", "training.pruef.jsonl", seiten(4))
    with pytest.raises(SystemExit, match="Prüfsatz"):
        tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten)])
    # A test set renamed to look like training data is caught by the test file next to it.
    export(tmp_path / "e", "training.pruef.jsonl", seiten(4))
    getarnt = export(tmp_path / "e", "training.jsonl", {**seiten(2), **seiten(6, start=10)})
    with pytest.raises(SystemExit, match="Prüfsatz"):
        tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(getarnt)])
    assert not tune.alle_laeufe()


def test_dokumente_des_pruefsatzes_sind_ausgeschlossen(tmp_path, umgebung):
    # Page 2 of a document whose page 1 is in the test set must not be trained either.
    export(tmp_path / "d", "alt.pruef.jsonl", {"001_Akte7.pdf_S1.png": "TYP: Leer"})
    daten = export(tmp_path / "d", "training.jsonl", {"002_Akte7.pdf_S2.png": "TYP: Leer", **seiten(4)})
    with pytest.raises(SystemExit, match="Prüfsatz"):
        tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten),
                   "--pruefsatz", str(tmp_path / "d" / "alt.pruef.jsonl")])


def test_andere_anweisung_und_fehlende_typzeile_werden_abgelehnt(tmp_path):
    with pytest.raises(SystemExit, match="ANWEISUNG"):
        tune.lies_jsonl(export(tmp_path, "a.jsonl", seiten(2), prompt=PROMPT))  # an export from before the decision
    with pytest.raises(SystemExit, match="Typzeile"):
        tune.lies_jsonl(export(tmp_path, "b.jsonl", {"001_A.pdf_S1.png": "Sehr geehrte Damen und Herren"}))


def test_validierung_nimmt_ganze_dokumente(tmp_path):
    zeilen = tune.lies_jsonl(export(tmp_path, "t.jsonl", seiten(200)))
    train, val = tune.teile(zeilen, 0.2)
    dok = lambda rows: {tune.dokument(r["images"][0]) for r in rows}  # noqa: E731
    assert val and len(train) + len(val) == 200
    assert not dok(train) & dok(val)
    assert dok(val) <= dok(tune.teile(zeilen, 0.4)[1])  # a larger share only adds documents
    assert tune.dokument(r"C:\training\012_Bericht 2019.pdf_S3.jpg") == "Bericht 2019.pdf"


def test_lauf_wird_nummeriert_abgelegt_und_nie_ueberschrieben(tmp_path, umgebung, capsys):
    daten = export(tmp_path / "d", "training.jsonl", seiten(40))
    assert tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten)]) == 0
    assert tune.main(["trainieren", "--modell", "glm-ocr", "--daten", str(daten), "--name", "probe"]) == 0
    erster, zweiter = tune.alle_laeufe()
    assert (erster.name, zweiter.name) == ("001-qwen2b", "002-glm-ocr-probe")

    cfg = erster / "einstellungen.yaml"
    assert einstellung(cfg, "model_name_or_path") == "Qwen/Qwen3.5-2B"
    assert einstellung(cfg, "template") == "qwen3_5_nothink"
    assert einstellung(cfg, "image_max_pixels") >= 1240 * 1754  # A4 at 150 dpi is not scaled down
    assert einstellung(cfg, "finetuning_type") == "lora" and einstellung(cfg, "bf16") is True
    assert einstellung(cfg, "warmup_steps") == 0.1  # a share; transformers 5 dropped warmup_ratio
    assert "adapter_name_or_path" not in cfg.read_text(encoding="utf-8")
    assert einstellung(zweiter / "einstellungen.yaml", "template") == "glm_ocr"
    assert einstellung(zweiter / "einstellungen.yaml", "trust_remote_code") is True

    info = json.loads((erster / "lauf.json").read_text(encoding="utf-8"))
    n_train, n_val = sum(info["seiten"]["training"].values()), sum(info["seiten"]["validierung"].values())
    assert n_train + n_val == 40 and n_val > 0
    assert info["status"] == "fertig" and info["weg"] == "neu"
    assert info["anweisung_sha256"] == tune.ANWEISUNG_SHA and info["regeln_sha256"] == tune.REGELN_SHA and len(info["daten"][0]["sha256"]) == 64
    assert info["verlust"] == {"training": 0.5, "validierung": 0.75, "validierung_je_epoche": [1.25, 0.75]}
    assert len((erster / "daten" / "training.jsonl").read_text(encoding="utf-8").splitlines()) == n_train
    assert set(json.loads((erster / "daten" / "dataset_info.json").read_text())) == {"training", "validierung"}

    assert tune.main(["laeufe"]) == 0
    assert "001-qwen2b" in capsys.readouterr().out
    # A number is never given twice, even if a run in between was removed or failed.
    (umgebung / "laeufe" / "007-irgendwas").mkdir()
    assert tune.neuer_lauf("qwen2b").name == "008-qwen2b"


def test_weitertrainieren_nutzt_adapter_mischt_alte_seiten_und_lernt_kleiner(tmp_path, umgebung):
    alt = export(tmp_path / "d1", "training.jsonl", seiten(60))
    neu = export(tmp_path / "d2", "training-2.jsonl", seiten(20, start=100))
    assert tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(alt), "--rang", "16"]) == 0
    assert tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(neu), "--weiter-von", "1"]) == 0
    erster, zweiter = tune.alle_laeufe()
    assert zweiter.name == "002-qwen2b-weiter-001"
    cfg = zweiter / "einstellungen.yaml"
    assert einstellung(cfg, "adapter_name_or_path") == str(erster / "adapter")
    assert einstellung(cfg, "learning_rate") < einstellung(erster / "einstellungen.yaml", "learning_rate")
    info = json.loads((zweiter / "lauf.json").read_text(encoding="utf-8"))
    assert info["weg"] == "weiter" and info["weiter_von"] == "001-qwen2b"
    gemischt = [json.loads(z)["images"][0] for z in (zweiter / "daten" / "training.jsonl").read_text().splitlines()]
    alte = [b for b in gemischt if "/d1/" in b]
    assert info["seiten"]["alte_seiten_beigemischt"] == len(alte) == len(gemischt) - len(alte) > 0
    # Old validation pages stay validation pages and are never mixed into the training.
    alt_val = {json.loads(z)["images"][0] for z in (erster / "daten" / "validierung.jsonl").read_text().splitlines()}
    assert not alt_val & set(gemischt)

    with pytest.raises(SystemExit, match="glm-ocr"):
        tune.main(["trainieren", "--modell", "glm-ocr", "--daten", str(neu), "--weiter-von", "1"])


def test_weitertrainieren_nach_geaenderter_vorgabe_wird_abgelehnt(tmp_path, umgebung):
    daten = export(tmp_path / "d", "training.jsonl", seiten(20))
    assert tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten)]) == 0
    lauf = tune.alle_laeufe()[0]
    info = json.loads((lauf / "lauf.json").read_text(encoding="utf-8"))
    info["regeln_sha256"] = "0" * 64  # as if PROMPT (the rules) had been changed since
    tune.schreibe_lauf(lauf, info)
    with pytest.raises(SystemExit, match="neu trainieren"):
        tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten), "--weiter-von", "1"])
    assert tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten), "--weiter-von", "1",
                      "--regeln-geaendert"]) == 0
    info["anweisung_sha256"] = "0" * 64  # a different request to the model: never continue
    tune.schreibe_lauf(lauf, info)
    with pytest.raises(SystemExit, match="ANWEISUNG"):
        tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten), "--weiter-von", "1",
                   "--regeln-geaendert"])


def test_fehlende_bilder_und_knapper_speicher_halten_den_start_auf(tmp_path, umgebung, monkeypatch):
    daten = export(tmp_path / "d", "training.jsonl", seiten(20))
    monkeypatch.setattr(tune, "frei_gb", lambda: 3.0)
    with pytest.raises(SystemExit, match="Speicher"):
        tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten)])
    assert tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten), "--trotzdem"]) == 0
    next((tmp_path / "d").glob("*.png")).unlink()
    with pytest.raises(SystemExit, match="Seitenbildern fehlen"):
        tune.main(["trainieren", "--modell", "qwen2b", "--daten", str(daten), "--trotzdem"])
    assert len(tune.alle_laeufe()) == 1


def test_yaml_zahlen_und_bildgroesse(tmp_path):
    assert tune.yaml_wert(5e-5) == "5.0e-05" and tune.yaml_wert(1e-4) == "0.0001" and tune.yaml_wert(3.0) == "3.0"
    assert tune.yaml_wert(True) == "true" and tune.yaml_wert("glm_ocr") == '"glm_ocr"'
    bild = tmp_path / "s.png"
    bild.write_bytes(png(1240, 20))
    assert tune.bildgroesse(bild) == (1240, 20)
    assert tune.pruefe_bilder([{"images": [str(bild)]}], 1000) == {"groesser_als_bildgroesse": 1, "groesste_mp": 0.02}


def test_zusammenfuehren_und_startbefehl_mit_speichergrenze(tmp_path, umgebung, capsys):
    daten = export(tmp_path / "d", "training.jsonl", seiten(20))
    assert tune.main(["trainieren", "--modell", "glm-ocr", "--daten", str(daten)]) == 0
    with pytest.raises(SystemExit, match="gibt es noch nicht"):
        tune.main(["starten", "--lauf", "1", "--fassung", "bf16"])
    assert tune.main(["zusammenfuehren", "--lauf", "1"]) == 0
    lauf = tune.alle_laeufe()[0]
    assert einstellung(lauf / "zusammenfuehren.yaml", "adapter_name_or_path") == str(lauf / "adapter")
    assert (lauf / "modell-bf16" / "betrieb.jinja").is_file()
    with pytest.raises(SystemExit, match="gibt es schon"):
        tune.main(["zusammenfuehren", "--lauf", "1"])

    befehl = tune.vllm_befehl(lauf, "bf16", port=8004, speicher_gb=12, gesamt=120, parallel=8)
    assert befehl[befehl.index("--gpu-memory-utilization") + 1] == "0.100"
    assert befehl[befehl.index("--served-model-name") + 1] == "001-glm-ocr-bf16"
    assert befehl[befehl.index("--name") + 1] == "tune-001-glm-ocr-bf16"
    assert "--chat-template" in befehl and "--trust-remote-code" in befehl
    # The image limit of the training also applies in operation; each model family names it differently.
    assert tune.bild_kwargs("qwen2b", tune.A4_PIXEL) == {"max_pixels": tune.A4_PIXEL}
    assert json.loads(befehl[befehl.index("--mm-processor-kwargs") + 1])["size"]["longest_edge"] == 2 * tune.A4_PIXEL
    assert befehl[befehl.index("--max-num-batched-tokens") + 1] == "16384" and "--oom-score-adj" in befehl
    capsys.readouterr()
    assert tune.main(["starten", "--lauf", "1", "--fassung", "bf16"]) == 0  # only prints the command
    assert "docker run" in capsys.readouterr().out


def test_fragen_schickt_bild_und_vorgabe(tmp_path, chat_server, capsys):
    bild = tmp_path / "seite.png"
    bild.write_bytes(png())
    chat_server["answers"] = ["TYP: Leer"]
    assert tune.main(["fragen", "--bild", str(bild), "--modellname", "probe", "--url", chat_server["url"]]) == 0
    inhalt = chat_server["requests"][0]["messages"][0]["content"]
    assert inhalt[0]["image_url"]["url"].startswith("data:image/png;base64,") and len(inhalt) == 1  # the image only
    assert "Typ Leer" in capsys.readouterr().out
