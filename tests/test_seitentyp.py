from __future__ import annotations

import csv

import ocr_vergleich
import seitentyp
from conftest import png


def test_parse_understands_the_answer_forms():
    assert seitentyp.parse("TYP: Plan") == "Plan"
    assert seitentyp.parse("typ: bild\nweiterer Text") == "Bild"
    assert seitentyp.parse("Leer") == "Leer"
    assert seitentyp.parse("TYP: Unsicher") == "Unsicher"
    assert seitentyp.parse("Das ist ein Foto") is None
    assert seitentyp.parse("") is None


def test_pages_are_typed_and_the_run_resumes(tmp_path, chat_server):
    for i in range(3):
        (tmp_path / f"00{i}_Beispiel_S1.png").write_bytes(png())
    chat_server["answers"] = ["TYP: Plan", "weiß nicht", "TYP: Bild"]
    model = ocr_vergleich.parse_model(f"Qwen|{chat_server['url']}|m")
    model["prompt"] = seitentyp.PROMPT
    stats = seitentyp.classify(tmp_path, model, parallel=1, max_tokens=20, timeout=5, extra=None)
    assert stats["seiten"] == 3 and stats["unverstanden"] == 1
    assert chat_server["requests"][0]["messages"][0]["content"][0]["text"] == seitentyp.PROMPT
    with (tmp_path / "seitentypen.csv").open(encoding="utf-8") as f:
        rows = {r["datei"]: r["typ"] for r in csv.DictReader(f, delimiter=";")}
    assert sorted(rows.values()) == ["", "Bild", "Plan"]
    chat_server["answers"] = ["TYP: Leer"]
    assert seitentyp.classify(tmp_path, model, parallel=1, max_tokens=20, timeout=5, extra=None)["seiten"] == 1
