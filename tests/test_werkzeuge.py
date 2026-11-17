"""The tools: the target form (ocr_vorgabe), the comparison (ocr_vergleich), drafts (ocr_entwurf) and the correction
tool with its JSONL export (ocr_korrektur). Invented pages only – no real data."""
from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request

import ocr_entwurf
import ocr_korrektur
import ocr_vergleich
import ocr_vorgabe

from conftest import png


def _pages(folder, n: int) -> None:
    for k in range(n):
        (folder / f"00{k}_Beispiel_S1.png").write_bytes(png())


def test_the_target_form_round_trips():
    assert ocr_vorgabe.split("TYP: Plan\nGrundriss EG, M 1:100") == ("Plan", "Grundriss EG, M 1:100")
    assert ocr_vorgabe.split("**TYP: Leer**") == ("Leer", "")
    assert ocr_vorgabe.split("Grundriss EG") == (None, "Grundriss EG")
    assert ocr_vorgabe.join("Leer", "Logo") == "TYP: Leer"
    assert set(ocr_vorgabe.TYPES) == {"Text", "Plan", "Foto", "Leer"}
    assert "Raumnamen" in ocr_vorgabe.PROMPT and "Erfinde nichts" in ocr_vorgabe.PROMPT


def test_the_loop_check():
    assert ocr_vergleich.looped("Kopf\n" + "| BSK 12 | geprüft | i.O. | Wartung 03/2024 |\n" * 40)
    assert ocr_vergleich.looped("Text " + " ".join(f"[Stempel: {k}]" for k in range(12)))
    assert not ocr_vergleich.looped("\n".join(f"| BSK {i} | geprüft |" for i in range(80)))


def test_the_comparison_writes_texts_summary_and_html(tmp_path, chat_server):
    pages = tmp_path / "pruef"
    pages.mkdir()
    _pages(pages, 2)
    (pages / "000_Beispiel_S1.tesseract.txt").write_text("bisheriger Text", encoding="utf-8")
    chat_server["answers"] = ["Abschrift der Seite"]
    rows = ocr_vergleich.run(pages, [ocr_vergleich.parse_model(f"Fake|{chat_server['url']}|m|OCR:")], parallel=2,
                             max_tokens=50, timeout=10, out=tmp_path / "ergebnis",
                             extra={"chat_template_kwargs": {"enable_thinking": False}})
    assert len(rows) == 2 and not any(r["fehler"] for r in rows)
    assert chat_server["requests"][0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat_server["requests"][0]["messages"][0]["content"][0]["text"] == "OCR:"
    report = (tmp_path / "ergebnis" / "vergleich.html").read_text(encoding="utf-8")
    assert "bisheriger Text" in report and "Abschrift der Seite" in report
    assert ocr_vergleich.summary(rows)[1].startswith("Fake | 2 |")


def test_drafts_follow_the_target_form_and_resume(tmp_path, chat_server):
    _pages(tmp_path, 3)
    chat_server["answers"] = ["TYP: Plan\nGrundriss EG"]
    model = ocr_vergleich.parse_model(f"Qwen|{chat_server['url']}|m")
    stats = ocr_entwurf.draft(tmp_path, model, parallel=1, max_tokens=50, timeout=5, extra=None)
    assert stats == {"seiten": 3, "fehler": 0, "ohne_typ": 0, "schleife": 0}
    assert chat_server["requests"][0]["messages"][0]["content"][0]["text"] == ocr_vorgabe.PROMPT
    assert (tmp_path / "000_Beispiel_S1.entwurf.txt").read_text(encoding="utf-8") == "TYP: Plan\nGrundriss EG"
    assert ocr_entwurf.draft(tmp_path, model, parallel=1, max_tokens=50, timeout=5, extra=None)["seiten"] == 0


def test_the_correction_tool_saves_discards_and_exports(tmp_path):
    _pages(tmp_path, 3)
    (tmp_path / "000_Beispiel_S1.entwurf.txt").write_text("TYP: Plan\nGrundriss", encoding="utf-8")
    tool = ocr_korrektur.Tool(tmp_path)
    server = ocr_korrektur.ThreadingHTTPServer(("127.0.0.1", 0), ocr_korrektur.handler(tool))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(i: int, **form) -> str:
        data = urllib.parse.urlencode(form).encode()
        return opener.open(urllib.request.Request(f"{base}/seite/{i}", data=data, method="POST")).url

    try:
        page = opener.open(f"{base}/seite/0").read().decode()
        assert "Grundriss" in page and "value='Plan' accesskey='2' checked" in page and "0 von 3 korrigiert" in page
        assert opener.open(f"{base}/bild/0").read().startswith(b"\x89PNG")
        assert post(0, typ="Plan", text="Grundriss EG\r\nIndex C", aktion="speichern").endswith("/seite/1")
        post(1, typ="Text", text="", aktion="verwerfen")
        post(2, typ="Leer", text="Logo", aktion="speichern")
    finally:
        server.shutdown()
    assert (tmp_path / "000_Beispiel_S1.soll.txt").read_text(encoding="utf-8") == "TYP: Plan\nGrundriss EG\nIndex C"
    assert (tmp_path / "001_Beispiel_S1.nicht-verwenden").exists()
    assert (tmp_path / "002_Beispiel_S1.soll.txt").read_text(encoding="utf-8") == "TYP: Leer"
    counts = ocr_korrektur.export(tmp_path, tmp_path / "training.jsonl")
    lines = [json.loads(x) for x in (tmp_path / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    assert counts == {"Text": 0, "Plan": 1, "Foto": 0, "Leer": 1} and len(lines) == 2
    assert lines[0]["messages"][0]["content"] == "<image>" + ocr_vorgabe.PROMPT
    assert lines[0]["messages"][1]["content"] == "TYP: Plan\nGrundriss EG\nIndex C"
    assert lines[0]["images"][0].endswith("000_Beispiel_S1.png")
