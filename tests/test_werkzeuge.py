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
    assert stats == {"seiten": 3, "fehler": 0, "ohne_typ": 0, "schleife": 0, "meldungen": {}}
    assert chat_server["requests"][0]["messages"][0]["content"][0]["text"] == ocr_vorgabe.PROMPT
    assert (tmp_path / "000_Beispiel_S1.entwurf.txt").read_text(encoding="utf-8") == "TYP: Plan\nGrundriss EG"
    assert ocr_entwurf.draft(tmp_path, model, parallel=1, max_tokens=50, timeout=5, extra=None)["seiten"] == 0


def test_drafts_name_the_reason_when_the_model_is_unreachable(tmp_path, capsys):
    _pages(tmp_path, 2)
    model = ocr_vergleich.parse_model("Qwen|http://127.0.0.1:9/v1|m")  # nothing listens on port 9
    stats = ocr_entwurf.draft(tmp_path, model, parallel=1, max_tokens=50, timeout=5, extra=None)
    assert stats["fehler"] == 2 and sum(stats["meldungen"].values()) == 2
    assert capsys.readouterr().out.count("Fehler bei") == 1  # each reason once, not once per page
    assert not list(tmp_path.glob("*.entwurf.txt"))


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
    assert counts["training"] == {"Text": 0, "Plan": 1, "Foto": 0, "Leer": 1} and len(lines) == 2
    assert lines[0]["messages"][0]["content"] == "<image>" + ocr_vorgabe.ANWEISUNG == "<image>"
    assert lines[0]["messages"][1]["content"] == "TYP: Plan\nGrundriss EG\nIndex C"
    assert lines[0]["images"][0].endswith("000_Beispiel_S1.png")


def test_the_export_sets_whole_documents_aside_as_the_test_set(tmp_path, capsys):
    folder = tmp_path / "training"
    folder.mkdir()
    listing = []
    for k in range(40):  # 20 invented documents with two pages each
        name = f"{k:03d}_Beispiel{k // 2}_S{k % 2 + 1}.png"
        (folder / name).write_bytes(png())
        (folder / name).with_suffix(".soll.txt").write_text("TYP: Text\nSeite", encoding="utf-8")
        listing.append({"datei": name, "dokument": f"Projekt/Beispiel{k // 2}.pdf", "seite": k % 2 + 1})
    (folder / "seiten.json").write_text(json.dumps(listing), encoding="utf-8")
    other = tmp_path / "seiten.json"
    other.write_text(json.dumps([{"datei": "x.png", "dokument": "Projekt/Beispiel0.pdf"}]), encoding="utf-8")

    def docs(path) -> list[str]:
        known = ocr_korrektur.documents(folder)
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        assert all(r["images"][0].startswith("/home/md/training/") for r in rows)
        return [known[r["images"][0].rsplit("/", 1)[1]] for r in rows]

    args = ["--ordner", str(folder), "--export", str(tmp_path / "t.jsonl"), "--bild-basis", "/home/md/training/",
            "--ohne-dokumente", str(other)]
    assert ocr_korrektur.main(args) == 0
    train, test = docs(tmp_path / "t.jsonl"), docs(tmp_path / "t.pruef.jsonl")
    assert test and not set(train) & set(test) and len(train) + len(test) == 38
    assert "Projekt/Beispiel0.pdf" not in train + test and "2 Seiten ausgelassen" in capsys.readouterr().out
    assert ocr_korrektur.main(args) == 0 and docs(tmp_path / "t.pruef.jsonl") == test  # the same split every time
    assert ocr_korrektur.main(args + ["--pruef-anteil", "0.5"]) == 0
    assert set(test) <= set(docs(tmp_path / "t.pruef.jsonl"))  # a larger share only adds documents
    assert ocr_korrektur.main(args + ["--pruef-anteil", "0"]) == 0 and len(docs(tmp_path / "t.jsonl")) == 38
    assert ocr_korrektur.document_of(folder / "007_Beispiel_S12.png", {}) == "Beispiel"
    assert ocr_korrektur.image_path(folder / "a.png", "D:\\training") == "D:\\training\\a.png"


def test_the_comparison_with_the_target_prompt_counts_the_types(tmp_path, chat_server, capsys):
    pages = tmp_path / "pruef"
    pages.mkdir()
    _pages(pages, 4)
    chat_server["answers"] = ["TYP: Plan\nGrundriss EG\n[Stempel: Genehmigt 2005]", "TYP: Leer\nEin Logo", "TYP: Foto",
                              "Abschrift ohne Typzeile"]
    args = ["--ordner", str(pages), "--modell", f"Fake|{chat_server['url']}|m", "--parallel", "1", "--vorgabe",
            "--ausgabe", str(tmp_path / "ergebnis")]
    assert ocr_vergleich.main(args) == 0
    assert chat_server["requests"][0]["messages"][0]["content"][0]["text"] == ocr_vorgabe.PROMPT
    head, line = capsys.readouterr().out.splitlines()[1:3]
    assert head.endswith("Text | Plan | Foto | Leer | ohne Typzeile | Leer mit Inhalt | Foto ohne Inhalt | "
                         "Seiten mit Stempel | Stempel gesamt")
    assert line.endswith("| 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1")
    assert "Typ: Plan" in (tmp_path / "ergebnis" / "vergleich.html").read_text(encoding="utf-8")


def test_the_correction_tool_exports_and_starts_a_training_from_the_browser(tmp_path):
    folder = tmp_path / "seiten"
    folder.mkdir()
    _pages(folder, 3)
    for i, answer in enumerate(["TYP: Text\nBericht", "TYP: Leer"]):
        (folder / f"00{i}_Beispiel_S1.soll.txt").write_text(answer, encoding="utf-8")
    # Stands in for training/tune.py: records how it was called.
    fake = tmp_path / "tune.py"
    fake.write_text("import sys, pathlib\nif sys.argv[1] == 'trainieren':\n    print('Lauf 001-qwen2b fertig')\n"
                    "    pathlib.Path(__file__).with_name('aufruf.txt').write_text(' '.join(sys.argv[1:]))\n")
    tool = ocr_korrektur.Tool(folder, tune_py=fake, share=0.0)
    server = ocr_korrektur.ThreadingHTTPServer(("127.0.0.1", 0), ocr_korrektur.handler(tool))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert "href='/training'" in opener.open(f"{base}/seite/0").read().decode()
        page = opener.open(f"{base}/training").read().decode()
        assert "2 von 3 Seiten korrigiert" in page and "Exportieren und trainieren" in page
        data = urllib.parse.urlencode({"modell": "glm-ocr"}).encode()
        answer = opener.open(urllib.request.Request(f"{base}/training", data=data, method="POST")).read().decode()
        assert "Training mit GLM-OCR gestartet (2 Seiten)" in answer
        tool.job.wait(timeout=30)
        assert "Lauf 001-qwen2b fertig" in opener.open(f"{base}/training").read().decode()  # the end of the log
    finally:
        server.shutdown()
    target = tmp_path / "seiten.jsonl"
    assert len(target.read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "aufruf.txt").read_text() == f"trainieren --modell glm-ocr --daten {target}"
    # Without the training tool next to the scripts (the Windows PC) there is no training page.
    assert ocr_korrektur.Tool(folder)._training_link() == ""


def test_text_lines_with_their_place_on_the_page(tmp_path):
    import ocr_bloecke
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "1\t1\t0\t0\t0\t0\t0\t0\t1000\t2000\t-1\t\n"
           "4\t1\t1\t1\t1\t0\t100\t200\t500\t40\t-1\t\n"
           "5\t1\t1\t1\t1\t1\t100\t200\t240\t40\t96\tPrüfbericht\n"
           "5\t1\t1\t1\t1\t2\t360\t200\t240\t40\t91\tNr. 4711\n"
           "4\t1\t2\t1\t1\t0\t0\t900\t1000\t4\t-1\t\n"
           "5\t1\t2\t1\t1\t1\t0\t900\t1000\t4\t30\t—\n")
    assert ocr_bloecke.parse_tsv(tsv) == [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.02, "t": "Prüfbericht Nr. 4711"}]
    assert ocr_bloecke.parse_tsv("") == []
    assert ocr_bloecke.score(tsv) == (len("Prüfbericht") + len("Nr4711"), (96 + 91 + 30) / 3)
    # A line found on the page turned clockwise, as a position on the page as it is stored.
    line = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.05, "t": "a"}
    assert ocr_bloecke.unrotate(line, 0) == line
    assert ocr_bloecke.unrotate(line, 90) == {"x": 0.2, "y": 0.4, "w": 0.05, "h": 0.5, "t": "a"}
    assert ocr_bloecke.unrotate(line, 180) == {"x": 0.4, "y": 0.75, "w": 0.5, "h": 0.05, "t": "a"}
    assert ocr_bloecke.unrotate(line, 270) == {"x": 0.75, "y": 0.1, "w": 0.05, "h": 0.5, "t": "a"}
    # The correction tool hands the lines of a page to the browser, and an empty list where there are none.
    _pages(tmp_path, 2)
    (tmp_path / "000_Beispiel_S1.bloecke.json").write_text('[{"x":0.1,"y":0.1,"w":0.5,"h":0.02,"t":"Prüfbericht"}]',
                                                           encoding="utf-8")
    tool = ocr_korrektur.Tool(tmp_path)
    server = ocr_korrektur.ThreadingHTTPServer(("127.0.0.1", 0), ocr_korrektur.handler(tool))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert json.loads(opener.open(f"{base}/bloecke/0").read())[0]["t"] == "Prüfbericht"
        assert json.loads(opener.open(f"{base}/bloecke/1").read()) == []
        assert "id='marke'" in opener.open(f"{base}/seite/0").read().decode()
    finally:
        server.shutdown()
    assert len(ocr_korrektur.pages_in(tmp_path)) == 2  # the side files are no pages
