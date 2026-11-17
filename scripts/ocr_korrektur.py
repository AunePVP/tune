#!/usr/bin/env python3
"""Correct the training pages in the browser: the page image on the left, the draft (``<Seite>.entwurf.txt``) on the
right to edit, the page type to choose – "Speichern & weiter" writes ``<Seite>.soll.txt`` in the target form
(ocr_vorgabe.py). Runs on this computer only (127.0.0.1), standard library only; the pages stay here.

    python ocr_korrektur.py --ordner training            # then open http://127.0.0.1:8765
    python ocr_korrektur.py --ordner training --export training.jsonl

``--export`` writes the corrected pages as training data (one JSON line per page: the image, the prompt, the target
answer – the "messages" + "images" layout LLaMA-Factory and ms-swift read). Pages marked "nicht verwenden" are left out.
Whole documents (``dokument`` in seiten.json) are set aside as the test set in ``<export>.pruef.jsonl`` and never go into
the training file; the split is a fixed hash of the document, so it is the same on every run.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import re
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_vergleich import MIME, pages_in  # noqa: E402
from ocr_vorgabe import PROMPT, TYPES, join, split  # noqa: E402

SKIPPED = ".nicht-verwenden"


def _side(page: Path, suffix: str) -> Path:
    return page.with_name(page.stem + suffix)


def state(page: Path) -> str:
    if _side(page, SKIPPED).exists():
        return "verworfen"
    return "korrigiert" if _side(page, ".soll.txt").exists() else "offen"


def current(page: Path) -> tuple[str, str]:
    """(type, text) to show: the correction if there is one, else the draft, else an empty text page."""
    for suffix in (".soll.txt", ".entwurf.txt"):
        f = _side(page, suffix)
        if f.exists():
            page_type, text = split(f.read_text(encoding="utf-8"))
            return page_type or "Text", text
    return "Text", ""


def save(page: Path, page_type: str, text: str) -> None:
    if page_type not in TYPES:
        raise ValueError(page_type)
    _side(page, SKIPPED).unlink(missing_ok=True)
    _side(page, ".soll.txt").write_text(join(page_type, text.replace("\r\n", "\n")), encoding="utf-8")


def discard(page: Path) -> None:
    _side(page, ".soll.txt").unlink(missing_ok=True)
    _side(page, SKIPPED).write_text("", encoding="utf-8")


def documents(folder: Path) -> dict[str, str]:
    """Page file name -> document, from the ``seiten.json`` of the export (empty if there is none)."""
    listing = folder / "seiten.json" if folder.is_dir() else folder
    if not listing.exists():
        return {}
    data = json.loads(listing.read_text(encoding="utf-8"))
    rows = data.get("seiten", []) if isinstance(data, dict) else data
    return {str(r["datei"]): str(r["dokument"]) for r in rows if isinstance(r, dict) and r.get("datei") and r.get("dokument")}


def document_of(page: Path, known: dict[str, str]) -> str:
    """The document a page belongs to; without seiten.json the file name without number and page suffix."""
    return known.get(page.name) or re.sub(r"_S\d+$", "", re.sub(r"^\d+_", "", page.stem))


def is_test(document: str, share: float) -> bool:
    """Whether a document belongs to the test set: a fixed hash of its name, so the split is the same on every run and
    on every computer, and a larger share only ever adds documents to the test set."""
    return int(hashlib.sha256(document.encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32 < share


def image_path(page: Path, base: str | None) -> str:
    """The image path written to the JSONL: absolute here, or below ``base`` on the training computer."""
    if not base:
        return str(page.resolve())
    sep = "\\" if "\\" in base and "/" not in base else "/"
    return base.rstrip("/\\") + sep + page.name


def export(folder: Path, target: Path, *, test_target: Path | None = None, share: float = 0.0,
           base: str | None = None, without: frozenset[str] = frozenset()) -> dict:
    """Write the corrected pages as JSONL. With ``test_target`` and ``share`` whole documents go to the test file
    instead of the training file; documents in ``without`` (those of another test set) are left out altogether."""
    known = documents(folder)
    counts = {"training": {t: 0 for t in TYPES}, "pruef": {t: 0 for t in TYPES}, "ausgelassen": 0}
    with contextlib.ExitStack() as stack:
        train = stack.enter_context(target.open("w", encoding="utf-8"))
        test = stack.enter_context(test_target.open("w", encoding="utf-8")) if test_target else None
        for page in pages_in(folder):
            if state(page) != "korrigiert":
                continue
            document = document_of(page, known)
            if document in without:
                counts["ausgelassen"] += 1
                continue
            to_test = test is not None and is_test(document, share)
            answer = _side(page, ".soll.txt").read_text(encoding="utf-8").strip()
            page_type, _text = split(answer)
            counts["pruef" if to_test else "training"][page_type or "Text"] += 1
            (test if to_test else train).write(json.dumps(
                {"messages": [{"role": "user", "content": "<image>" + PROMPT},
                              {"role": "assistant", "content": answer}],
                 "images": [image_path(page, base)]}, ensure_ascii=False) + "\n")
    return counts


_STYLE = ("body{font:14px system-ui,sans-serif;margin:0}header{padding:.5rem 1rem;background:#f3f4f6;display:flex;"
          "gap:1rem;align-items:center;flex-wrap:wrap}main{display:grid;grid-template-columns:1fr 1fr;gap:1rem;"
          "padding:1rem}img{max-width:100%;border:1px solid #ccc}textarea{width:100%;height:70vh;font:13px "
          "ui-monospace,monospace}.typ label{margin-right:1rem}button{padding:.4rem .9rem;margin-right:.4rem}"
          ".offen{color:#888}.korrigiert{color:#176}.verworfen{color:#a40}details{margin-top:.6rem}"
          "pre{white-space:pre-wrap;max-height:20rem;overflow:auto;background:#fafafa;padding:.4rem}")


MODELS = {"qwen2b": "Qwen3.5-2B", "glm-ocr": "GLM-OCR", "qwen4b": "Qwen3.5-4B (Rückfallebene)"}
TRAIN_FROM = 200  # corrected pages from which a first training is worthwhile


class Tool:
    def __init__(self, folder: Path, *, tune_py: Path | None = None, share: float = 0.2,
                 without: frozenset[str] = frozenset()):
        self.folder = folder
        self.pages = pages_in(folder)
        # Training from the browser (only where the training tool is installed, i.e. on the Spark): export the
        # corrected pages and start training/tune.py, so nobody has to log in for it.
        self.tune_py = tune_py if tune_py and tune_py.is_file() else None
        self.share, self.without = share, without
        self.target = folder.resolve().parent / (folder.resolve().name + ".jsonl")
        self.log = self.target.with_suffix(".training.log")
        self.job: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.job is not None and self.job.poll() is None

    def start_training(self, model: str) -> str:
        """Export the corrected pages (training file and test set) and start a training run in the background."""
        if not self.tune_py or model not in MODELS:
            return "Training ist hier nicht eingerichtet."
        with self.lock:
            if self.running():
                return "Es läuft schon ein Training."
            test_target = self.target.with_name(self.target.stem + ".pruef.jsonl") if self.share > 0 else None
            counts = export(self.folder, self.target, test_target=test_target, share=self.share, without=self.without)
            n_train, n_test = sum(counts["training"].values()), sum(counts["pruef"].values())
            if not n_train:
                return "Noch keine korrigierten Seiten für das Training."
            log = self.log.open("w", encoding="utf-8")
            log.write(f"Export: {n_train} Trainingsseiten nach {self.target}, {n_test} Prüfseiten beiseitegelegt.\n")
            log.flush()
            # Its own session: the training goes on even if this tool is stopped.
            self.job = subprocess.Popen([sys.executable, str(self.tune_py), "trainieren", "--modell", model,
                                         "--daten", str(self.target)], stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            return f"Training mit {MODELS[model]} gestartet ({n_train} Seiten)."

    def training_html(self, message: str = "") -> str:
        by_type = {t: 0 for t in TYPES}
        for page in self.pages:
            if state(page) == "korrigiert":
                by_type[split(_side(page, ".soll.txt").read_text(encoding="utf-8"))[0] or "Text"] += 1
        done = sum(by_type.values())
        running = self.running()
        if running:
            status = "Ein Training läuft. Diese Seite lädt sich alle 20 Sekunden neu."
        elif self.job is not None:
            status = "Das letzte Training ist fertig." if self.job.returncode == 0 else \
                f"Das letzte Training ist abgebrochen (Code {self.job.returncode}) – siehe Protokoll unten."
        else:
            status = "Seit dem Start dieses Werkzeugs wurde kein Training gestartet."
        hint = (f"Ab rund {TRAIN_FROM} korrigierten Seiten lohnt ein erster Lauf." if done < TRAIN_FROM
                else "Genug Seiten für einen Lauf.")
        runs = ""
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            runs = subprocess.run([sys.executable, str(self.tune_py), "laeufe"], capture_output=True, text=True,
                                  timeout=30).stdout
        tail = ""
        if self.log.exists():
            lines = self.log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
            tail = "\n".join(line[:240] for line in lines[-40:] if line.strip())
        radios = "".join(f"<label><input type='radio' name='modell' value='{k}'{' checked' if k == 'qwen2b' else ''}>"
                         f" {html.escape(v)}</label>" for k, v in MODELS.items())
        return (f"<!doctype html><meta charset='utf-8'><title>Training</title>"
                + ("<meta http-equiv='refresh' content='20;url=/training'>" if running else "")
                + f"<style>{_STYLE}section{{padding:1rem;max-width:70rem}}</style><header><b>Training</b>"
                f"<a href='/'>← zurück zum Korrigieren</a></header><section>"
                + (f"<p><b>{html.escape(message)}</b></p>" if message else "")
                + f"<p>{done} von {len(self.pages)} Seiten korrigiert: "
                + ", ".join(f"{t} {n}" for t, n in by_type.items()) + f". {hint}</p>"
                f"<form method='post' action='/training'><div class='typ'>Modell: {radios}</div>"
                f"<button{' disabled' if running else ''}>Exportieren und trainieren</button></form>"
                f"<p>Der Knopf schreibt die korrigierten Seiten nach <code>{html.escape(str(self.target))}</code> "
                f"(Prüfsatz daneben, wird nie trainiert) und startet einen neuen nummerierten Lauf. Weiter "
                f"korrigieren geht währenddessen. Es läuft immer nur ein Training.</p><p>{status}</p>"
                f"<h3>Läufe</h3><pre>{html.escape(runs or '–')}</pre>"
                f"<h3>Protokoll (Ende)</h3><pre>{html.escape(tail or '–')}</pre></section>")

    def _training_link(self) -> str:
        return "<a href='/training'>Training</a>" if self.tune_py else ""

    def next_open(self, after: int = -1) -> int:
        n = len(self.pages)
        for k in range(1, n + 1):
            i = (after + k) % n
            if state(self.pages[i]) == "offen":
                return i
        return min(after + 1, n - 1)

    def page_html(self, i: int) -> str:
        page = self.pages[i]
        page_type, text = current(page)
        done = sum(state(p) == "korrigiert" for p in self.pages)
        known = _side(page, ".tesseract.txt")
        radios = "".join(
            f"<label><input type='radio' name='typ' value='{t}' accesskey='{k}'{' checked' if t == page_type else ''}>"
            f" {t}</label>" for k, t in enumerate(TYPES, start=1))
        return (f"<!doctype html><meta charset='utf-8'><title>{i + 1}/{len(self.pages)} {html.escape(page.name)}</title>"
                f"<style>{_STYLE}</style><header><b>{done} von {len(self.pages)} korrigiert</b>"
                f"<span>Seite {i + 1}: {html.escape(page.name)} – <span class='{state(page)}'>{state(page)}</span></span>"
                f"<a href='/seite/{max(0, i - 1)}'>← zurück</a><a href='/seite/{min(len(self.pages) - 1, i + 1)}'>"
                f"weiter →</a><a href='/'>nächste offene</a>{self._training_link()}</header><main><div><img src='/bild/{i}' alt=''></div>"
                f"<form method='post' action='/seite/{i}'><div class='typ'>Typ: {radios}</div>"
                f"<textarea name='text' autofocus>{html.escape(text)}</textarea><div>"
                f"<button name='aktion' value='speichern'>Speichern &amp; weiter (Strg+Enter)</button>"
                f"<button name='aktion' value='verwerfen' formnovalidate>nicht verwenden</button></div>"
                f"<details><summary>Text bisher (Textebene/Tesseract)</summary><pre>"
                f"{html.escape(known.read_text(encoding='utf-8') if known.exists() else '–')}</pre></details>"
                f"<details><summary>Regeln</summary><pre>{html.escape(PROMPT)}</pre></details></form></main>"
                "<script>document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter'){"
                "document.querySelector(\"button[value=speichern]\").click()}})</script>")


def handler(tool: Tool):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _go(self, i: int) -> None:
            self.send_response(303)
            self.send_header("Location", f"/seite/{i}")
            self.end_headers()

        def _index(self, part: str) -> int | None:
            try:
                i = int(part)
            except ValueError:
                return None
            return i if 0 <= i < len(tool.pages) else None

        def do_GET(self):  # noqa: N802
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == [""]:
                return self._go(tool.next_open())
            if parts == ["training"] and tool.tune_py:
                return self._send(200, tool.training_html().encode("utf-8"))
            if len(parts) == 2 and (i := self._index(parts[1])) is not None:
                if parts[0] == "seite":
                    return self._send(200, tool.page_html(i).encode("utf-8"))
                if parts[0] == "bild":
                    page = tool.pages[i]
                    return self._send(200, page.read_bytes(), MIME[page.suffix.lower()])
            self._send(404, b"nicht gefunden")

        def do_POST(self):  # noqa: N802
            parts = self.path.strip("/").split("/")
            if parts == ["training"] and tool.tune_py:
                form = urllib.parse.parse_qs(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode())
                message = tool.start_training((form.get("modell") or [""])[0])
                return self._send(200, tool.training_html(message).encode("utf-8"))
            if len(parts) != 2 or parts[0] != "seite" or (i := self._index(parts[1])) is None:
                return self._send(404, b"nicht gefunden")
            form = urllib.parse.parse_qs(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8"),
                                         keep_blank_values=True)
            if (form.get("aktion") or [""])[0] == "verwerfen":
                discard(tool.pages[i])
            else:
                save(tool.pages[i], (form.get("typ") or ["Text"])[0], (form.get("text") or [""])[0])
            self._go(tool.next_open(i))

    return H


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trainingsseiten im Browser korrigieren")
    ap.add_argument("--ordner", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Adresse, unter der das Werkzeug erreichbar ist (Voreinstellung nur dieser Rechner). Für den "
                         "Zugriff über Tailscale die Tailscale-Adresse des Rechners angeben, nie 0.0.0.0: Das "
                         "Werkzeug hat keine Anmeldung")
    ap.add_argument("--export", help="korrigierte Seiten als Trainingsdaten (JSONL) in diese Datei schreiben")
    ap.add_argument("--pruef-anteil", type=float, default=0.2,
                    help="Anteil der Dokumente, die als Prüfsatz beiseitegelegt und nie trainiert werden (0.2). Die "
                         "Aufteilung ist fest je Dokument; den Anteil später nie verkleinern. 0 = kein Prüfsatz, "
                         "1 = der ganze Ordner ist Prüfsatz")
    ap.add_argument("--ohne-dokumente", help="seiten.json eines anderen Prüfsatzes: Seiten aus dessen Dokumenten "
                                             "werden ausgelassen")
    ap.add_argument("--bild-basis", help="Ordner der Seitenbilder auf dem Trainingsrechner, z. B. /home/md/training "
                                         "(Voreinstellung: der absolute Pfad auf diesem Rechner)")
    args = ap.parse_args(argv)
    folder = Path(args.ordner)
    if args.export:
        if not 0 <= args.pruef_anteil <= 1:
            raise SystemExit("--pruef-anteil muss zwischen 0 und 1 liegen")
        target = Path(args.export)
        test_target = target.with_name(target.stem + ".pruef.jsonl") if args.pruef_anteil > 0 else None
        without = frozenset(documents(Path(args.ohne_dokumente)).values()) if args.ohne_dokumente else frozenset()
        if args.ohne_dokumente and not without:
            raise SystemExit(f"Keine Dokumente in {args.ohne_dokumente} gefunden")
        counts = export(folder, target, test_target=test_target, share=args.pruef_anteil, base=args.bild_basis,
                        without=without)

        def line(name: str, where: Path) -> str:
            c = counts[name]
            return f"{sum(c.values())} Seiten nach {where}: " + ", ".join(f"{t} {n}" for t, n in c.items())
        print("Training: " + line("training", target))
        if test_target:
            print("Prüfsatz (nie trainieren): " + line("pruef", test_target))
        if counts["ausgelassen"]:
            print(f"{counts['ausgelassen']} Seiten ausgelassen (Dokumente aus {args.ohne_dokumente})")
        if not documents(folder):
            print("Hinweis: keine seiten.json im Ordner – das Dokument wurde aus dem Dateinamen abgeleitet.")
        return 0
    without = frozenset(documents(Path(args.ohne_dokumente)).values()) if args.ohne_dokumente else frozenset()
    tool = Tool(folder, tune_py=Path(__file__).resolve().parents[1] / "training" / "tune.py",
                share=args.pruef_anteil, without=without)
    if not tool.pages:
        raise SystemExit(f"Keine Seitenbilder in {folder}")
    server = ThreadingHTTPServer((args.host, args.port), handler(tool))
    print(f"{len(tool.pages)} Seiten – im Browser öffnen: http://{args.host}:{args.port}  (beenden mit Strg+C)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
