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
from ocr_vorgabe import ANWEISUNG, PROMPT, TYPES, join, split  # noqa: E402

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


def has_text(page: Path) -> bool:
    """Whether there is a correction or a draft; without either the type shown is only the default."""
    return _side(page, ".soll.txt").exists() or _side(page, ".entwurf.txt").exists()


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
                {"messages": [{"role": "user", "content": "<image>" + ANWEISUNG},
                              {"role": "assistant", "content": answer}],
                 "images": [image_path(page, base)]}, ensure_ascii=False) + "\n")
    return counts


_STYLE = """
:root{--bg:#16181d;--panel:#1e2128;--line:#2e333d;--text:#e7e9ee;--dim:#9aa1ae;--accent:#5b7cfa;--ok:#3fb98a;--warn:#d98a4a}
*{box-sizing:border-box}html,body{height:100%}
body{font:14px system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:var(--bg);color:var(--text);
display:flex;flex-direction:column}
a{color:inherit;text-decoration:none}
header{flex:none;height:52px;padding:0 1rem;background:var(--panel);border-bottom:1px solid var(--line);display:flex;
gap:.8rem;align-items:center}
header b{font-size:16px}header .sp{flex:1}
.badge{font-size:12px;color:var(--dim);background:var(--bg);border:1px solid var(--line);border-radius:999px;padding:.15rem .6rem}
button,.btn{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:6px;
padding:.45rem .9rem;cursor:pointer;white-space:nowrap}
button:hover,.btn:hover{border-color:var(--dim)}button:disabled{opacity:.5;cursor:default}
.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
main{flex:1;min-height:0;display:grid;grid-template-columns:290px minmax(0,1fr) minmax(0,1fr)}
nav{display:flex;flex-direction:column;min-height:0;border-right:1px solid var(--line)}
nav select{margin:.6rem;padding:.45rem;font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);
border-radius:6px}
#liste{overflow-y:auto;flex:1}
.it{display:flex;gap:.5rem;align-items:center;padding:.5rem .8rem;border-left:3px solid transparent;
border-bottom:1px solid var(--line);color:var(--dim)}
.it:hover{background:var(--panel)}.it.cur{background:#2a3350;border-left-color:var(--accent);color:var(--text)}
.it .n{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.it .t{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:.05rem .5rem}
.it[data-s=korrigiert] .n::before{content:"✓ ";color:var(--ok)}.it[data-s=verworfen] .n{text-decoration:line-through}
#mitte{display:flex;flex-direction:column;min-height:0;min-width:0;background:#0f1014}
#werkzeug{flex:none;display:flex;gap:.3rem;align-items:center;padding:.4rem .7rem;border-bottom:1px solid var(--line)}
#werkzeug button{padding:.25rem .6rem}#stufe{color:var(--dim);font-size:12px;min-width:3.5rem;text-align:center}
#bild{flex:1;min-height:0;overflow:auto;padding:.7rem;display:flex}
#blatt{position:relative;margin:auto;flex:none}
#dreh{position:absolute;left:50%;top:50%}
#dreh img{display:block;width:100%;height:100%;background:#fff;border:1px solid var(--line);cursor:crosshair}
#marke{position:absolute;display:none;border:2px solid var(--accent);background:rgba(91,124,250,.22);pointer-events:none}
#hinweis{margin-left:.6rem;color:var(--dim);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
textarea::selection{background:var(--accent);color:#fff}
form.seite{display:flex;flex-direction:column;min-height:0;padding:.7rem;gap:.6rem;border-left:1px solid var(--line)}
.typ{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap}.typ b{margin-right:.3rem}
.typ label{border:1px solid var(--line);border-radius:6px;padding:.35rem .7rem;cursor:pointer}
.typ input{position:absolute;opacity:0}.typ label:has(input:checked){background:var(--accent);border-color:var(--accent);color:#fff}
.typ .st{margin-left:auto;color:var(--dim);font-size:12px}
textarea{flex:1;min-height:0;width:100%;resize:none;font:13px/1.5 ui-monospace,Consolas,monospace;color:var(--text);
background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.7rem}
textarea:focus{outline:1px solid var(--accent)}
.knoepfe{display:flex;gap:.5rem;align-items:center}.knoepfe .sp{flex:1}.gefahr{color:var(--warn)}
details{color:var(--dim)}details pre{max-height:14rem}
pre{white-space:pre-wrap;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.6rem;
font:12px/1.45 ui-monospace,Consolas,monospace}
section{padding:1rem 1.2rem;max-width:72rem;overflow:auto}code{color:var(--dim)}
@media(max-width:1000px){main{grid-template-columns:1fr;overflow:auto}nav{max-height:30vh}textarea{min-height:50vh}}
"""

_SCRIPT = r"""
const liste=document.getElementById('liste'),wahl=document.getElementById('filter');
function filtern(){const f=wahl.value;try{localStorage.setItem('filter',f)}catch(e){}
 for(const a of liste.children)a.style.display=(f==='alle'||a.dataset.s===f||a.classList.contains('cur'))?'':'none'}
try{wahl.value=localStorage.getItem('filter')||'offen'}catch(e){}
wahl.addEventListener('change',filtern);filtern();
const cur=liste.querySelector('.cur');if(cur)cur.scrollIntoView({block:'center'});
// Zoom and rotation of the page image: for reading only, the image file is not changed. The rotation is remembered
// per page in this browser.
const bild=document.getElementById('bild'),blatt=document.getElementById('blatt'),dreh=document.getElementById('dreh'),
 img=dreh.querySelector('img'),marke=document.getElementById('marke'),stufe=document.getElementById('stufe'),
 hinweis=document.getElementById('hinweis'),feld=document.querySelector('textarea'),merk='drehung:'+img.dataset.name;
let grad=0,zoom=null;try{grad=parseInt(localStorage.getItem(merk)||'0')||0}catch(e){}
function passend(){const nw=img.naturalWidth,nh=img.naturalHeight,quer=grad%180!==0;
 return Math.min((bild.clientWidth-24)/(quer?nh:nw),(bild.clientHeight-24)/(quer?nw:nh))}
function zeichnen(){const nw=img.naturalWidth,nh=img.naturalHeight;if(!nw)return;
 const quer=grad%180!==0,z=zoom===null?passend():zoom;
 blatt.style.width=(quer?nh:nw)*z+'px';blatt.style.height=(quer?nw:nh)*z+'px';
 dreh.style.width=nw*z+'px';dreh.style.height=nh*z+'px';
 dreh.style.transform='translate(-50%,-50%) rotate('+grad+'deg)';stufe.textContent=Math.round(z*100)+' %'}
function drehen(d){grad=(grad+d+360)%360;try{localStorage.setItem(merk,grad)}catch(e){};zeichnen()}
function zoomen(f){zoom=Math.min(4,Math.max(.1,(zoom===null?passend():zoom)*f));zeichnen()}
document.getElementById('links').onclick=()=>drehen(-90);document.getElementById('rechts').onclick=()=>drehen(90);
document.getElementById('kleiner').onclick=()=>zoomen(1/1.25);document.getElementById('groesser').onclick=()=>zoomen(1.25);
document.getElementById('passen').onclick=()=>{zoom=null;zeichnen()};
document.getElementById('voll').onclick=()=>{zoom=1;zeichnen()};
img.ondblclick=()=>{zoom=zoom===null?1:null;zeichnen()};
bild.addEventListener('wheel',e=>{if(e.ctrlKey){e.preventDefault();zoomen(e.deltaY<0?1.15:1/1.15)}},{passive:false});
window.addEventListener('resize',zeichnen);img.addEventListener('load',zeichnen);zeichnen();
// Text lines with their place on the page (from ocr_bloecke.py): a click on the page marks the matching text on the
// right, the cursor in the text marks the line on the page. The match is found by comparing letter triples, because
// the text on the right is a draft by another model and never identical to what was read at that place.
let zeilen=[];
fetch('/bloecke/'+img.dataset.nr).then(r=>r.json()).then(d=>{zeilen=d;
 hinweis.textContent=d.length?'Klick auf eine Textstelle markiert sie rechts':'keine Textlagen für diese Seite'}).catch(()=>{});
function norm(t){return t.toLowerCase().replace(/[^a-z0-9äöüß]+/g,'')}
function tripel(t){const m=new Map();for(let k=0;k+3<=t.length;k++){const g=t.substr(k,3);m.set(g,(m.get(g)||0)+1)}return m}
function gleichheit(a,b){let n=0,sa=0,sb=0;for(const v of a.values())sa+=v;for(const v of b.values())sb+=v;
 if(!sa||!sb)return 0;for(const[g,v]of a)n+=Math.min(v,b.get(g)||0);return 2*n/(sa+sb)}
function zeigeMarke(z){if(!z){marke.style.display='none';return}
 marke.style.left=z.x*100+'%';marke.style.top=z.y*100+'%';marke.style.width=z.w*100+'%';marke.style.height=z.h*100+'%';
 marke.style.display='block'}
function sucheImText(wort){const ziel=norm(wort);if(ziel.length<3)return null;
 // positions of the kept characters, so a match in the normalised text can be mapped back to the textarea
 const roh=feld.value,orte=[];let glatt='';
 for(let k=0;k<roh.length;k++){const c=norm(roh[k]);if(c){glatt+=c;orte.push(k)}}
 const tz=tripel(ziel),len=ziel.length,schritt=Math.max(1,Math.floor(len/6));let best=0,wo=-1;
 for(let k=0;k+Math.min(len,glatt.length)<=glatt.length;k+=schritt){
  const g=gleichheit(tz,tripel(glatt.substr(k,len)));if(g>best){best=g;wo=k}}
 for(let k=Math.max(0,wo-schritt);wo>=0&&k<=Math.min(glatt.length-1,wo+schritt);k++){
  const g=gleichheit(tz,tripel(glatt.substr(k,len)));if(g>best){best=g;wo=k}}
 if(wo<0||best<0.45)return null;
 return{von:orte[wo],bis:orte[Math.min(orte.length-1,wo+len-1)]+1,guete:best}}
img.addEventListener('click',e=>{if(!zeilen.length)return;
 // the click in the rotated view, converted back to the unrotated image (fractions of width and height)
 const r=img.getBoundingClientRect(),u=(e.clientX-r.left)/r.width,v=(e.clientY-r.top)/r.height;
 const p=grad===90?[v,1-u]:grad===180?[1-u,1-v]:grad===270?[1-v,u]:[u,v];
 let treffer=null,nah=1e9;
 for(const z of zeilen){const dx=Math.max(z.x-p[0],0,p[0]-z.x-z.w),dy=Math.max(z.y-p[1],0,p[1]-z.y-z.h),d=dx*dx+dy*dy;
  if(d<nah){nah=d;treffer=z}}
 if(!treffer||nah>0.0004){zeigeMarke(null);hinweis.textContent='hier wurde kein Text erkannt';return}
 zeigeMarke(treffer);const f=sucheImText(treffer.t);
 if(!f){hinweis.textContent='im Text rechts nicht gefunden: „'+treffer.t.slice(0,60)+'“';return}
 hinweis.textContent='';feld.focus();feld.setSelectionRange(f.von,f.bis);
 const vor=feld.value.slice(0,f.von).split('\n').length-1,alle=feld.value.split('\n').length;
 feld.scrollTop=Math.max(0,feld.scrollHeight*vor/Math.max(1,alle)-feld.clientHeight/3)});
function vomText(){if(!zeilen.length||feld.selectionStart!==feld.selectionEnd)return;
 const t=feld.value,c=feld.selectionStart,a=t.lastIndexOf('\n',c-1)+1;let b=t.indexOf('\n',c);if(b<0)b=t.length;
 let stueck=t.slice(a,b);if(stueck.length>90)stueck=t.slice(Math.max(a,c-45),Math.min(b,c+45));
 const ts=tripel(norm(stueck));let best=0,treffer=null;
 for(const z of zeilen){const zt=norm(z.t),g=zt.length<=norm(stueck).length?gleichheit(tripel(zt),ts)
   :gleichheit(ts,tripel(zt));if(g>best){best=g;treffer=z}}
 zeigeMarke(best>=0.5?treffer:null)}
feld.addEventListener('click',vomText);feld.addEventListener('keyup',e=>{if(e.key.startsWith('Arrow'))vomText()});
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter'){document.querySelector('button[value=speichern]').click()}});
"""


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
                + f"<style>{_STYLE}</style><header><b>Training</b><span class='sp'></span>"
                f"<a class='btn' href='/'>← zurück zum Korrigieren</a></header><section>"
                + (f"<p><b>{html.escape(message)}</b></p>" if message else "")
                + f"<p>{done} von {len(self.pages)} Seiten korrigiert: "
                + ", ".join(f"{t} {n}" for t, n in by_type.items()) + f". {hint}</p>"
                f"<form method='post' action='/training'><div class='typ'>Modell: {radios}</div>"
                f"<p><button class='primary'{' disabled' if running else ''}>Exportieren und trainieren</button></p></form>"
                f"<p>Der Knopf schreibt die korrigierten Seiten nach <code>{html.escape(str(self.target))}</code> "
                f"(Prüfsatz daneben, wird nie trainiert) und startet einen neuen nummerierten Lauf. Weiter "
                f"korrigieren geht währenddessen. Es läuft immer nur ein Training.</p><p>{status}</p>"
                f"<h3>Läufe</h3><pre>{html.escape(runs or '–')}</pre>"
                f"<h3>Protokoll (Ende)</h3><pre>{html.escape(tail or '–')}</pre></section>")

    def _training_link(self) -> str:
        return "<a class='btn primary' href='/training'>Mit Korrekturen trainieren</a>" if self.tune_py else ""

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
        states = [state(p) for p in self.pages]
        done = states.count("korrigiert")
        known = _side(page, ".tesseract.txt")
        radios = "".join(
            f"<label><input type='radio' name='typ' value='{t}' accesskey='{k}'{' checked' if t == page_type else ''}>"
            f" {t}</label>" for k, t in enumerate(TYPES, start=1))
        items = "".join(
            f"<a class='it{' cur' if k == i else ''}' data-s='{states[k]}' href='/seite/{k}' title='{html.escape(p.name)}'>"
            f"<span class='n'>{html.escape(p.stem)}</span><span class='t'>{current(p)[0] if has_text(p) else '–'}</span></a>"
            for k, p in enumerate(self.pages))
        counts = {name: states.count(name) for name in ("offen", "korrigiert", "verworfen")}
        options = "".join(f"<option value='{v}'>{label}</option>" for v, label in (
            ("offen", f"Offen ({counts['offen']})"), ("korrigiert", f"Korrigiert ({counts['korrigiert']})"),
            ("verworfen", f"Nicht verwendet ({counts['verworfen']})"), ("alle", f"Alle ({len(self.pages)})")))
        after = min(len(self.pages) - 1, i + 1)
        return (f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>{i + 1}/{len(self.pages)} {html.escape(page.name)}</title><style>{_STYLE}</style>"
                f"<header><b>Seiten korrigieren</b><span class='badge'>{done} von {len(self.pages)} korrigiert</span>"
                f"<span class='sp'></span>{self._training_link()}</header><main>"
                f"<nav><select id='filter'>{options}</select><div id='liste'>{items}</div></nav>"
                f"<div id='mitte'><div id='werkzeug'><button type='button' id='kleiner' title='verkleinern'>−</button>"
                f"<span id='stufe'></span><button type='button' id='groesser' title='vergrößern (auch Strg+Mausrad)'>+</button>"
                f"<button type='button' id='passen'>Einpassen</button><button type='button' id='voll'>100 %</button>"
                f"<button type='button' id='links' title='nach links drehen'>⟲</button>"
                f"<button type='button' id='rechts' title='nach rechts drehen'>⟳</button>"
                f"<span id='hinweis'></span></div>"
                f"<div id='bild'><div id='blatt'><div id='dreh'><img src='/bild/{i}' alt='' "
                f"data-name='{html.escape(page.name)}' data-nr='{i}'><div id='marke'></div></div></div></div></div>"
                f"<form class='seite' method='post' action='/seite/{i}'><div class='typ'><b>Typ</b>{radios}"
                f"<span class='st'>Seite {i + 1} · {states[i] if has_text(page) else 'noch kein Entwurf'} · Alt+1–4</span></div>"
                f"<textarea name='text' autofocus spellcheck='false'>{html.escape(text)}</textarea>"
                f"<div class='knoepfe'><button class='primary' name='aktion' value='speichern'>Speichern &amp; weiter "
                f"(Strg+Enter)</button><a class='btn' href='/seite/{after}'>Überspringen</a><span class='sp'></span>"
                f"<button class='gefahr' name='aktion' value='verwerfen' formnovalidate>Nicht verwenden</button></div>"
                f"<details><summary>Text bisher (Textebene/Tesseract)</summary><pre>"
                f"{html.escape(known.read_text(encoding='utf-8') if known.exists() else '–')}</pre></details>"
                f"<details><summary>Regeln</summary><pre>{html.escape(PROMPT)}</pre></details></form></main>"
                f"<script>{_SCRIPT}</script>")


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
                if parts[0] == "bloecke":
                    lines = _side(tool.pages[i], ".bloecke.json")
                    body = lines.read_bytes() if lines.exists() else b"[]"
                    return self._send(200, body, "application/json; charset=utf-8")
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
