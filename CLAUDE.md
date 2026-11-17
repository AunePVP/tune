# tune – ein feingetunter Seitenleser für Projektakte

Diese Datei ist der Einstieg für jede KI-Sitzung in diesem Repository. Sie enthält alles, was man zum Weiterarbeiten
braucht: worum es geht, was schon entschieden und gemessen ist, wie gearbeitet wird und was als Nächstes kommt.
Lies sie ganz, bevor du etwas änderst.

---

## 1. Worum es geht

**Projektakte** (Repository `aunepvp/projektakte`) ist ein internes Werkzeug eines Brandschutz-Ingenieurbüros. Es liest
die Projektordner vom Dateiserver und die Mails, ordnet alles einer Projektnummer zu, macht es durchsuchbar und
beantwortet Fragen mit Quellenangabe („Welche Auflagen zur Rauchableitung stehen in den Dokumenten?“). Es läuft
vollständig im Büro, ohne Cloud.

Viele Dokumente sind **Scans**: alte Bauakten (ab den 1950ern), Prüfberichte, Bescheide, Formulare mit Handschrift,
Pläne, Fotos. Projektakte liest sie in zwei Stufen:
1. **Stufe 1:** die Textebene der PDF, falls vorhanden. Tesseract ist beim Nutzer ausgeschaltet (`TESSERACT=0`).
2. **Stufe 2, „Bildauswertung“:** Jede Scan-Seite geht als Bild an ein Sprachmodell mit Bildverständnis. Dessen Antwort
   wird der Text der Seite im Suchindex.

**Dieses Repository** baut das Modell für Stufe 2: einen **feingetunten Seitenleser**, der jede Seite so liest, wie
das Büro es festgelegt hat (Abschnitt 3). Rund 160.000 Seiten warten darauf. Die Indexierung in Projektakte **ruht**,
bis das Modell steht.

## 2. Das Problem, kurz

Getestet im Betrieb und auf 49 Prüfseiten, mit Qwen 3.5 4B, GLM-OCR, PaddleOCR-VL und TeleOCR:

- **Erfundenes:** Auf einem Brandschutzplan „las“ ein Modell Räume wie „Backerei“ oder „Firingblock“, die es nicht
  gibt. Auf einer Berichtsseite erfand Qwen rund 20 „[Stempel: …]“, die den Inhalt zusammenfassten.
- **Schleifen:** Auf einem Datenblatt, dessen Seitenbild nur ein Produktfoto zeigt, schrieb ein OCR-Modell eine
  erfundene, hochzählende Liste von Fragen („1. What is the purpose of the image? …“) bis zur Token-Grenze.
- **Rauschen:** Pläne ergeben Hunderte Fetzen (Maße, Flurstücke, Achsen), Fotos eine Bildbeschreibung („Es gibt keinen
  Text auf diesem Bild …“). Nichts davon gehört in den Index.
- **Eine Textebene beweist nichts:** Bei Plänen lag manchmal Text darunter, der auf der Seite nicht zu sehen ist.
- **Tempo:** Im Betrieb schrieb Qwen rund 1.150 Antwort-Tokens je Seite, der DGX Spark schaffte so rund 800 Seiten pro
  Stunde.

**Kern:** Eine einzige Anweisung „lies die Seite“ für alle Seitenarten führt bei kleinen Modellen zu Erfundenem,
Schleifen und Rauschen. Spezialisierte OCR-Modelle lesen sauber, befolgen aber keine Regeln je Seitentyp. Allgemeine
kleine Modelle verstehen die Regeln, halten sich aber nicht zuverlässig daran. Deshalb: **Feintuning auf das
gewünschte Verhalten.**

## 3. Die Vorgabe: was auf eine Seite gehört

Festgelegt mit dem Nutzer nach der Frage: *Was würde ein Mensch hinschreiben, der alle Dokumente von Hand indexiert?*
Der Index soll Dokumente **finden** und Fragen **mit Beleg** beantworten.

**Grundregeln:** nur was **sichtbar** auf der Seite steht. Nichts erfinden, nichts beschreiben, nichts zusammenfassen.
Unleserliches als `[unleserlich]`. Steht nichts Lesbares da, bleibt die Seite leer.

| Seitentyp | Hinschreiben | Nicht hinschreiben |
|---|---|---|
| **Text** (Brief, Bericht, Bescheid, Prüfbericht, Gutachten, Formular, Datenblatt) | den vollständigen Text wörtlich in Lesereihenfolge; Tabellen als Markdown-Tabellen; Ankreuzfelder als ☒/☐ mit Beschriftung | – |
| **Kopf- und Fußzeilen** | immer mitschreiben – Projektakte entfernt Zeilen, die auf fast jeder Seite eines Dokuments gleich wiederkehren (das Modell sieht nur eine Seite und weiß nicht, ob es die erste ist) | – |
| **Fotos in Textseiten** (Fotodokumentation, Datenblatt mit Bild) | Die Seite ist `Text`. Text im Foto nur, wenn er **klar lesbar** ist, an der Stelle des Fotos (bei der Bildunterschrift) | unscharfen Fototext: nichts, auch kein `[unleserlich]` (sonst Dutzende Marker ohne Nutzen, und kleine Modelle raten dort) |
| **Stempel** | jeden Stempel mit Vermerk, Datum, Aktenzeichen oder Prüf-/Listennummer (Genehmigung, Prüfvermerk, Eingang, Prüfingenieur, Verfasser mit Listennummer), **genau einmal**, wörtlich als `[Stempel: …]`; nicht lesbare Teile darin als `[unleserlich]`, z. B. `[Stempel: Genehmigt [unleserlich] 2005]` | reine Namens- und Adressstempel; nie zusammenfassen, nie erfinden |
| **Handschrift** | Vermerke („erledigt“, „i. O.“, Korrekturen, Daten) an ihrer Stelle als `[handschriftlich: …]`; Unterschriften als `[Unterschrift]` | – |
| **Logos** | nichts | – |
| **Plan** (Grundriss, Schnitt, Lageplan, Brandschutz- und Fluchtplan) | **Schriftfeld** (Planart, Objekt/Adresse, Geschoss/Bereich, Plan-Nr., Index, Datum, Maßstab, Verfasser), **Legende**, **Hinweis-Textblöcke** (zusammenhängender Text, mindestens ein ganzer Satz oder eine mehrzeilige Notiz), Stempel, **handschriftliche Eintragungen mit Worten** (Grüneintragungen der Behörde, Prüfvermerke) als `[handschriftlich: …]`; ist eine Eintragung erkennbar da, aber nicht lesbar, einmal `[handschriftlich: unleserlich]` (die Information „hier hat jemand etwas eingetragen“ ist selbst wertvoll: Original ansehen) | Raumnamen, Maße, Achsen, Flurstücke, einzelne Wörter oder Beschriftungen in der Zeichnung („Flur“, „Büro 1.04“, „T30“ an einer Tür), auch wenn sie fachlich klingen; reine Zahlen- und Maßkorrekturen |
| **Foto** (auch Produktfoto) | nur klar lesbaren Text, der im Foto zu sehen ist (Typenschilder, Zulassungsnummern, Beschriftungen); ohne solchen Text bleibt der Inhalt leer | eine Beschreibung des Fotos |
| **Leer** (unbedruckte Seite, reine Grafik, Logo, Hintergrund) | nichts | – |

**Typ nach Seitenart:** Ein Foto bleibt `TYP: Foto` und ein Plan `TYP: Plan`, auch wenn danach nichts zu schreiben ist.
`TYP: Leer` gilt nur für unbedruckte Seiten und reine Grafik. Eine Seite mit Fließtext oder Bildunterschriften ist
immer `Text`, auch mit eingebetteten Fotos.

**Für die Trainingsdaten:** Die meisten Seiten müssen **ohne Stempel** sein, sonst lernt das Modell, Stempel zu
erwarten und zu erfinden. Mehrere Stempel auf einer Seite: jeden genau einmal.

**Weitere Regeln (vom Nutzer bestätigt):** `[unleserlich]` je unleserlicher Stelle einmal (egal ob Wort
oder Absatz, nie mehrere hintereinander); auf einer reinen Fotoseite gilt für unscharfen Text dasselbe wie bei
eingebetteten Fotos (nichts schreiben); ganz handgeschriebene Seiten sind `Text` in einem `[handschriftlich: …]`;
gedrehte Seiten und Doppelseiten werden in normaler Lesereihenfolge gelesen (links vor rechts).

**Antwortform:** erste Zeile `TYP: Text`, `TYP: Plan`, `TYP: Foto` oder `TYP: Leer`, danach der Inhalt (bei `Leer`
nichts). Wozu: Die Suche kann nach Typ filtern, Projektakte erkennt Fehleinordnungen, `TYP: Leer` heißt sicher
„bewusst nichts“ und nicht „etwas ging schief“.

Die Anweisung an das Modell steht **wörtlich und nur einmal** in `scripts/ocr_vorgabe.py` (`PROMPT`, `TYPES`, `split`,
`join`). Entwürfe, Korrektur, Training, Prüfung und später der Betrieb in Projektakte nutzen genau diese Fassung. Wer sie
ändert, ändert die Vorgabe für alles. Dann müssen Trainingsdaten und Prüfung dazu passen, und Projektakte muss sie beim
Einbau übernehmen.

## 4. Was gemessen ist

Prüfsatz: 49 wartende Scan-Seiten aus vielen Projekten (ohne CAD- und Bilder-Ordner), 8 Anfragen gleichzeitig, schlichte
Anweisung „Seite vollständig und wörtlich abschreiben“ (nicht die Vorgabe aus Abschnitt 3):

| Modell | Hardware | Ø Antwort-Tokens | Schleifen | abgeschnitten | Bemerkung |
|---|---|---|---|---|---|
| Qwen 3.5 4B, Denken aus | DGX Spark (vLLM) | 583 | 2 | 1 | befolgt Anweisungen; beschreibt Bilder ohne Text |
| GLM-OCR (0,9B) | RTX 4080 (vLLM) | 558 | 0 | 3 | sauberer Leser; erfindet bei Bildern ohne Text und auf Plänen |
| PaddleOCR-VL 1.6 | RTX 4080 (vLLM) | 1.224 | 10 | 11 | ohne seine Layout-Erkennung unbrauchbar |
| TeleOCR | RTX 4080 | – | – | – | vLLM: Startfehler (`mrope_section`); SGLang: nur Zeichensalat |

Weitere Erkenntnisse:
- **Qwen mit Denken** brauchte 2.628 Tokens für 646 Zeichen, 21 von 49 Seiten abgeschnitten. Immer
  `{"chat_template_kwargs":{"enable_thinking":false}}` mitschicken (im Vergleichsskript: `--extra-body`).
- Bei gleicher schlichter Anweisung schreiben Qwen und GLM-OCR **gleich viel**. Der Unterschied liegt im Verhalten,
  nicht in der Antwortlänge.
- Seitenbilder gehen mit **150 dpi** der echten Seitengröße an das Modell (A4 ≈ 2,2 MP ≈ 2.600 Bild-Tokens). Eine
  frühere Messung des Nutzers: 150 dpi liest so viel wie 300 dpi bei einem Viertel der Tokens, 100 dpi schlechter.
- `presence_penalty` kostet Text (1,5: rund −14 Prozentpunkte gefundene Wörter) und verhindert Schleifen nicht
  zuverlässig. vLLM `repetition_detection` half im Betrieb.
- Auf dem Spark waren 22–24 gleichzeitige Anfragen am schnellsten. Begrenzt hat dort das Schreiben der Antworten
  (~260 Antwort-Tokens/s insgesamt), nicht das Einlesen.

## 5. Entschieden

- **Feintuning** eines Modells auf die Vorgabe aus Abschnitt 3. Ein Aufruf je Seite, keine eigene Klassifizierung davor.
  Das Modell bestimmt den Typ selbst und schreibt ihn in die erste Zeile.
- **Ein kleines Modell, kein größeres ohne Training** (Entscheidung des Nutzers, nicht erneut aufrollen): Der
  Seitenleser muss schnell sein. Das große Frage-Modell auf dem Spark steht zur Verfügung, wird aus Tempogründen aber
  nicht für die Bildauswertung genutzt. Kleine Modelle halten die Vorgabe ohne Training nicht ein. Der Vergleich ohne
  Training (Abschnitt 8.1) dient nur der Wahl des Grundmodells.
- **Devise: einmal richtig machen.** Sorgfalt bei Vorgabe, Trainingsdaten und Prüfung geht vor Tempo im Projekt.
- **Der Nutzer will das Feintuning selbst durchführen und verstehen.** Das Trainingsrezept wird deshalb als
  nachvollziehbare Anleitung geschrieben (was jeder Schritt und jede Einstellung bewirkt), nicht als Blackbox.
- **Pläne:** Schriftfeld, Legende, Hinweis-Textblöcke, Stempel und handschriftliche Eintragungen mit Worten
  (Abschnitt 3; der Nutzer hat sich dafür entschieden).
- **Prüfsatz als beiseitegelegter Teil der korrigierten Seiten:** Der Prüfsatz muss nicht belegen, dass es heute nicht
  funktioniert (das ist bekannt). Er misst **nach** dem Training: Abnahme vor dem Lauf über 160.000 Seiten, Vergleich
  von Trainingsläufen und Grundmodellen, Erkennen von Auswendiglernen. Er ist kein eigener Arbeitsgang: Von den
  ohnehin korrigierten Seiten werden rund 150–200 beiseitegelegt, immer **ganze Dokumente**, in der Mischung des
  Trainings (Scans, CAD-Pläne, digitale Seiten, Fotos). Sie werden nie trainiert. Die bisherigen 49 Seiten können
  darin aufgehen.
- **Training auf dem DGX Spark** (128 GB gemeinsamer Speicher, volle Genauigkeit BF16, die Daten liegen dort). Die RTX
  4080 (16 GB, Windows) dient für Entwürfe, Korrektur und schnelle Versuche, für ein 4B-Modell nur mit QLoRA.
- **CAD-Pläne kommen in die Trainingsdaten**, obwohl Projektakte sie im Betrieb gerade auslässt
  (`VISION_SKIP_CAD=1`, `SKIP_READ_FOLDERS=CAD`). Liest das Modell Pläne gut, können sie wieder dazu.
- **Verfahren:** LoRA (nur ein kleiner Zusatz wird trainiert).

## 6. Offen

1. **Grundmodell.** Sicherer Start ist Qwen: läuft schon, versteht Anweisungen, von Trainingswerkzeugen am besten
   unterstützt. Vor dem Training 2–3 Kandidaten **ohne Training** mit `PROMPT` auf den 49 Seiten sichten (ein kurzer
   Blick genügt, kein aufwendiger Maßstab) und dabei nach dem aktuellen Stand kleiner Bild-Sprachmodelle (2–4B, freie
   Lizenz, gute Unterstützung in vLLM und in den Trainingswerkzeugen) suchen. Die Trainingsdaten passen zu jedem
   Grundmodell; die endgültige Wahl fällt notfalls durch Training zweier Kandidaten und Messung am Prüfsatz.
2. **Trainingswerkzeug auf dem Spark** (ARM64 + CUDA): LLaMA-Factory, ms-swift oder Unsloth – was dort zuverlässig läuft.
   Am sichersten ist ein NVIDIA-PyTorch-Container. Erst prüfen, dann festlegen.
3. **Menge der Trainingsdaten:** Start ab ~200 korrigierten Seiten, Ziel 300–800. 10–20 % Fotos/Leeres und ebenso viele
   Pläne, sonst lernt das Modell „bei nichts nichts schreiben“ nicht.
4. Ob CAD-Pläne im Betrieb wieder dazukommen (nach der Prüfung).
5. Ob Tesseract in Projektakte wieder eingeschaltet wird – mit einem guten Seitenleser eher nicht nötig.
6. **Bildgröße großer Pläne:** A0 mit 150 dpi wären rund 35 MP. Prüfen, was der Export in Projektakte daraus macht.
   Wird stark verkleinert, sind Schriftfeld und Hinweise nicht lesbar; dann braucht es einen Ausschnitt oder Kacheln.
7. **Abnahmekriterium:** ab welchen Werten das Modell als fertig gilt. Vorschlag der KI, vom Nutzer noch nicht
   entschieden: keine Schleife, kein erfundener Stempel, kein Inhalt auf leeren Seiten und textlosen Fotos, Typ auf
   mindestens 95 % der Seiten richtig; Schwelle für die Zeichen-Fehlerrate nach der ersten Messung.
8. **Auslassungen in den Soll-Texten:** Aus einem einzigen Qwen-Entwurf übersieht man beim Korrigieren leicht, was
   fehlt. Vorschlag der KI, noch nicht entschieden: für die Prüfseiten Entwürfe zweier Modelle (bei digitalen Seiten
   zusätzlich die Textebene) und Abweichungen im Korrekturwerkzeug markieren.
## 7. Ablauf und Werkzeuge

| Schritt | Werkzeug | Wo | Stand |
|---|---|---|---|
| 1. Prüfseiten exportieren | Projektakte: `python -m projektakte.ocr_testseiten --anzahl 50 --ziel /data/ocr-test-2` | Spark | erledigt (49 Seiten) |
| 2. Trainingsseiten exportieren | Projektakte: `python -m projektakte.ocr_testseiten --training --anzahl 600 --ziel /data/training --ohne-seiten /data/ocr-test-2/seiten.json` | Spark | gebaut |
| 3. Modelle vergleichen | `scripts/ocr_vergleich.py` | PC oder Spark | gebaut |
| 4. Entwürfe schreiben lassen | `scripts/ocr_entwurf.py` | Spark (Qwen) | gebaut |
| 5. Korrigieren | `scripts/ocr_korrektur.py` (Browser, 127.0.0.1:8765) | PC | gebaut |
| 6. Trainingsdaten und Prüfsatz ausgeben | `scripts/ocr_korrektur.py --export training.jsonl` (schreibt auch `training.pruef.jsonl`) | PC | gebaut |
| 7. Training | Rezept je Werkzeug | Spark | **zu bauen** |
| 8. Prüfen gegen Soll-Texte | Erweiterung von `ocr_vergleich.py` | PC/Spark | **zu bauen** |
| 9. Einbau in Projektakte | im Repository `projektakte` | – | **zu bauen** |

Der **Export** (Schritt 1 und 2) bleibt in Projektakte, weil er die Datenbank, den abgesicherten Leseprozess und genau
die Bildaufbereitung der App braucht. Er schreibt je Seite `NNN_<Datei>_S<Seite>.jpg|png`, `….tesseract.txt` (der Text,
den Projektakte schon hat) und eine `seiten.json` (`datei`, `dokument`, `seite`, `mime`, `tesseract_zeichen`, bei
`--training` auch `quelle`: scan, cad, digital, foto). Mischung beim Training: 55 % wartende Scans, 15 % CAD-Pläne,
15 % digitale PDF-Seiten, 15 % Fotos, je Dokument höchstens zwei Seiten.

### Befehle (Spark: Ordner `~/projektakte/deploy`, Python heißt dort `python3`; PC: Windows/PowerShell, `python`)

Trainingsseiten exportieren und aus dem Container holen (Spark):
```bash
docker compose exec app python -m projektakte.ocr_testseiten --training --anzahl 600 --ziel /data/training --ohne-seiten /data/ocr-test-2/seiten.json
docker compose cp app:/data/training ./training
```

Entwürfe mit Qwen (Spark; Qwen läuft dort als Container `qwen35-4b`, vLLM, Host-Netz, Port 8003; fortsetzbar):
```bash
python3 ~/tune/scripts/ocr_entwurf.py --ordner training --modell "Qwen|http://localhost:8003/v1|qwen35-4b" --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' --parallel 8
```

Korrigieren (PC; Ordner `training` und `scripts/` aus diesem Repository dorthin kopieren):
```powershell
python scripts\ocr_korrektur.py --ordner training
```
Dann `http://127.0.0.1:8765`: links das Bild, rechts der Entwurf. Typ wählen (Alt+1–4), „Speichern & weiter“
(Strg+Enter), „nicht verwenden“ für unbrauchbare Seiten. Ergebnis: `<Seite>.soll.txt`.

Trainingsdaten und Prüfsatz ausgeben:
```powershell
python scripts\ocr_korrektur.py --ordner training --export training.jsonl --bild-basis /home/md/training
```
Das schreibt `training.jsonl` und `training.pruef.jsonl`. Eine Zeile je Seite:
`{"messages":[{"role":"user","content":"<image>…PROMPT…"},{"role":"assistant","content":"TYP: …"}],"images":["<Pfad>"]}`.
Das ist das Format, das LLaMA-Factory und ms-swift lesen.
- **Aufteilung:** Ganze Dokumente (Feld `dokument` der `seiten.json`) gehen in den Prüfsatz, Voreinstellung 20 %
  (`--pruef-anteil 0.2`). Entschieden wird über einen festen Hash des Dokumentnamens: Die Aufteilung ist bei jedem
  Lauf und auf jedem Rechner gleich, auch wenn später Seiten dazukommen. Ein größerer Anteil nimmt nur Dokumente
  dazu. **Den Anteil nie verkleinern**, sonst wandern Prüfseiten ins Training. `training.pruef.jsonl` nie trainieren.
- **Bildpfade:** `--bild-basis` ist der Ordner der Seitenbilder auf dem Trainingsrechner. Ohne die Option steht der
  absolute Pfad des Rechners darin, auf dem exportiert wurde.
- **Die 49 alten Prüfseiten:** eigener Ordner, dort mit `--pruef-anteil 1` ausgeben (alles Prüfsatz). Beim Export des
  Trainingsordners `--ohne-dokumente <Pfad>\ocr-test-2\seiten.json` angeben, dann fehlen Seiten aus deren Dokumenten
  in beiden Dateien.

Ein Modell vergleichen (Beispiel GLM-OCR auf dem PC mit Docker Desktop; immer nur **ein** Modell auf der Grafikkarte):
```powershell
docker run --rm --gpus all -p 8000:8000 -v "$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface" --entrypoint vllm vllm/vllm-openai:latest serve zai-org/GLM-OCR --trust-remote-code --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85
python scripts\ocr_vergleich.py --ordner ocr-test-2 --modell "GLM-OCR|http://localhost:8000/v1|zai-org/GLM-OCR|Text Recognition:" --parallel 8 --ausgabe ergebnis-glm
```
Die Übersicht (Seiten/Min., Ø Tokens, Schleifen, abgeschnitten) enthält keinen Seitentext und darf weitergegeben werden.
`ergebnis-…/vergleich.html` zeigt Bild, bisherigen Text und Modelltext nebeneinander. Seiten/Min. zwischen PC und Spark
sind nicht vergleichbar, Antwort-Tokens je Seite schon.

## 8. Nächste Aufgaben, der Reihe nach

1. **Grundmodell-Kandidaten sichten** (Abschnitt 6.1): aktuelle Kandidaten recherchieren und die 2–3 besten **ohne
   Training** mit `PROMPT` auf den 49 Seiten laufen lassen. Dafür `ocr_vergleich.py` um eine Option erweitern, die
   `PROMPT` aus `ocr_vorgabe.py` nimmt und die `TYP:`-Zeile auswertet (Typ-Verteilung, fehlende Typzeile). Ein kurzer
   Blick genügt: hält es sich grob an die Vorgabe, schleift es, läuft es in vLLM und im Trainingswerkzeug.
2. **Seiten exportieren, Entwürfe, Korrektur** durch den Nutzer: Trainings- und Prüfseiten in einem Zug, rund 800
   Seiten (sein Aufwand, grob 15–30 Stunden verteilt). Entwürfe und vorhandene Soll-Texte, die vor der Änderung der
   Vorgabe entstanden sind, passen nicht mehr und müssen neu erzeugt bzw. durchgesehen werden.
3. **Prüfsatz beiseitelegen:** Der Export (`ocr_korrektur.py --export`, gebaut) legt 20 % der Dokumente fest als
   Prüfsatz in `<Export>.pruef.jsonl` ab, bei 800 Seiten rund 160. Diese Seiten werden nie trainiert. Sie sind der
   Maßstab. Nach dem Export prüfen, ob jeder Seitentyp im Prüfsatz ausreichend vertreten ist (die Ausgabe zählt je Typ).
4. **Prüfskript gegen Soll-Texte:** je Seite Typ richtig/falsch, Zeichen-Fehlerrate (CER) gegen den Soll-Text,
   fehlende und erfundene Zeilen, erfundene Stempel, Schleifen, Antwort-Tokens, getrennt je Seitentyp. Ausgabe als
   Übersicht ohne Seitentext und als HTML zum Durchsehen.
5. **Trainingsrezept** auf dem Spark: Container, Werkzeug, LoRA-Einstellungen, Bildgröße wie im Betrieb (150 dpi),
   Validierung. Erst ein kleiner Lauf mit ~200 Seiten, dann prüfen (Schritt 4), dann mehr Daten.
6. **Übergabe an Projektakte:** fertiges Modell (LoRA zusammengeführt oder als Adapter in vLLM), Startbefehl auf dem
   Spark mit begrenztem Speicher (das Modell läuft neben einem großen Frage-Modell und dem Embedding-Dienst, eher
   8–12 GB), Beschreibung für den Einbau. Der Einbau selbst geschieht im Repository `projektakte`: eigene Einstellungen
   `VISION_URL`/`VISION_MODEL`, `PROMPT` aus `ocr_vorgabe.py`, `TYP:` auswerten und speichern, wiederkehrende Kopf- und
   Fußzeilen je Dokument entfernen, Schleifenerkennung beibehalten.

## 9. Regeln für dieses Repository

- **Keine echten Daten im Repository.** Seitenbilder, Entwürfe, Soll-Texte, `seiten.json`, `*.jsonl`, Ergebnisse und
  Modelldateien sind echte Projektdaten bzw. große Dateien. Sie bleiben auf dem Spark, dem PC oder einer Freigabe im
  Büro. `.gitignore` schließt sie aus. Vor jedem Commit `git status` prüfen. Auch Projekt- und Dateinamen aus den
  Akten gehören nicht in Code, Tests oder Doku (Beispiele erfinden).
- **Die Prüfseiten nie trainieren**, und keine anderen Seiten aus deren Dokumenten. Die Aufteilung nach Dokument und
  getrennte Ordner sichern das.
- **Keine Cloud.** Kein Hochladen von Seiten, keine Cloud-Modelle, keine Telemetrie. Modelle werden heruntergeladen und
  laufen lokal.
- **Lieber nichts als falsch:** Bei Zweifel ist eine leere oder kürzere Antwort besser als eine erfundene. Das gilt für
  die Vorgabe, die Trainingsdaten und die Bewertung.
- **Sprache:** Alles für den Nutzer Sichtbare auf Deutsch (Ausgaben, Meldungen, Doku). Code-Kommentare auf Englisch.
- **Werkzeuge nur mit der Standardbibliothek**, damit sie ohne Installation auf dem Windows-PC und dem Spark laufen.
  Trainingscode darf mehr brauchen, aber in eigenem Ordner mit eigener Anleitung.
- **Tests:** `python -m pytest -q` (nur `pytest` nötig) muss grün sein.
- **Ein Commit je Thema**, deutsche Commit-Titel. Diese Datei aktuell halten: Stand, Messungen, Entscheidungen, offene
  Punkte.

## 10. Umgebung

- **DGX Spark** (Büronetz, Hostname `spark-0eac`, Benutzer `md`): NVIDIA GB10, 128 GB gemeinsamer Speicher, ARM64.
  Läuft dort: Projektakte (Docker Compose in `~/projektakte/deploy`, Container `projektakte-app-1`, Daten in
  `/data` im Container), `qwen35-4b` (vLLM, Port 8003), ein großes Frage-Modell (Qwen 3.8 Flash Next), Embeddings
  (llama.cpp, Port 8001). Modelle liegen im Hugging-Face-Cache `~/.cache/huggingface`.
- **PC des Nutzers:** Windows, RTX 4080 (16 GB), Docker Desktop mit WSL2, Python. Modelle in
  `%USERPROFILE%\.cache\huggingface`. Arbeitsordner für Tests: `C:\ocr-test-lauf`.
- Die KI-Dienste sind nur im Büronetz erreichbar. Eine Cloud-Sitzung (wie die, in der dieses Repository angelegt wurde)
  erreicht sie nicht. Messungen macht der Nutzer und schickt die Übersichten.

## 11. Dateien

| Datei | Zweck |
|---|---|
| `scripts/ocr_vorgabe.py` | die Vorgabe: `PROMPT`, `TYPES`, `split`/`join` der Antwortform |
| `scripts/ocr_vergleich.py` | Modelle auf einem Seitenordner vergleichen (OpenAI-kompatible Server; Tempo, Tokens, Schleifen; HTML nebeneinander; `--extra-body`) |
| `scripts/ocr_entwurf.py` | Entwürfe nach der Vorgabe erzeugen (`<Seite>.entwurf.txt`, fortsetzbar) |
| `scripts/ocr_korrektur.py` | Korrigieren im Browser (`<Seite>.soll.txt`), Export als JSONL, getrennt in Training und Prüfsatz nach Dokument (`--pruef-anteil`, `--ohne-dokumente`, `--bild-basis`) |
| `tests/` | Tests der Werkzeuge (ohne echte Daten) |
| Projektakte: `projektakte/ocr_testseiten.py` | Seiten exportieren (Prüfsatz, `--training`) |
| Projektakte: `DECISIONS.md` „Was auf eine Seite gehört“ und frühere Abschnitte zur Bildauswertung | Hintergrund |
