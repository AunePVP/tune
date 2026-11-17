# tune – ein feingetunter Seitenleser für Projektakte

Diese Datei ist der Einstieg für jede KI-Sitzung in diesem Repository. Sie enthält alles, was man zum Weiterarbeiten
braucht: worum es geht, was schon entschieden und gemessen ist, wie gearbeitet wird und was als Nächstes kommt.
Lies sie ganz, bevor du etwas änderst.

---

## Stand (17.11.2026)

- **Fertig:** die Vorgabe (Abschnitt 3, mit dem Nutzer geschärft), die Werkzeuge für Vergleich, Entwürfe, Korrektur
  und Export mit festem Prüfsatz (Abschnitt 7), die Wahl der Trainingskandidaten (Abschnitt 5).
- **Als Nächstes, in eigenen Sitzungen:** das Trainingswerkzeug bauen (Abschnitt 8.1), das Prüfskript bauen
  (Abschnitt 8.2). Parallel exportiert und korrigiert der Nutzer die Seiten (Abschnitt 8.3).
- **Rollen:** Der Nutzer trainiert selbst. Die KI baut und prüft die Werkzeuge und Anleitungen dafür; sie startet
  kein echtes Training, höchstens einen kurzen Funktionstest mit erfundenen Seiten.

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
| GLM-OCR (1,33B) | RTX 4080 (vLLM) | 558 | 0 | 3 | sauberer Leser; erfindet bei Bildern ohne Text und auf Plänen |
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
- **Nicht festgehalten:** mit welcher Fassung von Qwen 3.5 4B gemessen wurde (auf dem Spark liegen nur quantisierte,
  Abschnitt 10) und warum Qwen im Betrieb rund 1.150 Antwort-Tokens je Seite schrieb, auf den 49 Seiten aber 583
  (vermutlich andere Anweisung und andere Seitenmischung).
- Auf dem Spark waren 22–24 gleichzeitige Anfragen am schnellsten. Begrenzt hat dort das Schreiben der Antworten
  (~260 Antwort-Tokens/s insgesamt), nicht das Einlesen.

## 5. Entschieden

- **Feintuning** eines Modells auf die Vorgabe aus Abschnitt 3. Ein Aufruf je Seite, keine eigene Klassifizierung davor.
  Das Modell bestimmt den Typ selbst und schreibt ihn in die erste Zeile.
- **Ein kleines Modell, kein größeres ohne Training** (Entscheidung des Nutzers, nicht erneut aufrollen): Der
  Seitenleser muss schnell sein. Das große Frage-Modell auf dem Spark steht zur Verfügung, wird aus Tempogründen aber
  nicht für die Bildauswertung genutzt. Kleine Modelle halten die Vorgabe ohne Training nicht ein.
- **Devise: einmal richtig machen.** Sorgfalt bei Vorgabe, Trainingsdaten und Prüfung geht vor Tempo im Projekt.
- **Der Nutzer trainiert selbst und will es verstehen.** Die KI baut dafür ein Trainingswerkzeug mit Anleitung
  (was jeder Schritt und jede Einstellung bewirkt), keine Blackbox, und startet selbst kein echtes Training.
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
- **Trainingskandidaten** (Entscheidung des Nutzers, 17.11.2026): **`Qwen/Qwen3.5-2B`** und **`zai-org/GLM-OCR`**,
  beide mit denselben Daten trainiert und am Prüfsatz verglichen. `Qwen/Qwen3.5-4B` ist die Rückfallebene, falls 2B
  nicht reicht. Weitere Kandidaten (Xiaomi-OCR-0, LightOnOCR-3, Gemma 4) sind gesichtet und zurückgestellt
  (Abschnitt 12). Welcher der beiden in den Betrieb geht, entscheidet die Messung am Prüfsatz.
- **Trainingswerkzeug: LLaMA-Factory** (vorbehaltlich des Funktionstests auf dem Spark, Abschnitt 8.1). Es deckt
  beide Kandidaten ab; Ausweichwege sind ms-swift und Unsloth.
- **Betrieb mit niedrigerer Genauigkeit** (Wunsch des Nutzers: Tempo, weniger Speicher). Trainiert wird auf dem
  Original in voller Genauigkeit (BF16), weil die Trainingswerkzeuge auf den Betriebsformaten FP8/NVFP4 nicht
  trainieren können; das betrifft nur die Stunden des Trainings. Danach: Adapter zusammenführen, quantisieren
  (FP8 oder NVFP4, z. B. mit llm-compressor), in vLLM starten. **Abgenommen wird die quantisierte Fassung**, also die,
  die im Betrieb läuft: Prüfsatz mit BF16, FP8 und NVFP4 messen. Laut Berichten hält FP8 bei Bild-Sprachmodellen
  über 99 % der Genauigkeit, NVFP4 kostet gerade bei kleinen Modellen spürbar mehr. Ein kleineres Modell bringt mehr
  Tempo als eine niedrigere Genauigkeit.
- **Später weitertrainieren können** (Wunsch des Nutzers). Das Trainingsrezept bietet zwei Wege:
  1. **Neu trainieren mit allen Daten** (Regelfall): sauberster Weg, bei einigen hundert Seiten kurz.
  2. **Auf dem vorhandenen Adapter weitertrainieren** mit neuen Seiten: schneller, aber das Modell kann Älteres
     verlernen. Deshalb immer einen Teil der alten Seiten beimischen und kleiner lernen lassen.

  Was dafür gilt:
  - **Jeden Lauf aufbewahren:** Adapter (nicht nur das zusammengeführte Modell), Checkpoints, Einstellungen und
    Datenstand (welche JSONL, Seitenzahl, Fassung von `PROMPT`), je Lauf ein nummerierter Ordner.
  - **Neue Seiten:** weiterer Export aus Projektakte in einen eigenen Ordner (`training-2`, mit `--ohne-seiten` der
    bisherigen), Entwürfe, Korrektur, Export. Die JSONL-Dateien werden aneinandergehängt. Der Hash je Dokument hält
    die Aufteilung in Training und Prüfsatz über alle Ordner gleich.
  - **Derselbe Prüfsatz** für alle Läufe (er darf wachsen), sonst sind sie nicht vergleichbar.
  - **Ändert sich `PROMPT`:** alle Soll-Texte anpassen und neu trainieren, nicht weitertrainieren.

## 6. Offen

1. **Abnahmekriterium:** ab welchen Werten das Modell als fertig gilt. Vorschlag der KI, vom Nutzer noch nicht
   entschieden: keine Schleife, kein erfundener Stempel, kein Inhalt auf leeren Seiten, Typ auf mindestens 95 % der
   Seiten richtig; Schwelle für die Zeichen-Fehlerrate nach der ersten Messung.
2. **Auslassungen in den Soll-Texten:** Aus einem einzigen Qwen-Entwurf übersieht man beim Korrigieren leicht, was
   fehlt. Vorschlag der KI, noch nicht entschieden: für die Prüfseiten einen zweiten Entwurf eines anderen Modells
   (GLM-OCR oder Nemotron 3.5 Parse, bei digitalen Seiten zusätzlich die Textebene) und Abweichungen im
   Korrekturwerkzeug markieren.
3. **Menge der Trainingsdaten:** Start ab ~200 korrigierten Trainingsseiten, Ziel 300–800, dazu der Prüfsatz (20 % der
   Dokumente). 10–20 % Fotos/Leeres und ebenso viele Pläne, sonst lernt das Modell „bei nichts nichts schreiben“
   nicht. Ob der Export genug wirklich leere Seiten liefert, ist offen (der Bilder-Ordner liefert Fotos).
4. **Digitale PDF-Seiten im Training (15 %):** Sie gehen im Betrieb nie an den Seitenleser. Offen ist, ob sie bleiben
   (als leicht zu korrigierende Textseiten mit Textebene als Gegenprobe) oder durch Scans ersetzt werden.
5. **Speicher beim Training:** ob das Training neben dem großen Frage-Modell läuft oder dieses dafür angehalten
   wird (Abschnitt 10). Erst messen, dann entscheidet der Nutzer.
6. **Quantisierung für den Betrieb:** FP8 oder NVFP4, und mit welchem Werkzeug (llm-compressor ist recherchiert, auf
   dem Spark nicht ausprobiert). Entscheidet die Messung am Prüfsatz.
7. Ob CAD-Pläne im Betrieb wieder dazukommen (nach der Prüfung).
8. Ob Tesseract in Projektakte wieder eingeschaltet wird – mit einem guten Seitenleser eher nicht nötig.
9. **Bildgröße großer Pläne:** A0 mit 150 dpi wären rund 35 MP. Prüfen, was der Export in Projektakte daraus macht.
   Wird stark verkleinert, sind Schriftfeld und Hinweise nicht lesbar; dann braucht es einen Ausschnitt oder Kacheln.

## 7. Ablauf und Werkzeuge

| Schritt | Werkzeug | Wo | Stand |
|---|---|---|---|
| 1. Prüfseiten exportieren | Projektakte: `python -m projektakte.ocr_testseiten --anzahl 50 --ziel /data/ocr-test-2` | Spark | erledigt (49 Seiten) |
| 2. Trainingsseiten exportieren | Projektakte: `python -m projektakte.ocr_testseiten --training --anzahl 600 --ziel /data/training --ohne-seiten /data/ocr-test-2/seiten.json` | Spark | gebaut |
| 3. Modelle vergleichen | `scripts/ocr_vergleich.py` (mit `--vorgabe`: `PROMPT` und Auswertung der Typzeile) | PC oder Spark | gebaut |
| 4. Entwürfe schreiben lassen | `scripts/ocr_entwurf.py` | Spark (Qwen) | gebaut |
| 5. Korrigieren | `scripts/ocr_korrektur.py` (Browser, 127.0.0.1:8765) | PC | gebaut |
| 6. Trainingsdaten und Prüfsatz ausgeben | `scripts/ocr_korrektur.py --export training.jsonl` (schreibt auch `training.pruef.jsonl`) | PC | gebaut |
| 7. Training | Trainingswerkzeug in `training/` (LLaMA-Factory), startet der Nutzer | Spark | **zu bauen** (Abschnitt 8.1) |
| 8. Prüfen gegen Soll-Texte | Prüfskript, liest `<Export>.pruef.jsonl` | Spark | **zu bauen** (Abschnitt 8.2) |
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
Mit der Vorgabe vergleichen (fragt mit `PROMPT` aus `ocr_vorgabe.py` und wertet die `TYP:`-Zeile aus):
```bash
python3 ~/tune/scripts/ocr_vergleich.py --ordner ocr-test-2 --vorgabe --modell "Qwen-4B|http://localhost:8003/v1|qwen35-4b" --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' --parallel 8 --ausgabe ergebnis-vorgabe-qwen4b
```
Die Übersicht bekommt dann weitere Spalten: Seiten je Typ, „ohne Typzeile“, „Leer mit Inhalt“ (Widerspruch),
„Foto ohne Inhalt“, Seiten mit Stempel und Stempel gesamt (viele Stempel deuten auf Erfundenes).

Die Übersicht (Seiten/Min., Ø Tokens, Schleifen, abgeschnitten) enthält keinen Seitentext und darf weitergegeben werden.
`ergebnis-…/vergleich.html` zeigt Bild, bisherigen Text und Modelltext nebeneinander. Seiten/Min. zwischen PC und Spark
sind nicht vergleichbar, Antwort-Tokens je Seite schon.

## 8. Nächste Aufgaben

Jede Aufgabe ist so beschrieben, dass eine neue Sitzung sie ohne Vorwissen übernehmen kann. 8.1 und 8.2 baut die KI,
8.3 und 8.4 macht der Nutzer, 8.5 und 8.6 folgen danach.

### 8.1 Trainingswerkzeug bauen (KI, eigene Sitzung auf dem Spark)

**Ziel:** Der Nutzer startet ein Training mit einem Befehl, versteht, was dabei geschieht, und kann später
weitertrainieren. Code in `training/` mit eigener `training/README.md` (Deutsch); er darf mehr als die
Standardbibliothek brauchen. Installation und Läufe liegen **außerhalb** des Repositorys (`~/tune-training`).

**Was es können muss:**
1. **Einrichten:** ein Skript, das LLaMA-Factory mit PyTorch für CUDA 13 in `~/tune-training` installiert und prüft
   (Grafikkarte sichtbar, beide Modelle ladbar). Was am 17.11.2026 schon geklärt wurde, steht unten.
2. **Trainieren:** ein Befehl mit Modell (`qwen2b`, `glm-ocr`, `qwen4b`) und einer oder mehreren JSONL-Dateien aus
   dem Export. Das Werkzeug weigert sich, eine `*.pruef.jsonl` zu trainieren.
3. **Zwei Wege** (Abschnitt 5): neu mit allen Daten (Regelfall) und weiter auf dem Adapter eines früheren Laufs.
4. **Jeden Lauf nummeriert ablegen:** Adapter, Checkpoints, Einstellungen, Datenstand (welche JSONL, Seitenzahl je
   Typ, Prüfsumme von `PROMPT`), Verlustkurve. Ein Lauf überschreibt nie einen anderen.
5. **Bildgröße wie im Betrieb:** Seiten mit 150 dpi, A4 ≈ 2,2 MP. Die Beispiel-Einstellung von LLaMA-Factory
   (`image_max_pixels: 262144`) ist dafür rund achtmal zu klein und muss gesetzt werden; zu klären ist, was mit
   größeren Seiten (Pläne, Abschnitt 6.9) geschieht.
6. **Validierung beim Training:** ein kleiner Teil der Trainingsdaten (ganze Dokumente), nie der Prüfsatz.
7. **Nach dem Training:** Adapter zusammenführen, quantisieren (FP8, NVFP4), Startbefehl für vLLM mit begrenztem
   Speicher. Jede Fassung bekommt einen Namen, unter dem das Prüfskript sie misst.
8. **Anleitung:** jede Einstellung in einem Satz erklärt (Rang, Lernrate, Epochen, Stapelgröße, Bildgröße, was
   eingefroren ist), dazu „woran erkenne ich, dass ein Lauf gut oder schlecht war“ und „was tun bei zu wenig Speicher“.

**Abnahme der Aufgabe:** Funktionstest mit wenigen **erfundenen** Seiten (im Test erzeugt, keine echten Daten) für
beide Modelle: einrichten, kurz trainieren, weitertrainieren, zusammenführen, in vLLM starten, eine Seite fragen.
Kein echtes Training durch die KI. `python -m pytest -q` grün für alles, was ohne Grafikkarte prüfbar ist.

**Am 17.11.2026 auf dem Spark geklärt:**
- LLaMA-Factory (Stand `885a4ca` vom 17.11.2026) kennt die Vorlagen `qwen3_5`, `qwen3_5_nothink` (ohne Denken, die
  richtige für hier), `glm_ocr` und `gemma4`. Es verlangt Python ≥ 3.11, transformers ≤ 5.17.0, peft 0.18.
- Offizielle GLM-OCR-Anleitung (LoRA): `template: glm_ocr`, `trust_remote_code: true`, `lora_rank: 8`,
  `lora_target: all`, Lernrate 1e-4, 3 Epochen, `bf16`, ab 8 GB; Zusammenführen mit `llamafactory-cli export`.
  Laut Anleitung brauchte GLM-OCR eine neuere transformers-Fassung als die damals voreingestellte.
- Das Datenformat des Exports (`messages` + `images`, „sharegpt“) passt; es braucht einen Eintrag in
  `dataset_info.json` und eine `<image>`-Marke je Bild (ist im Export so).
- Spark: Python 3.12.3 **ohne** `Python.h` (`python3-dev` fehlt; ohne es scheitert das Training laut
  NVIDIA-Anleitung). Ausweg ohne `sudo`: `uv venv --python-preference only-managed --python 3.12` (`uv` liegt in
  `~/.local/bin`). Treiber 580, CUDA 13.0, PyTorch-Index `https://download.pytorch.org/whl/cu130` (unbestätigt, ob
  es dort ARM64-Räder gibt). 2,4 TB frei.
- In `~/tune-training` liegen ein Klon von LLaMA-Factory und eine **leere** venv; die Installation wurde abgebrochen.
- Ein Erfahrungsbericht nennt den Fehler „model type 'qwen3_5' not recognized“ bei zu alter transformers-Fassung.

### 8.2 Prüfskript bauen (KI)

Liest `<Export>.pruef.jsonl` (Bild, Anweisung, Soll-Text), fragt ein laufendes Modell und vergleicht: Typ richtig
oder falsch (Verwechslungstabelle), Zeichen-Fehlerrate (CER) gegen den Soll-Text, fehlende und erfundene Zeilen,
erfundene Stempel, Inhalt auf leeren Seiten, Schleifen, Antwort-Tokens, alles getrennt je Seitentyp. Ausgabe als
Übersicht **ohne Seitentext** und als HTML zum Durchsehen. Mehrere Läufe und Fassungen (BF16, FP8, NVFP4)
nebeneinander. Nur Standardbibliothek, in `scripts/`, mit Tests.

### 8.3 Seiten exportieren, Entwürfe, Korrektur (Nutzer)

Trainings- und Prüfseiten in einem Zug, rund 800 Seiten (grob 15–30 Stunden verteilt), Befehle in Abschnitt 7.
Entwürfe und Soll-Texte, die vor dem 17.11.2026 entstanden sind, passen nicht mehr zur Vorgabe. Vor dem Export in
Projektakte klären: Bildgröße großer Pläne (Abschnitt 6.9) und ob `--ohne-seiten` ganze Dokumente ausschließen soll.

### 8.4 Trainieren und messen (Nutzer, mit dem Werkzeug aus 8.1 und 8.2)

Erst ein kleiner Lauf mit ~200 Seiten je Kandidat, messen, dann mehr Daten. Nach dem Export prüfen, ob jeder
Seitentyp im Prüfsatz ausreichend vertreten ist (die Ausgabe zählt je Typ). Ergebnisse in Abschnitt 4 eintragen.

### 8.5 Abnahme

Die quantisierte Fassung des besseren Kandidaten gegen das Abnahmekriterium (Abschnitt 6.1) messen.

### 8.6 Übergabe an Projektakte

Fertiges Modell, Startbefehl auf dem Spark mit begrenztem Speicher (das Modell läuft neben dem großen Frage-Modell
und dem Embedding-Dienst, eher 8–12 GB), Beschreibung für den Einbau. Der Einbau selbst geschieht im Repository
`projektakte`: eigene Einstellungen `VISION_URL`/`VISION_MODEL`, `PROMPT` aus `ocr_vorgabe.py`, `TYP:` auswerten und
speichern, wiederkehrende Kopf- und Fußzeilen je Dokument entfernen, Schleifenerkennung beibehalten, Antworten ohne
Typzeile oder mit Inhalt bei `TYP: Leer` als Fehler behandeln.

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
- **Stand auf dem Spark am 17.11.2026:** Von 121 GB waren rund 30 GB verfügbar; das große Frage-Modell (Container
  `vllm-fn-tp1`) belegt den Großteil. Der Container `qwen35-4b` lief nicht, ein App-Container von Projektakte auch
  nicht. Im Hugging-Face-Cache liegen `zai-org/GLM-OCR` und zwei quantisierte Qwen-3.5-4B-Fassungen, kein Original.
  Für das Training (BF16, große Bilder) reicht der freie Speicher neben dem Frage-Modell voraussichtlich nicht;
  vorher messen oder das Frage-Modell für die Dauer des Trainings anhalten (Entscheidung des Nutzers).
- **Sitzung direkt auf dem Spark:** Dieses Repository liegt dort auch unter `~/tune`, eine KI-Sitzung kann also auf
  dem Spark selbst laufen und Befehle, Container und Training dort starten. Die KI ist dabei weiterhin ein
  Cloud-Modell: Sie darf **keine Seitenbilder, Seitentexte, Entwürfe, Soll-Texte oder `seiten.json` öffnen oder
  ausgeben** (Regel „Keine Cloud“). Erlaubt sind Übersichten ohne Seitentext, Zähler, Protokolle des Trainings.

## 11. Dateien

| Datei | Zweck |
|---|---|
| `scripts/ocr_vorgabe.py` | die Vorgabe: `PROMPT`, `TYPES`, `split`/`join` der Antwortform |
| `scripts/ocr_vergleich.py` | Modelle auf einem Seitenordner vergleichen (OpenAI-kompatible Server; Tempo, Tokens, Schleifen; HTML nebeneinander; `--extra-body`; `--vorgabe` für `PROMPT` und Typzeile) |
| `scripts/ocr_entwurf.py` | Entwürfe nach der Vorgabe erzeugen (`<Seite>.entwurf.txt`, fortsetzbar) |
| `scripts/ocr_korrektur.py` | Korrigieren im Browser (`<Seite>.soll.txt`), Export als JSONL, getrennt in Training und Prüfsatz nach Dokument (`--pruef-anteil`, `--ohne-dokumente`, `--bild-basis`) |
| `tests/` | Tests der Werkzeuge (ohne echte Daten) |
| Projektakte: `projektakte/ocr_testseiten.py` | Seiten exportieren (Prüfsatz, `--training`) |
| Projektakte: `DECISIONS.md` „Was auf eine Seite gehört“ und frühere Abschnitte zur Bildauswertung | Hintergrund |

## 12. Anhang: Recherche zu Grundmodellen und Werkzeugen (17.11.2026)

Recherche vom 17.11.2026 (Quellen: Modellkarten, vLLM-Rezepte, Anleitungen der
Trainingswerkzeuge; Sekundärquellen, vor dem Training an der Modellkarte gegenprüfen). Kandidaten:

| Kandidat | Größe | Lizenz | vLLM | Training | Wozu |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | 4,66B | Apache 2.0 | Rezept vorhanden (ab 0.17) | LLaMA-Factory (`template: qwen3_5`), ms-swift (offizielle Anleitung), Unsloth (Bild-Feintuning) | sicherer Start: im Betrieb bekannt, versteht Anweisungen |
| `Qwen/Qwen3.5-2B` | 2,27B | Apache 2.0 | wie 4B | wie 4B | gleiche Familie, halbe Größe: nach dem Training vielleicht gut genug und deutlich schneller |
| `zai-org/GLM-OCR` | 1,33B | MIT | läuft (Abschnitt 4) | offizielle Anleitung mit LLaMA-Factory | sauberster Leser, am schnellsten; offen, ob das kleine Sprachmodell Typ und Auslassen lernt |
| Gemma 4 E4B | „effektiv 4B“ | Apache 2.0 | seit Erscheinen unterstützt | noch zu prüfen | Gegenprobe aus anderer Familie |

**Ergebnis:** Trainiert werden Qwen3.5-2B und GLM-OCR (Abschnitt 5). GLM-OCR ist auf feste Kurzanweisungen
trainiert und befolgt `PROMPT` ohne Training nicht; es zeigt sich erst nach dem Training.

**Weitere aktuelle Modelle, gesichtet am 17.11.2026 (Sekundärquellen, nicht selbst gemessen):**

| Modell | Größe | Lizenz | Einschätzung |
|---|---|---|---|
| NVIDIA Nemotron 3.5 Parse | 0,63B | OpenMDW-1.1 | reiner Parser (Bild-Encoder + mBART-Decoder) mit festem Ausgabeformat, nimmt keine Anweisung, kein Feintuning-Weg gefunden. Kein Grundmodell; brauchbar als **zweiter Entwurf** für Soll-Texte |
| Baidu Unlimited-OCR-2 | 3,3B | MIT | Stärke sind hundertseitige Dokumente in einem Durchgang; hier wird je Seite gelesen. Nicht kleiner als Qwen 4B, Feintuning-Weg nicht gefunden |
| LightOnOCR-3 | 0,8B / 1B / 4B | Apache 2.0 | neu (10/2026), vLLM-Unterstützung; **möglicher dritter Trainingskandidat**, wenn sich ein Feintuning-Weg bestätigt |
| OvisOCR2 | 0,8B | Apache 2.0 | Spitzenwerte im Benchmark, Feintuning-Weg nicht gefunden |
| MinerU2.5-Pro | 1,2B | Apache 2.0 | zweistufig mit eigener Layout-Erkennung, passt nicht zu „ein Aufruf je Seite“ |
| HunyuanOCR-1.5 | 1B | unklar | Veröffentlichung der Gewichte und Lizenz unklar |
| DeepSeek-OCR 2, dots.mocr | 3B | Apache 2.0 / MIT mit Zusatz | kein Größenvorteil gegenüber Qwen |
| TeleOCR, PaddleOCR-VL 1.6 | – | – | lief beim Nutzer nicht bzw. nur mit eigener Layout-Erkennung (Abschnitt 4) |

**Ist Qwen 3.5 veraltet? Stand 17.11.2026 (Hugging Face, Trendliste und Modellangaben):** In der Klasse bis 5B ist
Qwen 3.5 (02/2026) weiterhin das neueste von Qwen; Qwen 3.6 und 3.8 gibt es offen erst ab 27B. Neue kleine Modelle
vom Herbst 2026 bauen auf Qwen 3.5 auf (`xiaomi-research/Xiaomi-OCR-0` auf Qwen3.5-0.8B-Base). Neuere kleine
Modelle anderer Hersteller:

| Modell | Größe | Lizenz | Einschätzung |
|---|---|---|---|
| `LiquidAI/LFM3-VL-2B` (10/2026) | 1,9B | LFM Open License 1.0 (eigene Lizenz) | laut Hersteller vor Qwen3.5-2B, aber nur mit Denken (71,4 gegen 64,0); ohne Denken 62,1. Hier läuft das Modell ohne Denken. vLLM-Unterstützung „in Arbeit“ |
| `tencent/WeVisOne-VL-4B` (10/2026) | 4,8B | Apache 2.0 | Schwerpunkt Objekte, Masken, Szenentext; baut auf Qwen3-VL auf; keine Dokument-Werte |
| `google/gemma-4-E4B-it` (03/2026) | – | Apache 2.0 | gleich alt wie Qwen 3.5 |
| `xiaomi-research/Xiaomi-OCR-0` (09/2026) | 0,87B | Apache 2.0 | OCR-Leser auf Qwen3.5-0.8B-Base; ließe sich vermutlich mit denselben Werkzeugen wie Qwen 3.5 trainieren (Annahme der KI, ungeprüft). **Möglicher Kandidat** |
| `openbmb/MiniCPM-OCR-1.0` (10/2026) | 1,3B | Apache 2.0 | OCR-Leser auf MiniCPM-V 4.6, eigene Bibliothek; Feintuning-Weg ungeprüft |

Die Benchmark-Werte dieser Leser liegen dicht beieinander (95–97 auf OmniDocBench) und messen reines Abschreiben.
Entscheidend ist hier, was noch niemand gemessen hat: das Verhalten **nach** dem Feintuning auf die Vorgabe
(Typ bestimmen, weglassen, nichts erfinden). Gewählt wird deshalb nach Feintuning-Weg, vLLM, Lizenz und Tempo.

**Achtung:** Auf dem Spark liegen von Qwen 3.5 4B nur **quantisierte** Fassungen (`AxionML/Qwen3.5-4B-NVFP4`,
`RedHatAI/Qwen3.5-4B-FP8-dynamic`). Zum Training braucht es das Original `Qwen/Qwen3.5-4B` (BF16), das noch
heruntergeladen werden muss. Mit welcher Fassung die Messungen in Abschnitt 4 liefen, ist nicht festgehalten.

**Trainingswerkzeuge:** LLaMA-Factory deckt beide Kandidaten ab (Vorlagen für Qwen 3.5, offizielle GLM-OCR-Anleitung)
und hat eine NVIDIA-Anleitung für den DGX Spark (venv mit PyTorch für CUDA 13; BF16 statt FP8 empfohlen). ms-swift
hat eine offizielle Qwen-3.5-Anleitung, Unsloth unterstützt Bild-Feintuning für Qwen 3.5 und hat ebenfalls eine
Spark-Anleitung.
