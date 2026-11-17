# tune – ein feingetunter Seitenleser für Projektakte

Diese Datei ist der Einstieg für jede KI-Sitzung in diesem Repository. Sie enthält alles, was man zum Weiterarbeiten
braucht: worum es geht, was schon entschieden und gemessen ist, wie gearbeitet wird und was als Nächstes kommt.
Lies sie ganz, bevor du etwas änderst.

---

## Stand (17.11.2026)

- **Fertig:** die Vorgabe (Abschnitt 3, mit dem Nutzer geschärft), die Werkzeuge für Vergleich, Entwürfe, Korrektur
  und Export mit festem Prüfsatz (Abschnitt 7), die Wahl der Trainingskandidaten (Abschnitt 5), das
  **Trainingswerkzeug** in `training/` (Abschnitt 8.1; auf dem Spark eingerichtet, Funktionstest mit erfundenen
  Seiten für beide Kandidaten durchgelaufen).
- **Als Nächstes:** das Prüfskript bauen (Abschnitt 8.2, eigene Sitzung). Parallel exportiert und korrigiert der
  Nutzer die Seiten (Abschnitt 8.3) und trainiert dann selbst (Abschnitt 8.4, Anleitung `training/README.md`).
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

**Regeln und Anfrage stehen nur einmal** in `scripts/ocr_vorgabe.py`:
- `PROMPT`: die Regeln dieses Abschnitts als Anweisung in Worten. Damit werden **untrainierte** Modelle gefragt
  (Entwürfe, Vergleich mit `--vorgabe`), und das Korrekturwerkzeug zeigt sie unter „Regeln“.
- `ANWEISUNG`: womit das **feingetunte** Modell im Training und im Betrieb gefragt wird. Sie ist **leer**: Das Modell
  bekommt nur das Seitenbild (Entscheidung des Nutzers, 17.11.2026, Abschnitt 5). Export, Training, Prüfung und
  später Projektakte nutzen genau diese Fassung.
- `TYPES`, `split`, `join`: die Antwortform.

Wer `PROMPT` ändert, ändert die Regeln: Soll-Texte müssen dazu passen. Wer `ANWEISUNG` ändert, muss neu trainieren,
und Projektakte muss sie übernehmen.

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
- **Das feingetunte Modell wird ohne Anweisung gefragt** (Entscheidung des Nutzers, 17.11.2026): Es hat nur diese
  eine Aufgabe und bekommt nur das Seitenbild (`ANWEISUNG = ""` in `ocr_vorgabe.py`). Die Regeln lernt es allein aus
  den korrigierten Seiten. Hinweis der KI dazu: Das spart rund 650 Eingabe-Tokens je Seite, bringt wenig Tempo (die
  Zeit kostet das Schreiben der Antwort) und braucht eher mehr Trainingsseiten; ob es reicht, zeigt die Messung am
  Prüfsatz. Der Weg zurück ist eine Zeile (`ANWEISUNG = PROMPT`), dann neu exportieren und neu trainieren.
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
- **Trainingswerkzeug: LLaMA-Factory** (Funktionstest auf dem Spark am 17.11.2026 bestanden, Abschnitt 8.1). Es deckt
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
    Datenstand (welche JSONL, Seitenzahl, Fassung von `ANWEISUNG` und `PROMPT`), je Lauf ein nummerierter Ordner.
  - **Neue Seiten:** weiterer Export aus Projektakte in einen eigenen Ordner (`training-2`, mit `--ohne-seiten` der
    bisherigen), Entwürfe, Korrektur, Export. Die JSONL-Dateien werden aneinandergehängt. Der Hash je Dokument hält
    die Aufteilung in Training und Prüfsatz über alle Ordner gleich.
  - **Derselbe Prüfsatz** für alle Läufe (er darf wachsen), sonst sind sie nicht vergleichbar.
  - **Ändern sich die Regeln (`PROMPT`):** alle Soll-Texte anpassen und neu trainieren, nicht weitertrainieren
    (nach einer bloßen Klarstellung: `--regeln-geaendert`). Ändert sich `ANWEISUNG`: immer neu trainieren.

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
   wird (Abschnitt 10). Gemessen (Abschnitt 8.1): rund 15 GB für Qwen3.5-2B, rund 6 GB für GLM-OCR bei kurzen
   Antworten; mit rund 18 GB frei neben dem Frage-Modell wird es für Qwen knapp. Der Nutzer entscheidet.
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
| 3. Modelle vergleichen | `scripts/ocr_vergleich.py` (mit `--vorgabe`: `PROMPT` und Auswertung der Typzeile; mit `--trainiert`: feingetunte Modelle, nur das Bild) | PC oder Spark | gebaut |
| 4. Entwürfe schreiben lassen | `scripts/ocr_entwurf.py` | Spark (Qwen) | gebaut |
| 5. Korrigieren | `scripts/ocr_korrektur.py` (Browser, 127.0.0.1:8765) | PC | gebaut |
| 6. Trainingsdaten und Prüfsatz ausgeben | `scripts/ocr_korrektur.py --export training.jsonl` (schreibt auch `training.pruef.jsonl`) | PC | gebaut |
| 7. Training, Zusammenführen, Quantisieren, Start in vLLM | `training/tune.py` (LLaMA-Factory, llm-compressor; Anleitung `training/README.md`), startet der Nutzer | Spark | gebaut, Funktionstest bestanden |
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

Korrigieren **auf dem Spark über Tailscale** (so arbeitet der Nutzer seit dem 17.11.2026; nichts muss kopiert
werden): Das Korrekturwerkzeug läuft dort als Dienst `tune-korrektur` (systemd des Benutzers, Vorlage
`training/tune-korrektur.service`) für `~/training` und ist im Browser unter `http://100.97.122.72:8765`
erreichbar (nur im Tailnet; `--host` nie auf `0.0.0.0`, das Werkzeug hat keine Anmeldung). Der Link „Training“
öffnet eine Seite mit dem Knopf **„Exportieren und trainieren“**: Er schreibt `~/training.jsonl` und
`~/training.pruef.jsonl` und startet `training/tune.py trainieren` für das gewählte Modell; die Seite zeigt Läufe
und das Ende des Protokolls (`~/training.training.log`). Es läuft immer nur ein Training. Dienst ansehen oder neu
starten: `systemctl --user status|restart tune-korrektur`.

Vor dem Korrigieren einmal die Lage der Textzeilen ermitteln (Spark, einige Minuten, fortsetzbar):
`~/tune-training/venv/bin/python ~/tune/scripts/ocr_bloecke.py --ordner ~/training` (mit diesem Python werden seitlich gescannte Seiten erst aufrecht gedreht; das Werkzeug zeigt sie dann auch aufrecht an). Im Werkzeug: Klick auf eine Textstelle im Bild markiert
den passenden Text rechts; der Cursor im Text zeigt die Zeile im Bild. Zoom mit den Knöpfen, Strg+Mausrad oder
Doppelklick, Drehen mit ⟲/⟳ (nur Ansicht).

Korrigieren auf dem PC (Alternative; Ordner `training` und `scripts/` aus diesem Repository dorthin kopieren):
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
`{"messages":[{"role":"user","content":"<image>"},{"role":"assistant","content":"TYP: …"}],"images":["<Pfad>"]}`.
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

### 8.1 Trainingswerkzeug (erledigt am 17.11.2026)

Gebaut in `training/`, Anleitung in `training/README.md` (dort steht jede Einstellung erklärt). Installation, Modelle
und Läufe liegen außerhalb des Repositorys in `~/tune-training` (`venv`, `venv-quant`, `LLaMA-Factory`, `modelle`,
`laeufe`, `funktionstest`). Kurzfassung der Befehle (Spark, Ordner `~/tune`):

```bash
bash training/einrichten.sh                      # einmal; mit --quantisierung auch llm-compressor
python3 training/tune.py pruefen --modell qwen2b --modell glm-ocr
python3 training/tune.py trainieren --modell qwen2b --daten ~/training.jsonl                # neu
python3 training/tune.py trainieren --modell qwen2b --daten ~/training-2.jsonl --weiter-von 1   # weiter
python3 training/tune.py laeufe
python3 training/tune.py zusammenfuehren --lauf 1
python3 training/tune.py quantisieren --lauf 1 --fassung fp8      # oder nvfp4
python3 training/tune.py starten --lauf 1 --fassung fp8 --speicher-gb 10 [--ausfuehren]
python3 training/tune.py fragen --lauf 1 --fassung fp8 --bild <Seite>
```

**Was das Werkzeug festlegt:**
- Je Lauf ein nummerierter Ordner `~/tune-training/laeufe/NNN-<modell>` mit `einstellungen.yaml`, `lauf.json`
  (JSONL mit Prüfsumme, Seiten je Typ, Prüfsummen von `ANWEISUNG` und `PROMPT`, Verlust), `daten/`, `adapter/` (Checkpoint je Epoche,
  Verlustkurve), `protokoll.log`, später `modell-bf16`, `modell-fp8`, `modell-nvfp4`. Nummern werden nie
  wiederverwendet.
- Abgelehnt werden: `*.pruef.jsonl`; Seiten oder Dokumente, die in der `*.pruef.jsonl` neben der Datendatei oder in
  einem mit `--pruefsatz` genannten Prüfsatz vorkommen; Zeilen mit einer anderen Anfrage als `ANWEISUNG` oder ohne
  Typzeile; fehlende Bilder; ein Start bei zu wenig freiem Speicher (außer mit `--trotzdem`).
- Validierung: 10 % der Dokumente der Trainingsdaten (fester Hash je Dokument mit eigenem Salz, unabhängig vom
  Prüfsatz). Das Dokument wird aus dem Bildnamen abgeleitet (`NNN_<Datei>_S<Seite>`), weil die JSONL kein Feld
  `dokument` hat.
- Weitertrainieren: Adapter des früheren Laufs, halbe Lernrate (5e-5), 2 Epochen, alte Trainingsseiten beigemischt
  (`--alt-anteil 0.5`); abgelehnt bei anderem Modell, geänderter `ANWEISUNG` oder geänderten Regeln (`PROMPT`, außer mit
  `--regeln-geaendert`).
- Voreinstellungen wie in der offiziellen GLM-OCR-Anleitung: Rang 8, Lernrate 1e-4, 3 Epochen, BF16, LoRA auf allen
  Schichten des Sprachteils, Bildteil und Brücke eingefroren; Stapel 1 × Sammeln 8; Länge 8.192 Tokens.
- **Bildgröße:** `image_max_pixels` 2.359.296 (1536², A4 mit 150 dpi bleibt unverkleinert). Größere Seiten werden
  proportional verkleinert, im Training und – über den Startbefehl – genauso im Betrieb (mit einer übergroßen Seite
  für beide Modelle geprüft). Für große Pläne heißt das: A3 kommt mit rund 110 dpi an, A0 mit rund 40 dpi
  (Abschnitt 6.9 bleibt offen).
- **Vorlagen:** LLaMA-Factory verpackt eine Anfrage im Training anders als die dem Modell beiliegende Vorlage, die
  vLLM sonst nimmt (bei Qwen fügt diese einen leeren Denkblock ein). `training/vorlagen/*.jinja` erzeugen genau die
  Form des Trainings; sie liegen als `betrieb.jinja` neben jedem fertigen Modell und gehen mit `--chat-template` an
  vLLM. `tune.py pruefen` vergleicht beide Formen Token für Token. **Für den Einbau in Projektakte (8.6):** das
  Modell mit dem Startbefehl aus `tune.py starten` betreiben; `enable_thinking` muss dann nicht mitgeschickt werden.
- Name einer Fassung für Betrieb und Prüfskript: `<Lauf>-<Fassung>`, z. B. `001-qwen2b-fp8`; Container
  `tune-<Name>`, Port 8004, nur lokal erreichbar.
- Schutz bei Speichermangel: Training, Quantisieren und der vLLM-Container des Werkzeugs sind so markiert
  (`oom_score_adj`), dass das System sie zuerst beendet und nicht andere Dienste.

**Funktionstest am 17.11.2026** (erfundene Seiten, `training/funktionstest.py`; kein echtes Training):
- Durchgelaufen für `qwen2b` (BF16, FP8, NVFP4) und `glm-ocr` (BF16): trainieren, weitertrainieren,
  zusammenführen, quantisieren, in vLLM starten, Seite fragen.
- Installiert: PyTorch 2.14.1+cu130 (ARM64-Räder gibt es), transformers 5.17.0, peft 0.18.1, LLaMA-Factory
  `885a4ca`, llm-compressor 0.15.0 (eigene `venv-quant`, weil es `datasets` ≥ 4.8 braucht, LLaMA-Factory ≤ 4.0);
  vLLM 0.27.1 im Image `vllm/vllm-openai:latest` (Stand 08/2026; aktuell wäre 0.34).
- Speicher im Training mit A4-Seiten und kurzen Antworten: `qwen2b` rund 15 GB, `glm-ocr` rund 6 GB; rund 5 s bzw.
  2,5 s je Seite und Epoche. In vLLM mit 8-GB-Grenze belegten beide 4–6 GB (zwei gleichzeitige Anfragen).
- **GLM-OCR in FP8 läuft in vLLM 0.27.1 nicht richtig:** Die quantisierte Fassung liest direkt mit transformers
  richtig, in vLLM schreibt sie Unsinn in Schleife. Qwen in FP8 läuft richtig. Vor einer Messung von GLM-OCR in FP8
  ein neueres vLLM-Image holen und an einer Seite prüfen. GLM-OCR in NVFP4 ist nicht ausprobiert.
- Das kaum trainierte Qwen-Modell schrieb auf manchen Seiten einen leeren Denkblock (`<think></think>`) vor die
  Typzeile – das Grundmodell kennt ihn so. Nach echtem Training prüfen; bleibt er, ist die Alternative die Vorlage
  `qwen3_5` mit abgeschaltetem Denken (setzt den leeren Denkblock in die Anfrage). Das Prüfskript (8.2) sollte
  solche Antworten als „ohne Typzeile“ zählen.
- Stolpersteine, im Werkzeug berücksichtigt: transformers 5 kennt `warmup_ratio` und `overwrite_output_dir` nicht
  mehr; `uv` braucht für die großen NVIDIA-Pakete ein längeres Zeitlimit; der Hugging-Face-Cache von GLM-OCR gehört
  `root` (von Docker geladen), deshalb der eigene Ordner `~/tune-training/modelle`; vLLM lehnt A4-Seiten für GLM-OCR
  ohne `--max-num-batched-tokens 16384` ab; GLM-OCR zählt die Bildgrenze im Betrieb doppelt
  (`size.longest_edge` = 2 × Bildfläche); „CUDA out of memory“ gleich beim Start kommt vom vollen Dateicache.
- Nicht geprüft: `qwen4b` (Original nicht heruntergeladen), ein Training mit echten Seiten und langen Antworten,
  das Tempo in vLLM, die Güte von FP8/NVFP4 (das misst erst das Prüfskript).

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
`projektakte`: eigene Einstellungen `VISION_URL`/`VISION_MODEL`, Anfrage wie `ANWEISUNG` in `ocr_vorgabe.py` (nur das Bild, kein Text), `TYP:` auswerten und
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
  Neben dem Frage-Modell waren später nur rund 18 GB verfügbar; das Training braucht rund 15 GB (Qwen3.5-2B) bzw.
  6 GB (GLM-OCR), Abschnitt 8.1. Für den Funktionstest hat der Nutzer das Frage-Modell selbst gestoppt; ob es für
  echte Trainings angehalten wird, entscheidet er. Die Grundmodelle `Qwen3.5-2B` und `GLM-OCR` liegen jetzt in
  `~/tune-training/modelle` (der Cache-Ordner von GLM-OCR gehört `root`); `Qwen/Qwen3.5-4B` fehlt weiterhin.
  `python3` des Systems hat kein `pytest`; Tests laufen mit `uv run --no-project --with pytest python -m pytest -q`.
- **Sitzung direkt auf dem Spark:** Dieses Repository liegt dort auch unter `~/tune`, eine KI-Sitzung kann also auf
  dem Spark selbst laufen und Befehle, Container und Training dort starten. Die KI ist dabei weiterhin ein
  Cloud-Modell: Sie darf **keine Seitenbilder, Seitentexte, Entwürfe, Soll-Texte oder `seiten.json` öffnen oder
  ausgeben** (Regel „Keine Cloud“). Erlaubt sind Übersichten ohne Seitentext, Zähler, Protokolle des Trainings.

## 11. Dateien

| Datei | Zweck |
|---|---|
| `scripts/ocr_vorgabe.py` | die Vorgabe: `PROMPT` (Regeln, für untrainierte Modelle), `ANWEISUNG` (Anfrage an das feingetunte Modell: leer), `TYPES`, `split`/`join` der Antwortform |
| `scripts/ocr_vergleich.py` | Modelle auf einem Seitenordner vergleichen (OpenAI-kompatible Server; Tempo, Tokens, Schleifen; HTML nebeneinander; `--extra-body`; `--vorgabe` für `PROMPT` und Typzeile) |
| `scripts/ocr_entwurf.py` | Entwürfe nach der Vorgabe erzeugen (`<Seite>.entwurf.txt`, fortsetzbar); `--trainiert`, wenn ein feingetuntes Modell die Entwürfe schreibt |
| `scripts/ocr_bloecke.py` | Lage der Textzeilen je Seite mit dem lokalen `tesseract` ermitteln (`<Seite>.bloecke.json`); damit markiert das Korrekturwerkzeug nach einem Klick auf die Seite die passende Textstelle und umgekehrt. Nur Hilfe beim Korrigieren, geht nie in die Soll-Texte |
| `scripts/ocr_korrektur.py` | Korrigieren im Browser (`<Seite>.soll.txt`), Export als JSONL, getrennt in Training und Prüfsatz nach Dokument (`--pruef-anteil`, `--ohne-dokumente`, `--bild-basis`); `--host` für den Zugriff über Tailscale; Seite „Training“ (Export und Start von `training/tune.py`), wenn `training/` daneben liegt |
| `training/tune-korrektur.service` | Vorlage: Korrekturwerkzeug als Dienst auf dem Spark |
| `training/README.md` | Anleitung zum Training: jeder Schritt, jede Einstellung, „war der Lauf gut?“, „zu wenig Speicher“ |
| `training/einrichten.sh` | Trainingsumgebung in `~/tune-training` anlegen (LLaMA-Factory, PyTorch für CUDA 13; `--quantisierung`: llm-compressor in `venv-quant`) |
| `training/tune.py` | alle Befehle des Trainings: `pruefen`, `trainieren` (neu oder `--weiter-von`), `laeufe`, `zusammenfuehren`, `quantisieren`, `starten`, `stoppen`, `fragen` (nur Standardbibliothek) |
| `training/venv_hilfe.py`, `training/quantisieren.py` | Teile, die die Trainings- bzw. Quantisierungsumgebung brauchen |
| `training/vorlagen/*.jinja` | Gesprächsvorlagen für den Betrieb in vLLM, in genau der Form des Trainings |
| `training/funktionstest.py` | ganze Kette mit erfundenen Seiten |
| `tests/` | Tests der Werkzeuge (ohne echte Daten); `test_training.py` prüft das Trainingswerkzeug ohne Grafikkarte |
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
