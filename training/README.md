# Training des Seitenlesers

Mit diesem Ordner trainierst du den Seitenleser selbst: ein Befehl je Schritt, jeder Lauf in einem eigenen
nummerierten Ordner. Diese Anleitung erklärt, was jeder Schritt und jede Einstellung bewirkt. Hintergrund,
Vorgabe und Entscheidungen stehen in [`../CLAUDE.md`](../CLAUDE.md).

Alle Befehle laufen auf dem Spark im Ordner `~/tune`. Installation, Läufe und Modelle liegen **außerhalb** des
Repositorys in `~/tune-training` (anderer Ort: Umgebungsvariable `TUNE_TRAINING`).

## Überblick

| Schritt | Befehl | Ergebnis |
|---|---|---|
| 1. Einrichten (einmal) | `bash training/einrichten.sh` | `~/tune-training/venv` mit PyTorch und LLaMA-Factory |
| 2. Prüfen | `python3 training/tune.py pruefen --modell qwen2b --modell glm-ocr` | Modelle liegen bereit, Vorlagen stimmen |
| 3. Trainieren | `python3 training/tune.py trainieren --modell qwen2b --daten ~/training.jsonl` | Lauf `001-qwen2b` mit Adapter |
| 4. Übersicht | `python3 training/tune.py laeufe` | Tabelle aller Läufe mit Verlust |
| 5. Zusammenführen | `python3 training/tune.py zusammenfuehren --lauf 1` | `modell-bf16` |
| 6. Quantisieren | `python3 training/tune.py quantisieren --lauf 1 --fassung fp8` | `modell-fp8` bzw. `modell-nvfp4` |
| 7. Starten | `python3 training/tune.py starten --lauf 1 --fassung fp8 --speicher-gb 10` | Startbefehl für vLLM |
| 8. Fragen | `python3 training/tune.py fragen --lauf 1 --fassung fp8 --bild seite.png` | Antwort auf eine Seite |

Gemessen wird danach mit dem Prüfskript (Abschnitt 8.2 in `CLAUDE.md`) gegen `training.pruef.jsonl`.

## Was beim Training geschieht

Das Grundmodell besteht aus einem **Bildteil** (er zerlegt die Seite in Kacheln und macht daraus Zahlenreihen,
„Bild-Tokens“), einer **Brücke** und einem **Sprachteil**, der aus Bild-Tokens und Anweisung die Antwort schreibt.
Lesen kann das Modell schon. Es soll lernen, sich an die Vorgabe zu halten: Typzeile, weglassen, nichts erfinden.

- **LoRA:** Die Gewichte des Modells bleiben, wie sie sind. Neben die Schichten des Sprachteils wird je ein kleiner
  Zusatz gelegt, der **Adapter**; nur er wird gelernt (unter 1 % der Gewichte). Bildteil und Brücke sind eingefroren.
- **Ein Lernschritt:** Das Modell bekommt Seite und Anweisung, dazu die Soll-Antwort. Gemessen wird, wie
  unwahrscheinlich es jedes Zeichen der Soll-Antwort fand – der **Verlust**. Der Adapter wird ein kleines Stück so
  verändert, dass der Verlust sinkt.
- **Validierung:** Ein Teil der Dokumente (10 %) wird nicht gelernt. Nach jeder Epoche wird der Verlust auf diesen
  Seiten gemessen. Daran sieht man, ob das Modell lernt oder nur auswendig lernt. Die Validierung ist **nicht** der
  Prüfsatz: Sie stammt aus den Trainingsdaten und begleitet das Training, der Prüfsatz misst das fertige Modell.

## 1. Einrichten

```bash
bash training/einrichten.sh
```

Das Skript holt LLaMA-Factory in einem festen Stand, legt mit `uv` eine eigene Python-Umgebung an (das Python des
Systems hat auf dem Spark kein `Python.h`, das von `uv` verwaltete bringt es mit), installiert PyTorch für CUDA 13
und prüft, ob die Grafikkarte sichtbar ist. Es kann beliebig oft laufen. Für das Quantisieren kommt später eine
zweite Umgebung dazu (ihre Pakete vertragen sich nicht mit denen von LLaMA-Factory):

```bash
bash training/einrichten.sh --quantisierung
```

Danach die Modelle prüfen. Fehlt eines, wird es von Hugging Face nach `~/tune-training/modelle` heruntergeladen
(nur die Modelle; Seiten verlassen den Rechner nie, und Training, Zusammenführen und Quantisieren arbeiten ohne
Netz). Der eigene Modellordner ist Absicht: Teile des gemeinsamen Hugging-Face-Caches gehören auf dem Spark `root`,
weil Docker-Container sie geladen haben, und lassen sich vom Training nicht benutzen.

```bash
python3 training/tune.py pruefen --modell qwen2b --modell glm-ocr
```

Die Zeile „Anfrage im Betrieb … stimmt mit dem Training überein“ ist wichtig, siehe „Vorlagen“ unten.

## 2. Trainieren

```bash
python3 training/tune.py trainieren --modell qwen2b --daten ~/projektakte/deploy/training.jsonl
```

| Modell | Grundmodell | Vorlage |
|---|---|---|
| `qwen2b` | `Qwen/Qwen3.5-2B` | `qwen3_5_nothink` |
| `glm-ocr` | `zai-org/GLM-OCR` | `glm_ocr` |
| `qwen4b` | `Qwen/Qwen3.5-4B` (Rückfallebene) | `qwen3_5_nothink` |

**Ohne Anmelden auf dem Spark:** Das Korrekturwerkzeug läuft dort als Dienst und ist über Tailscale im Browser
erreichbar (`http://100.97.122.72:8765`). Oben rechts führt „Training“ zu einer Seite mit dem Knopf „Exportieren
und trainieren“: Modell wählen, Knopf drücken. Das Werkzeug exportiert die korrigierten Seiten (Prüfsatz daneben)
und startet genau den Befehl oben. Die Seite zeigt die Läufe und das Ende des Protokolls und lädt sich neu, solange
das Training läuft; weiterkorrigieren kannst du währenddessen. Die folgenden Abschnitte erklären, was dabei geschieht.

`--daten` nimmt eine oder mehrere JSONL-Dateien aus `ocr_korrektur.py --export`. Vor dem Start prüft das Werkzeug:

- **Kein Prüfsatz:** Eine `*.pruef.jsonl` wird abgelehnt. Liegt neben `training.jsonl` eine `training.pruef.jsonl`
  (so schreibt sie der Export), darf keine Trainingsseite aus einem ihrer Dokumente stammen. Weitere Prüfsätze nennt
  `--pruefsatz` (z. B. den der 49 alten Prüfseiten).
- **Aktuelle Anfrage:** Das feingetunte Modell wird **ohne Anweisung** gefragt, es bekommt nur das Seitenbild
  (`ANWEISUNG` in `scripts/ocr_vorgabe.py` ist leer). Jede Zeile muss genau so aussehen und mit einer Typzeile
  antworten; ältere Exporte mit der langen Anweisung werden abgelehnt.
- **Bilder da:** Jedes Seitenbild muss unter dem Pfad aus der JSONL liegen (beim Export: `--bild-basis`).
- **Speicher:** Ist weniger frei als nötig, startet der Lauf nicht (siehe „Zu wenig Speicher“).

Dann legt es den Lauf an, startet LLaMA-Factory und zeigt dessen Ausgabe. Abbrechen mit Strg+C.

### Der Laufordner

```
~/tune-training/laeufe/001-qwen2b/
  einstellungen.yaml   alle Einstellungen, jede mit einem Satz erklärt (genau diese Datei bekommt LLaMA-Factory)
  lauf.json            Datenstand: welche JSONL (mit Prüfsumme), Seiten je Typ, Prüfsummen von ANWEISUNG und PROMPT, Verlust, Status
  daten/               die Seiten dieses Laufs: training.jsonl, validierung.jsonl (echte Texte – bleiben auf dem Spark)
  adapter/             der Adapter, je Epoche ein Checkpoint, training_loss.png, trainer_log.jsonl
  protokoll.log        die ganze Ausgabe
  modell-bf16/         nach dem Zusammenführen
  modell-fp8/, modell-nvfp4/   nach dem Quantisieren
```

Die Nummer zählt hoch. Ein Lauf überschreibt nie einen anderen, auch ein abgebrochener behält seine Nummer.

### Die Einstellungen

| Einstellung | Option | Voreinstellung | Was sie bewirkt |
|---|---|---|---|
| Epochen | `--epochen` | 3 (weiter: 2) | Wie oft jede Seite gesehen wird. Mehr lernt gründlicher, ab einem Punkt aber auswendig. |
| Lernrate | `--lernrate` | 1e-4 (weiter: 5e-5) | Wie stark ein Lernschritt den Adapter verändert. Zu hoch: Verlust springt; zu niedrig: es tut sich wenig. |
| Rang | `--rang` | 8 | Größe des Adapters. Höher kann mehr lernen, braucht mehr Daten und lernt eher auswendig. |
| Stapelgröße | `--stapel` | 1 | Seiten, die gleichzeitig durch das Modell gehen. Mehr braucht entsprechend mehr Speicher. |
| Sammeln | `--sammeln` | 8 | So viele Stapel werden gesammelt, bevor gelernt wird. Wirkt wie Stapelgröße 8 ohne deren Speicherbedarf. |
| Bildgröße | `--bild-pixel` | 2.359.296 | Größte Bildfläche. A4 mit 150 dpi (2,18 MP) bleibt unverkleinert, Größeres wird proportional verkleinert. |
| Länge | `--laenge` | 8192 | Obergrenze für Bild, Anweisung und Antwort zusammen in Tokens. Was länger ist, wird abgeschnitten. |
| Validierung | `--val-anteil` | 0,1 | Anteil der Dokumente, die nicht gelernt, sondern nach jeder Epoche gemessen werden. |

Fest eingestellt (in `einstellungen.yaml` nachzulesen): LoRA auf allen Schichten des Sprachteils, Bildteil und
Brücke **eingefroren**, Rechnen in BF16, Lernrate mit Anlauf (10 % der Schritte) und danach sanft fallend,
Checkpoint und Validierung nach jeder Epoche, fester Zufallsstartwert.

**Schritte ausrechnen:** Lernschritte je Epoche = Trainingsseiten ÷ (Stapel × Sammeln). 200 Seiten ergeben 25
Schritte je Epoche, bei 3 Epochen 75.

**Dauer und Speicher** (gemessen am 17.11.2026 im Funktionstest mit A4-Seiten und kurzen Antworten; echte Seiten
mit langen Antworten brauchen mehr von beidem):

| Modell | Speicher im Training | Zeit je Seite und Epoche | 200 Seiten, 3 Epochen (grob) |
|---|---|---|---|
| `qwen2b` | rund 15 GB | rund 5 s | 1 Stunde |
| `glm-ocr` | rund 6 GB | rund 2,5 s | eine halbe Stunde |
| `qwen4b` | nicht gemessen (Schätzung 30 GB) | – | – |

Eine A4-Seite ergibt bei Qwen rund 2.100 Bild-Tokens, bei GLM-OCR rund 2.800. Eine Anweisung kommt nicht dazu
(gemessen wurde noch mit der langen Anweisung von rund 650 Tokens).

**Bildgröße:** Sie muss im Training und im Betrieb gleich sein, sonst sieht das Modell im Betrieb andere Bilder,
als es gelernt hat. Der Startbefehl für vLLM übernimmt sie deshalb aus dem Lauf. Seiten über der Grenze (A3, große
Pläne) werden verkleinert; das Werkzeug meldet vor dem Start, wie viele das sind. Ein A3-Blatt mit 150 dpi
(4,4 MP) kommt so mit rund 110 dpi an, ein A0-Plan mit rund 40 dpi und wäre unlesbar (offener Punkt 6.9 in `CLAUDE.md`). Eine
größere Grenze kostet Speicher und Zeit im gleichen Verhältnis.

## 3. War der Lauf gut?

`python3 training/tune.py laeufe` zeigt je Lauf den letzten Verlust im Training und in der Validierung. Die Kurve
liegt als Bild in `adapter/training_loss.png`, die Zahlen stehen in `adapter/trainer_log.jsonl`.

| Was du siehst | Was es bedeutet | Was tun |
|---|---|---|
| Beide Verluste fallen und flachen ab | Das Modell lernt, wie es soll. | Am Prüfsatz messen. |
| Training fällt weiter, Validierung steigt wieder | Auswendiglernen. | Weniger Epochen (der Checkpoint der besten Epoche liegt in `adapter/`), kleinerer Rang oder mehr Seiten. |
| Beide fallen bis zum Schluss deutlich | Es wäre noch mehr zu holen. | Eine Epoche mehr. |
| Der Verlust springt oder steigt | Lernrate zu hoch. | Lernrate halbieren. |
| Der Verlust bewegt sich kaum | Lernrate zu niedrig oder Daten widersprüchlich. | Lernrate verdoppeln; Soll-Texte stichprobenweise ansehen. |
| Verlust nahe null im Training | Auswendig gelernt (bei wenigen Seiten schnell erreicht). | Wie oben; die Messung am Prüfsatz entscheidet. |

Der Verlust sagt nur, wie gut das Modell die Soll-Texte trifft. Ob es Stempel erfindet oder in Schleifen gerät,
zeigt erst die Messung am Prüfsatz – und zwar an der quantisierten Fassung, die in den Betrieb geht.

## 4. Weitertrainieren

Zwei Wege, wenn neue korrigierte Seiten dazukommen:

**Neu mit allen Daten (Regelfall).** Alle JSONL-Dateien angeben. Sauberster Weg; bei einigen hundert Seiten kurz.

```bash
python3 training/tune.py trainieren --modell qwen2b --daten ~/training.jsonl ~/training-2.jsonl
```

**Weiter auf dem Adapter eines früheren Laufs.** Nur die neuen Seiten angeben:

```bash
python3 training/tune.py trainieren --modell qwen2b --daten ~/training-2.jsonl --weiter-von 1
```

Das geht schneller, aber das Modell kann dabei Älteres verlernen. Deshalb mischt das Werkzeug alte Seiten aus dem
früheren Lauf bei (`--alt-anteil 0.5`: die Hälfte der Trainingsseiten sind alte, immer dieselbe Auswahl) und lernt
kleiner (halbe Lernrate, 2 Epochen). Die Validierungsseiten des früheren Laufs bleiben Validierungsseiten. Der neue
Lauf bekommt eine eigene Nummer, der alte bleibt unverändert. Der Rang ist der des früheren Laufs.

Abgelehnt wird das Weitertrainieren, wenn das Modell ein anderes ist oder sich `ANWEISUNG` oder die Regeln (`PROMPT`) seit dem früheren Lauf
geändert hat. Dann: alle Soll-Texte anpassen und neu trainieren.

Für alle Läufe gilt **derselbe Prüfsatz** (er darf wachsen), sonst sind sie nicht vergleichbar.

## 5. Zusammenführen und Quantisieren

```bash
python3 training/tune.py zusammenfuehren --lauf 1
python3 training/tune.py quantisieren --lauf 1 --fassung fp8
python3 training/tune.py quantisieren --lauf 1 --fassung nvfp4
```

- **Zusammenführen** rechnet den Adapter in das Grundmodell ein. Heraus kommt ein gewöhnliches Modell in voller
  Genauigkeit (`modell-bf16`), das vLLM ohne Zusatz laden kann. Der Adapter bleibt daneben erhalten.
- **Quantisieren** speichert die Gewichte des Sprachteils mit weniger Bits (llm-compressor). Bildteil und
  Ausgabeschicht bleiben in voller Genauigkeit.
  - **FP8** (8 Bit): braucht keine Daten, kostet erfahrungsgemäß kaum Genauigkeit.
    **Achtung GLM-OCR:** Im Funktionstest las die FP8-Fassung von GLM-OCR direkt (mit transformers) richtig, in
    vLLM 0.27.1 (dem Image `vllm/vllm-openai:latest` auf dem Spark, Stand 08/2026) schrieb sie aber nur Unsinn in
    Schleife. Qwen in FP8 lief dort richtig. Vor einer Messung von GLM-OCR in FP8 ein neueres vLLM-Image holen
    (`starten --image …`) und an einer Seite prüfen.
  - **NVFP4** (4 Bit): kleiner und schneller, kostet bei kleinen Modellen spürbar mehr. Es braucht Beispielseiten
    zum Einmessen („Kalibrierung“); genommen werden 64 Trainingsseiten des Laufs (`--seiten`), nie Prüfseiten.

Jede Fassung hat einen Namen aus Lauf und Fassung, z. B. `001-qwen2b-fp8`. Unter diesem Namen läuft sie in vLLM,
und unter ihm misst sie das Prüfskript. **Abgenommen wird die quantisierte Fassung**, deshalb BF16, FP8 und NVFP4
am Prüfsatz nebeneinander messen.

## 6. Starten in vLLM

```bash
python3 training/tune.py starten --lauf 1 --fassung fp8 --speicher-gb 10
```

Das zeigt den vollständigen `docker run`-Befehl; mit `--ausfuehren` startet ihn das Werkzeug selbst (Container
`tune-001-qwen2b-fp8`, nur auf diesem Rechner erreichbar, Port 8004). Anhalten: `tune.py stoppen --lauf 1
--fassung fp8`. Das Werkzeug fasst nur Container an, deren Name mit `tune-` beginnt.

Was der Befehl festlegt:

- **Speichergrenze:** vLLM nimmt sich sonst fast den ganzen Speicher. `--speicher-gb 10` wird in den Anteil am
  Gesamtspeicher umgerechnet (`--gpu-memory-utilization`). Zu knapp bemessen, startet vLLM nicht und sagt das im
  Protokoll (`docker logs`). Im Funktionstest liefen beide Modelle mit 8 GB und zwei gleichzeitigen Anfragen
  (belegt waren 4–6 GB); für 24 gleichzeitige Anfragen ist mehr nötig, das zeigt die Messung im Betrieb.
- **Token-Budget:** `--max-num-batched-tokens 16384`. Mit der Voreinstellung von vLLM (2.048) wird eine A4-Seite
  bei GLM-OCR abgelehnt, weil ihre 2.800 Bild-Tokens nicht hineinpassen.
- **Bildgröße wie im Training** (`--mm-processor-kwargs`), siehe oben. Geprüft mit einer übergroßen Seite: Beide
  Modelle verkleinern sie im Betrieb auf die Grenze des Trainings.
- **Vorlage des Trainings** (`--chat-template`), siehe unten.
- **Ein Bild je Anfrage**, Antwort höchstens 4.096 Tokens, Temperatur 0 (immer die wahrscheinlichste Fortsetzung).

Eine Seite fragen:

```bash
python3 training/tune.py fragen --lauf 1 --fassung fp8 --bild ~/projektakte/deploy/training/<Seite>.jpg
```

### Vorlagen: im Betrieb fragen wie im Training

Eine Anfrage wird für das Modell in Steuerzeichen verpackt („Hier spricht der Nutzer, hier kommt das Bild, jetzt
antworte“). LLaMA-Factory verpackt im Training nach seiner eigenen Vorlage, vLLM im Betrieb nach der, die dem
Modell beiliegt. Die beiden sind nicht gleich (die beiliegende fügt z. B. einen leeren Denkblock ein). Ein Modell,
das im Betrieb anders gefragt wird als im Training, antwortet schlechter. Deshalb liegt in `training/vorlagen/` je
Modellfamilie eine Vorlage, die genau die Form des Trainings erzeugt; sie wird als `betrieb.jinja` neben jedes
fertige Modell gelegt und vLLM beim Start mitgegeben. `tune.py pruefen --modell …` vergleicht beide Formen Token
für Token.

Eine Beobachtung dazu aus dem Funktionstest: Das kaum trainierte Qwen-Modell schrieb auf manchen Seiten von sich
aus einen leeren Denkblock (`<think></think>`) vor die Typzeile, weil das Grundmodell ihn so kennt. Mit echtem
Training lernt es, direkt mit `TYP:` zu beginnen. Taucht der Denkblock nach dem Training noch auf, ist die
Alternative die LLaMA-Factory-Vorlage `qwen3_5` mit abgeschaltetem Denken; sie setzt den leeren Denkblock schon in
die Anfrage, wie es die dem Modell beiliegende Vorlage tut. Das wäre eine Änderung in `tune.py` (`MODELLE`) und
`vorlagen/` und ein neues Training.

**Für den Einbau in Projektakte heißt das:** das Modell mit dem Startbefehl dieses Werkzeugs starten,
dann ist die Reihenfolge von Bild und Text in der Anfrage gleichgültig, und `enable_thinking` muss nicht mehr
mitgeschickt werden.

## 7. Zu wenig Speicher

Der Spark hat einen gemeinsamen Speicher für Prozessor und Grafikkarte. Geht er aus, beendet das System Prozesse –
das kann auch das große Frage-Modell treffen. Deshalb prüft das Werkzeug vor dem Start, wie viel frei ist.

- **Schutz:** Training, Zusammenführen, Quantisieren und der vLLM-Container dieses Werkzeugs sind so markiert, dass
  das System bei Speichermangel **sie zuerst** beendet und nicht die anderen Dienste.
- **Nachsehen:** `free -g` (Spalte „verfügbar“) und `tune.py pruefen`.
- **„CUDA error: out of memory“ gleich beim Start**, obwohl Speicher „verfügbar“ ist: Die Grafikkarte kann nur
  wirklich freien Speicher nehmen (Spalte „frei“), nicht den Dateicache (Spalte „Puffer/Cache“), der sich z. B.
  nach großen Downloads füllt. Kurz warten oder den Cache leeren: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.
- **Platz schaffen:** andere Modelle für die Dauer des Trainings anhalten (Entscheidung des Nutzers; das große
  Frage-Modell belegt den Großteil). Nie zwei Trainings oder Training und vLLM-Test gleichzeitig.
- **Bedarf senken**, in dieser Reihenfolge: Stapelgröße 1 lassen; `--laenge` senken, wenn die Antworten kurz sind;
  `--bild-pixel` senken (dann im Betrieb dieselbe kleinere Größe, das kostet Lesbarkeit); kleineres Modell.
- **Der Rang** ändert am Speicher fast nichts.
- Bricht ein Lauf mit „out of memory“ ab, ist nichts verloren: Checkpoints bis dahin liegen in `adapter/`, der
  nächste Versuch bekommt eine neue Nummer.

## 8. Funktionstest

Prüft die ganze Kette mit einem Dutzend **erfundener** Seiten (im Test gezeichnet): kurz trainieren,
weitertrainieren, zusammenführen, optional quantisieren, in vLLM starten, eine Seite fragen. Das Modell lernt dabei
nichts Brauchbares; es geht nur darum, dass alles zusammenspielt. Die Läufe liegen getrennt von den echten in
`~/tune-training/funktionstest`.

```bash
~/tune-training/venv/bin/python training/funktionstest.py --modell glm-ocr
~/tune-training/venv/bin/python training/funktionstest.py --modell qwen2b --fassung fp8
```

Stand 17.11.2026 auf dem Spark: für beide Modelle durchgelaufen (GLM-OCR in BF16, Qwen3.5-2B in BF16, FP8 und
NVFP4). Installiert sind PyTorch 2.14.1 (CUDA 13), transformers 5.17.0, peft 0.18.1, LLaMA-Factory `885a4ca`,
llm-compressor 0.15.0; vLLM 0.27.1 im Docker-Image.

## Dateien

| Datei | Zweck |
|---|---|
| `einrichten.sh` | Umgebung in `~/tune-training` anlegen und prüfen |
| `tune.py` | alle Befehle (nur Standardbibliothek; ruft `llamafactory-cli`, `docker` und `quantisieren.py`) |
| `venv_hilfe.py` | Prüfungen, die die Trainingsumgebung brauchen (Modelle laden, Vorlagen vergleichen) |
| `quantisieren.py` | FP8 und NVFP4 mit llm-compressor (läuft in `venv-quant`) |
| `vorlagen/*.jinja` | Gesprächsvorlagen für den Betrieb in der Form des Trainings |
| `funktionstest.py` | Funktionstest mit erfundenen Seiten |
