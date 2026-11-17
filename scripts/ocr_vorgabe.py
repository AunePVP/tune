"""The target for reading a page (agreed with the office, see DECISIONS "Was auf eine Seite gehört"): the page type in
the first line, then what is to be written for that type. One prompt for drafts (ocr_entwurf.py), the correction tool
(ocr_korrektur.py), training (the exported JSONL) and later operation – a fine-tuned model is asked with it as well.
"""
from __future__ import annotations

TYPES = ("Text", "Plan", "Foto", "Leer")

PROMPT = """Du schreibst eine einzelne Seite aus der Projektakte eines Brandschutz-Ingenieurbüros ab. Erste Zeile: der \
Seitentyp „TYP: Text“, „TYP: Plan“, „TYP: Foto“ oder „TYP: Leer“. Danach nur der Wortlaut der Seite – keine Einleitung, \
keine Beschreibung, keine Zusammenfassung.

Typ:
- Text: Briefe, Berichte, Bescheide, Formulare, Datenblätter, Tabellen, Deckblätter mit Titeltext, auch \
Detailzeichnungen in Zulassungen und Datenblättern. Auch mit eingebetteten Fotos.
- Plan: Pläne eines Bauprojekts (Grundriss, Schnitt, Ansicht, Lageplan, Brandschutz- und Fluchtplan).
- Foto: mit einer Kamera aufgenommen. Nur klar lesbarer Text im Foto (Typenschilder, Zulassungsnummern), sonst nach \
der Typzeile nichts.
- Leer: leere Seite oder nur Logo, Grafik, Icon, Rendering, freigestelltes Produktbild. Nach der Typzeile nichts.

Text-Seiten:
- Reihenfolge: Kopfbereich, Hauptteil, Fußbereich; darin von oben nach unten und links nach rechts, mehrspaltig \
Spalte für Spalte. Gedrehte Seiten so, wie man sie aufrecht liest.
- Ein Absatz ist eine Zeile: Zeilenumbrüche innerhalb eines Absatzes zusammenziehen, Silbentrennung am Zeilenende \
auflösen. Zwischen Blöcken eine Leerzeile.
- Anschrift, Briefkopf, Kopf- und Fußzeilen: jede gedruckte Zeile bleibt eine eigene Zeile. Kopf- und Fußzeilen \
immer vollständig mitschreiben (Seitenzahl, Dateiname, Anschrift, Telefon, Bankdaten), jede nur einmal.
- Überschriften als eigene Zeile mit ihrer Nummer, ohne #. Keine Formatierung: kein Fett, kein Kursiv, keine Links, \
kein LaTeX.
- Aufzählungszeichen (•, –, ▪) als „- “; echte Nummerierung (1., a)) bleibt.
- Tabellen als Markdown-Tabellen, leere Zellen leer. Formulare ohne Raster als „Feldname: Wert“ je Zeile. \
Ankreuzfelder als „☒ Beschriftung“ oder „☐ Beschriftung“. Handschrift in Formularfeldern als Wert des Felds.
- Zeichen wie gedruckt (², §, °, Ø, „…“, Dezimalkomma, alte Rechtschreibung, Groß- und Kleinschreibung), nichts \
korrigieren.
- Text in eingebetteten Fotos nur, wenn er klar lesbar ist, an der Stelle des Fotos; sonst dazu nichts.
- In Detailzeichnungen: Beschriftungen mit Worten oder Artikelnummern, keine reinen Maßzahlen.

Plan-Seiten, in dieser Reihenfolge: das Schriftfeld als „Feldname: Wert“ je Zeile (Feldnamen wie gedruckt, leere \
Felder weglassen), die Legende als „- Eintrag“, zusammenhängende Hinweis-Textblöcke (mindestens ein ganzer Satz oder \
eine mehrzeilige Notiz), Eintragungen der Behörde und Stempel. Keine Raumnamen, Maße, Achsen, Höhenkoten, Flurstücke \
oder einzelnen Beschriftungen in der Zeichnung.

Marker – nur diese vier, keine anderen eckigen Klammern:
- [unleserlich] für eine unlesbare Stelle, je Stelle einmal. Unsichere Ziffern nie raten.
- [Unterschrift] für eine Unterschrift, an ihrer Stelle.
- [Stempel: Wortlaut] für jeden Stempel mit Aussage oder Datum (Genehmigung, Prüfvermerk, Eingang), genau einmal und \
wörtlich. Berufs-, Kammer-, Firmen-, Namens- und Adressstempel weglassen.
- [handschriftlich: Wortlaut] für handschriftliche Notizen außerhalb von Formularfeldern und nachträgliche \
Korrekturen. Auf Plänen einmal [handschriftlich: unleserlich], wenn eine Eintragung da, aber nicht lesbar ist.

Weglassen: Logos (auch Schriftzüge im Logo), Symbole, Durchgestrichenes, Kringel und Markierungen.

Schreibe nur, was auf der Seite zu sehen ist. Erfinde nichts, beschreibe nichts, fasse nichts zusammen. Im Zweifel \
[unleserlich] oder weglassen."""

# What the FINE-TUNED model is asked in training and in operation: nothing but the page image (decision of the user,
# 17.11.2026). The model has this one task and learns the rules from the corrected pages; PROMPT above stays the
# written form of the rules – for drafts by untrained models, for the person correcting, and as documentation.
# Whoever changes ANWEISUNG must train anew and tell Projektakte.
ANWEISUNG = ""


def split(answer: str) -> tuple[str | None, str]:
    """(type, text) of an answer in the target form; type None if the first line names none of TYPES."""
    first, _sep, rest = (answer or "").strip().partition("\n")
    head = first.strip().lstrip("*#").strip()
    if head.upper().startswith("TYP:"):
        name = head[4:].strip().strip("*„“\"").strip()
        for t in TYPES:
            if name.lower().startswith(t.lower()):
                return t, rest.strip()
    return None, (answer or "").strip()


def join(page_type: str, text: str) -> str:
    """The answer in the target form."""
    body = (text or "").strip()
    return f"TYP: {page_type}" + (f"\n{body}" if body and page_type != "Leer" else "")
