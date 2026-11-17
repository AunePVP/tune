"""The target for reading a page (agreed with the office, see DECISIONS "Was auf eine Seite gehört"): the page type in
the first line, then what is to be written for that type. One prompt for drafts (ocr_entwurf.py), the correction tool
(ocr_korrektur.py), training (the exported JSONL) and later operation – a fine-tuned model is asked with it as well.
"""
from __future__ import annotations

TYPES = ("Text", "Plan", "Foto", "Leer")

PROMPT = (
    "Du liest eine einzelne Seite aus der Projektakte eines Brandschutz-Ingenieurbüros. Schreibe in die erste Zeile "
    "den Seitentyp: „TYP: Text“, „TYP: Plan“, „TYP: Foto“ oder „TYP: Leer“. Der Typ richtet sich nach der Art der "
    "Seite, auch wenn danach nichts zu schreiben ist. Danach:\n"
    "- Text (Brief, Bericht, Bescheid, Prüfbericht, Gutachten, Formular, Datenblatt, auch mit eingebetteten Fotos "
    "oder Abbildungen): den gesamten sichtbaren Text wörtlich in Lesereihenfolge, auch Kopf- und Fußzeilen und "
    "Bildunterschriften. Tabellen als Markdown-Tabellen, Ankreuzfelder als ☒ oder ☐ mit ihrer Beschriftung, "
    "Handschrift als [handschriftlich: …] an ihrer Stelle, Unterschriften als [Unterschrift]. Text in eingebetteten "
    "Fotos nur, wenn er klar lesbar ist, an der Stelle des Fotos; ist er es nicht, schreibst du dazu nichts, auch "
    "kein [unleserlich].\n"
    "- Plan (Grundriss, Schnitt, Lageplan, Brandschutz- und Fluchtplan): das Schriftfeld (Planart, Objekt und "
    "Adresse, Geschoss oder Bereich, Plan-Nr., Index, Datum, Maßstab, Verfasser), die Legende, zusammenhängende "
    "Textblöcke mit Hinweisen oder Auflagen (mindestens ein ganzer Satz oder eine mehrzeilige Notiz), Stempel und "
    "handschriftliche Eintragungen mit Worten als [handschriftlich: …]. Ist eine solche Eintragung da, aber nicht "
    "lesbar, schreibst du einmal [handschriftlich: unleserlich]. Keine Raumnamen, Maße, Achsen, Flurstücke oder "
    "einzelnen Beschriftungen in der Zeichnung.\n"
    "- Foto: nur klar lesbarer Text, der im Foto zu sehen ist (Typenschilder, Zulassungsnummern, Beschriftungen) – "
    "keine Beschreibung des Fotos. Ohne solchen Text nach der Typzeile nichts.\n"
    "- Leer (unbedruckte Seite oder nur Grafik, Logo, Hintergrund): nach der Typzeile nichts.\n"
    "Stempel: jeden Stempel, der einen Vermerk, ein Datum, ein Aktenzeichen oder eine Prüf- oder Listennummer trägt "
    "(Genehmigung, Prüfvermerk, Eingang), genau einmal und wörtlich als [Stempel: …], nicht lesbare Teile darin als "
    "[unleserlich]. Reine Namens- und Adressstempel und Logos lässt du weg.\n"
    "Schreibe nur, was auf der Seite zu sehen ist. Unleserliche Stellen als [unleserlich], je Stelle einmal. Erfinde "
    "nichts, beschreibe nichts, fasse nichts zusammen."
)


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
