# tune

Ein feingetunter Seitenleser für die Bildauswertung von **Projektakte**: Er liest gescannte Seiten aus Projektakten
eines Brandschutz-Ingenieurbüros nach einer festen Vorgabe – Seitentyp in der ersten Zeile (`TYP: Text|Plan|Foto|Leer`),
bei Text die vollständige Abschrift, bei Plänen nur Schriftfeld und Legende, bei Fotos nur sichtbaren Text, bei Leerem
nichts. Nichts erfinden, nichts beschreiben.

Alles Weitere – Problem, Vorgabe, Messungen, Entscheidungen, Ablauf, Befehle, nächste Schritte, Regeln – steht in
[`CLAUDE.md`](CLAUDE.md).

**Keine echten Daten in dieses Repository:** Seitenbilder, Texte daraus und Modelle bleiben im Büro.

```bash
python -m pytest -q
```
