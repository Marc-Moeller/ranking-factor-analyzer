"""Shared Unicode tokenization and small language-specific lexical rules.

Bundled stopword lists cover 28 European languages. Unknown languages retain
Unicode words but use no English lexical heuristics. See data/README.md.
This is tokenization, not stemming or compound-word segmentation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import unicodedata

from ranklens.markets import COUNTRY_LANGUAGE

_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
QUESTION_WORDS = {
    "bg": frozenset("кой коя кое какво кога къде защо как колко".split()),
    "ca": frozenset("qui què quan on com quin quina quins quines quant quanta quants quantes".split()),
    "cs": frozenset("kdo co kdy kde proč jak který která které kolik".split()),
    "da": frozenset("hvem hvad hvornår hvor hvorfor hvordan hvilken hvilket hvilke kan er har skal bør må".split()),
    "de": frozenset("wer was wann wo warum wie welche welcher welches wieso weshalb".split()),
    "el": frozenset("ποιος ποια ποιο τι πότε πού γιατί πως πώς πόσο πόσα".split()),
    "en": frozenset("who what when where why how which can do does is are".split()),
    "es": frozenset("quién quiénes qué cuándo dónde adónde cómo cuál cuáles cuánto cuánta cuántos cuántas".split()),
    "et": frozenset("kes mis millal kus miks kuidas milline mitu".split()),
    "eu": frozenset("nor zer noiz non zergatik nola zein zenbat".split()),
    "fi": frozenset("kuka ketä mikä mitä milloin missä mistä miksi miten kuinka".split()),
    "fr": frozenset("qui quoi quand où pourquoi comment quel quelle quels quelles".split()),
    "ga": frozenset("cé cad cathain cá cén conas".split()),
    "gl": frozenset("quen que cando onde como cal cales canto canta cantos cantas".split()),
    "hr": frozenset("tko što kada gdje zašto kako koji koja koje koliko".split()),
    "hu": frozenset("ki mi mikor hol miért hogyan melyik mennyi".split()),
    "is": frozenset("hver hvað hvenær hvar hvers hvernig".split()),
    "it": frozenset("chi cosa quando dove perché come quale quali quanto quanta quanti quante".split()),
    "lt": frozenset("kas ką kada kur kodėl kaip kuris kuri kiek".split()),
    "lv": frozenset("kas kad kur kāpēc kā kurš kura kuri cik".split()),
    "nl": frozenset("wie wat wanneer waar waarom hoe welke".split()),
    "no": frozenset("hvem hva når hvor hvorfor hvordan hvilken hvilket hvilke".split()),
    "pl": frozenset("kto co kiedy gdzie dlaczego jak który która które ile".split()),
    "pt": frozenset("quem que quando onde como qual quais quanto quanta quantos quantas".split()),
    "ro": frozenset("cine ce când unde cum care cât câte câți".split()),
    "sk": frozenset("kto čo kedy kde prečo ako ktorý ktorá ktoré koľko".split()),
    "sl": frozenset("kdo kaj kdaj kje zakaj kako kateri katera katero koliko".split()),
    "sv": frozenset("vem vad när var vart varför hur vilken vilket vilka".split()),
    "tr": frozenset("kim ne nerede neden niçin nasıl hangi kaç".split()),
    "uk": frozenset("хто що коли де чому як який яка яке скільки".split()),
}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower().replace("’", "'")


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


def language_code(language: str) -> str:
    code = (language or "en").lower().replace("_", "-").split("-")[0]
    return "no" if code in {"nb", "nn"} else code


def stopwords(language: str, english: set[str]) -> set[str] | frozenset[str]:
    code = language_code(language)
    return english if code == "en" else STOPWORDS.get(code, frozenset())


def default_language(country: str) -> str:
    return COUNTRY_LANGUAGE.get(country.strip().lower(), "en")


# Data is bundled under the upstream MIT licence; no runtime download is needed.
# English retains each existing call site's stoplist for backward compatibility.
_STOPWORDS_PATH = Path(__file__).parent / "data" / "stopwords.json"
STOPWORDS = {
    code: frozenset(normalize_text(word) for word in words)
    for code, words in json.loads(_STOPWORDS_PATH.read_text(encoding="utf-8")).items()
}
