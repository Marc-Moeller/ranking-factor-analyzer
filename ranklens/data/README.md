# Language resources

`stopwords.json` is a selected, alphabetically sorted subset of the Stopwords ISO
collection distributed in `stopwordsiso` 0.6.1. It is bundled as data, so an
analysis needs no external download, dependency import or provider request.

- Collection: https://github.com/stopwords-iso/stopwords-iso
- Original sources and contributors: https://github.com/stopwords-iso/stopwords-iso/blob/master/CREDITS.md
- Licence: `STOPWORDS_LICENSE.txt` (MIT, Gene Diaz)

Included codes: bg, ca, cs, da, de, el, es, et, eu, fi, fr, ga, gl, hr, hu, it,
lt, lv, nl, no, pl, pt, ro, sk, sl, sv, tr, uk.

The Norwegian list is shared for no/nb/nn. English keeps the existing stoplists
at each call site to preserve existing English behavior. Regional language tags
(e.g. fr-CA, pt-BR, de_AT) select their base language for lexical rules; the
original requested tag is still passed to providers, whose supported tags may
be more limited.

Stopwords are a heuristic for excluding common words, not proof that a word is
irrelevant to a particular search. These lists may include multiword entries;
current filtering applies only to individual tokens. Question starters are also
heuristics and do not perform grammatical parsing.

No dedicated stopword lists are bundled for Icelandic, Albanian, Serbian,
Bosnian, Macedonian or Maltese. These languages still get Unicode tokenization
and explicit market/language selection, but no English stopword fallback.
Word segmentation for languages without spaces, stemming, compound splitting,
elision handling and report translation are outside this patch.
