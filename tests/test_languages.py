"""Shared lexical behavior across European languages, without network calls."""
import pytest

from ranklens.analyze.semantic import _tokens, _tfidf_matrix
from ranklens.clients.dataforseo import _location_code
from ranklens.extract.corpus import build_corpus
from ranklens.extract.factors import extract_html_factors
from ranklens.keywords import build_variation_set
from ranklens.language import STOPWORDS, default_language, language_code, tokens
from ranklens.models import AnalyzeRequest


CASES = [
    ('da', 'og', 'trægulv', 'Hvordan'),
    ('de', 'und', 'wärmepumpe', 'Warum'),
    ('fr', 'avec', 'rénovation', 'Comment'),
    ('es', 'para', 'aislamiento', 'Cómo'),
    ('pt', 'para', 'isolamento', 'Como'),
    ('it', 'con', 'riscaldamento', 'Come'),
    ('nl', 'met', 'vloerverwarming', 'Waarom'),
    ('sv', 'och', 'värmepump', 'Hur'),
    ('no', 'og', 'varmepumpe', 'Hvordan'),
    ('fi', 'ja', 'lämpöpumppu', 'Miten'),
    ('pl', 'oraz', 'ocieplenie', 'Jak'),
    ('cs', 'ale', 'čerpadlo', 'Jak'),
    ('sk', 'ale', 'čerpadlo', 'Ako'),
    ('sl', 'in', 'črpalka', 'Kako'),
    ('hr', 'ali', 'izolacija', 'Kako'),
    ('ro', 'pentru', 'încălzire', 'Cum'),
    ('hu', 'hogy', 'hőszivattyú', 'Hogyan'),
    ('el', 'και', 'μόνωση', 'Πώς'),
    ('bg', 'или', 'изолация', 'Как'),
    ('tr', 'ile', 'yalıtım', 'Nasıl'),
    ('lv', 'un', 'siltumsūknis', 'Kā'),
    ('lt', 'ir', 'šildymas', 'Kaip'),
    ('et', 'ja', 'soojuspump', 'Kuidas'),
    ('ca', 'amb', 'aïllament', 'Com'),
    ('gl', 'para', 'illamento', 'Como'),
    ('ga', 'agus', 'insliú', 'Conas'),
    ('eu', 'eta', 'isolamendua', 'Nola'),
    ('uk', 'але', 'утеплення', 'Як'),
]


@pytest.mark.parametrize('language,stop,word,question', CASES)
def test_localized_lexical_pipeline(language, stop, word, question):
    vset = build_variation_set(f'{stop} {word}', language)
    assert word in vset.variations
    assert stop not in vset.variations
    assert _tokens(f'{stop} {word}', language) == [word]
    corpus = build_corpus([f'{stop} {word}', f'{stop} {word}', 'xyzsolar', 'abcsolar'], build_variation_set('keyword', language))
    assert word in corpus.lsi_terms
    assert stop not in corpus.lsi_terms
    matrix = _tfidf_matrix([f'{stop} {word}', word, 'unrelatedfixture'], language)
    assert matrix[0] @ matrix[1] > .99
    factors, _ = extract_html_factors(f'<body><h1>{question} {word}</h1></body>', 'https://example.com', 'example.com', vset, 1)
    assert factors['QUESTIONS_H1'] == 1
    assert 'FLESCH_EASE' not in factors


def test_every_bundled_stoplist_has_a_language_regression():
    assert set(STOPWORDS) == {row[0] for row in CASES}


@pytest.mark.parametrize('country,location,language', [
    ('dk', 2208, 'da'), ('de', 2276, 'de'), ('fr', 2250, 'fr'),
    ('nl', 2528, 'nl'), ('se', 2752, 'sv'), ('no', 2578, 'nb'),
    ('fi', 2246, 'fi'), ('es', 2724, 'es'), ('it', 2380, 'it'),
    ('pt', 2620, 'pt'), ('pl', 2616, 'pl'), ('cz', 2203, 'cs'),
    ('br', 2076, 'pt'), ('mx', 2484, 'es'), ('ca', 2124, 'en'),
    ('ch', 2756, 'de'), ('be', 2056, 'nl'), ('ua', 2804, 'uk'),
])
def test_market_selection(country, location, language):
    assert _location_code(country) == location
    assert default_language(country) == language
    assert AnalyzeRequest(keyword='test', country=country).language == language
    assert AnalyzeRequest(keyword='test', country=country, language='fr').language == 'fr'


@pytest.mark.parametrize('tag,base', [('fr-CA','fr'), ('pt-BR','pt'), ('de_AT','de'), ('nb-NO','no'), ('nn','no')])
def test_regional_and_norwegian_lexical_aliases(tag, base):
    assert language_code(tag) == base


def test_typographic_apostrophe_and_unicode_letters():
    assert tokens('L’isolation café Straße blå') == ["l'isolation", 'café', 'straße', 'blå']
    assert build_variation_set("O'Neill", 'en').count('O’Neill') == 1


def test_article_header_remains_in_clean_content():
    html = '<body><header>Site menu</header><main><article><header><h1>Rénovation</h1></header><p>Isolation</p></article></main></body>'
    factors, body = extract_html_factors(html, 'https://example.fr', 'example.fr', build_variation_set('rénovation','fr'), 1)
    assert body == 'Rénovation Isolation'
    assert factors['WORD_COUNT'] == 2


@pytest.mark.parametrize('language,heading', [('es', '¿Cómo elegir aislamiento'), ('fr', '« Comment choisir isolation'), ('da', 'Hvordan vælger du trægulv')])
def test_question_heading_with_leading_punctuation(language, heading):
    factors, _ = extract_html_factors(f'<h1>{heading}</h1>', 'https://example.com', 'example.com', build_variation_set('keyword', language), 1)
    assert factors['QUESTIONS_H1'] == 1
