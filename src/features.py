"""
features.py -- Feature engineering pipeline.
43 hand-crafted stylometric features + word/char/function-word TF-IDF.
All fitted INSIDE CV folds -- no leakage.
"""
import re
import string
from typing import Dict, List
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from src.utils import get_logger

logger = get_logger(__name__)

# ── Function-word vocabulary ──────────────────────────────────────────────────
# These words carry style (who is writing) rather than topic (what is written).
# Different LLMs have very distinct distributional fingerprints in this space.
FUNCTION_WORDS = [
    # Determiners
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "among", "under", "over", "within", "without", "upon",
    # Coordinating conjunctions
    "and", "but", "or", "so", "yet",
    # Subordinating conjunctions
    "although", "because", "since", "while", "if", "unless", "until",
    "when", "where", "whether", "though",
    # Auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    # Personal pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
    # Discourse / transition markers (strong style fingerprint across LLMs)
    "however", "therefore", "moreover", "furthermore", "additionally",
    "nevertheless", "nonetheless", "thus", "hence", "consequently",
    "accordingly", "whereas", "meanwhile", "subsequently", "overall",
    "also", "too", "either", "neither",
    # Degree / intensifiers
    "very", "quite", "rather", "somewhat", "even", "just", "only",
    "indeed", "certainly", "particularly", "especially", "generally",
    "actually", "basically", "literally", "simply",
    # Quantifiers
    "all", "some", "any", "each", "every", "many", "much", "few", "little",
    "more", "most", "other", "another", "same", "different", "various", "several",
    # Relative / interrogative
    "which", "who", "whom", "whose", "what", "how", "why",
    # Negation
    "not", "no", "never", "nor",
]


class StyleometricTransformer(BaseEstimator, TransformerMixin):
    """
    43 features.
    Key additions over v1 (20 features):
      - very_short/long sentence ratios (DeepSeek short, Grok long)
      - numbered_list_rate    (DeepSeek loves numbered lists)
      - markdown_header_rate  (Gemini/Claude use headers heavily)
      - code_block_rate, bold_rate, italic_rate, link_rate
      - parenthesis_rate, ellipsis_rate, dash_rate, caps_word_ratio
      - avg_para_len_chars, starts_with_i_ratio
    """
    _NUM_RE  = re.compile(r'^\s*\d+[\.)].+', re.MULTILINE)
    _HEAD_RE = re.compile(r'^#{1,6}\s', re.MULTILINE)
    _BOLD_RE = re.compile(r'\*\*')
    _ITAL_RE = re.compile(r'(?<!\*)\*(?!\*)')
    _CODE_RE = re.compile(r'`')
    _LINK_RE = re.compile(r'\[.+?\]\(.+?\)')
    _ELIP_RE = re.compile(r'\.\.\.')
    _CAPS_RE = re.compile(r'\b[A-Z]{2,}\b')
    # Transition/academic words favoured by LLMs (especially DeepSeek)
    _TRANS_RE = re.compile(
        r'\b(however|therefore|moreover|furthermore|additionally|consequently|'
        r'nevertheless|nonetheless|whereas|thus|hence|accordingly|subsequently|'
        r'in conclusion|in summary|for example|for instance|in particular)\b',
        re.IGNORECASE
    )
    _HEDGE_RE = re.compile(
        r'\b(may|might|could|possibly|perhaps|likely|suggests|suggest|'
        r'appears|appear|seems|seem|potentially|presumably|arguably|'
        r'apparently|typically|usually|often)\b',
        re.IGNORECASE
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([self._f(t) for t in X], dtype=np.float32)

    def _f(self, text):
        if not text or not isinstance(text, str):
            return [0.0] * 43
        NL2 = '\n\n'
        NL1 = '\n'
        ch = len(text)
        words = text.split(); nw = max(len(words), 1)
        sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        ns = max(len(sents), 1)
        paras = [p.strip() for p in text.split(NL2) if p.strip()]
        np_ = max(len(paras), 1)
        lines = text.split(NL1); nl_ = max(len(lines), 1)

        wl = [len(w.strip(string.punctuation)) for w in words if w.strip(string.punctuation)]
        awl = float(np.mean(wl)) if wl else 0.0
        ttr = len(set(w.lower().strip(string.punctuation) for w in words)) / nw
        lwr = sum(1 for l in wl if l > 6) / nw
        swc = [len(s.split()) for s in sents]
        vss = sum(1 for c in swc if c < 5) / ns
        vls = sum(1 for c in swc if c > 30) / ns
        apc = float(np.mean([len(p) for p in paras])) if paras else 0.0

        cm  = text.count(',') / nw
        per = text.count('.') / nw
        ex  = text.count('!') / nw
        qu  = text.count('?') / nw
        co  = text.count(':') / nw
        se  = text.count(';') / nw
        qt  = (text.count('"') + text.count("'")
               + text.count('\u201c') + text.count('\u201d')) / nw
        pa  = (text.count('(') + text.count(')')) / nw
        el  = len(self._ELIP_RE.findall(text)) / nw
        da  = (text.count('-') + text.count('\u2014')) / nw

        alpha = [c for c in text if c.isalpha()]
        na    = max(len(alpha), 1)
        ucr   = sum(1 for c in alpha if c.isupper()) / na
        dr    = sum(1 for c in text if c.isdigit()) / max(ch, 1)
        cwr   = len(self._CAPS_RE.findall(text)) / nw
        nlr   = text.count(NL1) / max(ch, 1)

        br  = sum(1 for l in lines if l.lstrip().startswith(('-', '*', '\u2022'))) / nl_
        nr  = len(self._NUM_RE.findall(text)) / nl_
        hr  = len(self._HEAD_RE.findall(text)) / nl_
        cr  = len(self._CODE_RE.findall(text)) / max(ch, 1) * 100
        bor = len(self._BOLD_RE.findall(text)) / max(ch, 1) * 100
        ir  = len(self._ITAL_RE.findall(text)) / max(ch, 1) * 100
        lkr = len(self._LINK_RE.findall(text)) / nw
        isr = sum(1 for s in sents if re.match(r'^I\s', s.strip())) / ns

        # ---- DeepSeek vs Grok discriminators -----------------------------------
        # punct_variety: how many distinct punctuation chars are used (normalised)
        # DeepSeek uses very few (mostly period + colon); Grok more varied
        punct_chars  = [c for c in text if c in string.punctuation]
        punct_variety = len(set(punct_chars)) / max(len(punct_chars), 1)

        # sent_length_cv: coefficient of variation of sentence word counts
        # DeepSeek sentences are uniformly short; Grok has high variance
        if len(swc) > 1:
            sent_cv = float(np.std(swc) / max(np.mean(swc), 1))
        else:
            sent_cv = 0.0

        # transition_word_rate: academic connector words per sentence
        # Both DeepSeek and LLMs use these, but at different rates
        trans_rate = len(self._TRANS_RE.findall(text)) / ns

        # ---- New Run 6: DS/Grok fine-grained discriminators -------------------
        # first_sent_words: word count of first sentence
        #   DS short texts often start with a very concise definition sentence
        fsw = float(len(sents[0].split())) if sents else 0.0

        # proper_noun_density: capitalised words that are NOT sentence-starters / total words
        #   Factual DS texts tend to have more named entities (scientists, places, concepts)
        cap_words = sum(1 for w in words if w and w[0].isupper())
        proper_noun_dens = max(cap_words - ns, 0) / nw

        # hedge_rate: uncertainty language per sentence (Grok hedges more than DS)
        hedge_rate = len(self._HEDGE_RE.findall(text)) / ns

        # question_per_sent: "?" per sentence (DS almost never asks questions)
        qps = text.count('?') / ns

        # sent_range: max - min sentence word count; DS is more length-uniform
        sent_range = float(max(swc) - min(swc)) if len(swc) > 1 else 0.0

        # text_len_log: log10 of char count — length bucket is a style signal
        import math
        text_len_log = math.log10(max(ch, 1))

        return [ch, nw, ns, np_, awl, nw/ns, nw/np_, ttr, lwr,
                vss, vls, apc, cm, per, ex, qu, co, se, qt, pa, el, da,
                ucr, dr, cwr, nlr, br, nr, hr, cr, bor, ir, lkr, isr,
                punct_variety, sent_cv, trans_rate, fsw, proper_noun_dens,
                hedge_rate, qps, sent_range, text_len_log]


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return X


def build_word_tfidf(cfg: Dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=cfg.get('analyzer', 'word'),
        ngram_range=tuple(cfg.get('ngram_range', [1, 2])),
        max_features=cfg.get('max_features', 100000),
        min_df=cfg.get('min_df', 2),
        max_df=cfg.get('max_df', 0.95),
        sublinear_tf=cfg.get('sublinear_tf', True),
        strip_accents='unicode',
        token_pattern=r'(?u)\b\w+\b',
    )


def build_char_tfidf(cfg: Dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=cfg.get('analyzer', 'char_wb'),
        ngram_range=tuple(cfg.get('ngram_range', [2, 6])),
        max_features=cfg.get('max_features', 100000),
        min_df=cfg.get('min_df', 3),
        max_df=cfg.get('max_df', 0.95),
        sublinear_tf=cfg.get('sublinear_tf', True),
    )


class DenseToSparse(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return sp.csr_matrix(X if isinstance(X, np.ndarray) else np.array(X))


class StyleometricPipeline(BaseEstimator, TransformerMixin):
    """Extractor + MaxAbsScaler + sparse conversion."""
    _FEATURE_NAMES = [
        "ch", "nw", "ns", "np", "awl", "nw_per_s", "nw_per_p", "ttr", "lwr",
        "vss", "vls", "apc", "cm", "per", "ex", "qu", "co", "se", "qt", "pa",
        "el", "da", "ucr", "dr", "cwr", "nlr", "br", "nr", "hr", "cr", "bor",
        "ir", "lkr", "isr", "punct_variety", "sent_cv", "trans_rate",
        "first_sent_words", "proper_noun_density", "hedge_rate",
        "question_per_sent", "sent_range", "text_len_log",
    ]

    def __init__(self):
        self.extractor = StyleometricTransformer()
        self.scaler    = MaxAbsScaler()

    def fit(self, X, y=None):
        self.scaler.fit(self.extractor.transform(X))
        return self

    def transform(self, X, y=None):
        return sp.csr_matrix(self.scaler.transform(self.extractor.transform(X)))

    def get_feature_names_out(self, input_features=None):
        return np.array(self._FEATURE_NAMES)


def build_function_word_tfidf(cfg: Dict) -> TfidfVectorizer:
    """
    TF-IDF restricted to the FUNCTION_WORDS vocabulary.
    Captures style fingerprint (how the author connects ideas)
    rather than topic (what the text is about).
    """
    return TfidfVectorizer(
        analyzer="word",
        vocabulary=FUNCTION_WORDS,
        ngram_range=(1, 1),
        sublinear_tf=cfg.get("sublinear_tf", True),
        token_pattern=r"(?u)\b\w+\b",
    )


def build_feature_union(config: Dict) -> FeatureUnion:
    fc  = config.get("features", {})
    wc  = fc.get("word_tfidf", {})
    cc  = fc.get("char_tfidf", {})
    sc  = fc.get("stylometric", {})
    fw  = fc.get("function_word_tfidf", {})

    trans = [
        ("word_tfidf", build_word_tfidf(wc)),
        ("char_tfidf", build_char_tfidf(cc)),
    ]
    if sc.get("enabled", True):
        trans.append(("stylometric", StyleometricPipeline()))
        logger.info("  Stylometric features: ENABLED (43 features)")
    else:
        logger.info("  Stylometric features: disabled")

    if fw.get("enabled", False):
        trans.append(("function_word_tfidf", build_function_word_tfidf(fw)))
        logger.info(f"  Function-word TF-IDF: ENABLED ({len(FUNCTION_WORDS)} words)")

    return FeatureUnion(transformer_list=trans)
