"""
features.py -- Feature engineering pipeline.
45 hand-crafted stylometric features + word/char/function-word TF-IDF.
All fitted INSIDE CV folds -- no leakage.
"""
import math
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
    45 features.
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
    # DS/Grok encyclopedic discriminators (Run 12)
    _DEF_RE  = re.compile(r'^[A-Z][^.!?\n]{0,80}\b(is|are|was|were)\b', re.IGNORECASE)
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([self._f(t) for t in X], dtype=np.float32)

    def _f(self, text):
        if not text or not isinstance(text, str):
            return [0.0] * 45
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
        text_len_log = math.log10(max(ch, 1))

        # Run 10 recovery: these two features were documented in the experiment log
        # but had drifted out of the current code.
        starts_with_the = 1.0 if text.lstrip().lower().startswith("the ") else 0.0
        clause_per_sent = text.count(',') / ns

        return [ch, nw, ns, np_, awl, nw/ns, nw/np_, ttr, lwr,
                vss, vls, apc, cm, per, ex, qu, co, se, qt, pa, el, da,
                ucr, dr, cwr, nlr, br, nr, hr, cr, bor, ir, lkr, isr,
                punct_variety, sent_cv, trans_rate, fsw, proper_noun_dens,
                hedge_rate, qps, sent_range, text_len_log,
                starts_with_the, clause_per_sent]


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
    """Extractor + MaxAbsScaler + sparse conversion. 45 features."""
    _FEATURE_NAMES = [
        "ch", "nw", "ns", "np", "awl", "nw_per_s", "nw_per_p", "ttr", "lwr",
        "vss", "vls", "apc", "cm", "per", "ex", "qu", "co", "se", "qt", "pa",
        "el", "da", "ucr", "dr", "cwr", "nlr", "br", "nr", "hr", "cr", "bor",
        "ir", "lkr", "isr", "punct_variety", "sent_cv", "trans_rate",
        "first_sent_words", "proper_noun_density", "hedge_rate",
        "question_per_sent", "sent_range", "text_len_log",
        "starts_with_the", "clause_per_sent",
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


class FunctionWordAnalyzer:
    """
    Callable analyzer that extracts n-grams composed only of function words.

    Unlike vocabulary=FUNCTION_WORDS (unigram-only), this supports bigrams and
    trigrams of function words: e.g., "however , the", "it is a", "not only but".
    These transition n-grams are strong LLM style fingerprints.

    Serializable (class with __call__) — safe for joblib/pickle.
    """

    def __init__(self, ngram_range=(1, 2)):
        self.ngram_range = ngram_range
        self._fw_set = frozenset(FUNCTION_WORDS)
        self._tok_re = re.compile(r'(?u)\b\w+\b')

    def __call__(self, text: str):
        tokens = [t for t in self._tok_re.findall(text.lower()) if t in self._fw_set]
        min_n, max_n = self.ngram_range
        grams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                grams.append(' '.join(tokens[i:i + n]))
        return grams


def build_function_word_tfidf(cfg: Dict) -> TfidfVectorizer:
    """
    TF-IDF over function words.

    ngram_range=[1,1] (default / proven): uses vocabulary=FUNCTION_WORDS — a fixed
    151-word unigram vocabulary.  This is the Run 10 configuration (CV 0.9393).
    Stable, no overfitting, captures LLM style fingerprint cleanly.

    ngram_range=[1,N] for N>1: uses FunctionWordAnalyzer to generate bigrams/trigrams
    of function words.  Higher capacity but more prone to overfit on 2400 samples
    (Run 12 regression: 0.9393 → 0.9339 when switched to [1,2]).
    """
    ngram_range = tuple(cfg.get("ngram_range", [1, 1]))

    if ngram_range == (1, 1):
        # Proven unigram approach: fixed vocabulary, no max_features ceiling,
        # exact 151 features — numerically stable across all CV folds.
        return TfidfVectorizer(
            analyzer='word',
            vocabulary=FUNCTION_WORDS,
            sublinear_tf=cfg.get("sublinear_tf", True),
        )

    # Bigrams/trigrams: FunctionWordAnalyzer with max_features cap
    return TfidfVectorizer(
        analyzer=FunctionWordAnalyzer(ngram_range),
        max_features=cfg.get("max_features", 5000),
        min_df=cfg.get("min_df", 2),
        sublinear_tf=cfg.get("sublinear_tf", True),
    )


class DSGrokSubspaceTfidf(BaseEstimator, TransformerMixin):
    """
    Word TF-IDF fitted ONLY on DeepSeek (class 1) + Grok (class 2) training samples.

    Creates a topic-neutral vocabulary tailored to the DS/Grok decision boundary.
    By training only on these two classes the vocabulary is free of signal from
    Human/Claude/Gemini/ChatGPT text, giving the DS/Grok discriminator a clean slate.

    During fit  : filters to y==1|2, fits TF-IDF on those ~240 samples only.
    During transform: applies to ALL samples (zero activations for clearly non-DS/Grok
                      texts are intentional — other classes look "foreign" to this TF-IDF).
    """

    _DEEPSEEK = 1
    _GROK = 2

    def __init__(self, max_features: int = 10000, ngram_range=(1, 3), min_df: int = 1):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

    def fit(self, X, y=None):
        if y is not None:
            y_arr = np.array(y)
            mask = (y_arr == self._DEEPSEEK) | (y_arr == self._GROK)
            X_dg = [X[i] for i in range(len(X)) if mask[i]]
        else:
            X_dg = list(X)  # fallback: fit on all (shouldn't happen in CV)

        if len(X_dg) < 4:
            X_dg = list(X)  # safety fallback

        self.tfidf_ = TfidfVectorizer(
            analyzer='word',
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b',
        )
        self.tfidf_.fit(X_dg)
        return self

    def transform(self, X, y=None):
        return self.tfidf_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.tfidf_.get_feature_names_out()


class DelexTfidfVectorizer(BaseEstimator, TransformerMixin):
    """
    Word TF-IDF on delexicalized text: digit sequences replaced with __NUM__.

    Reduces topic leakage from specific numbers/years (a factual text about
    "World War 2" and one about "the Silk Road" both become less distinguishable
    by their numbers, shifting the model toward structural n-gram patterns).

    The __NUM__ token itself becomes a feature capturing "number density" and
    "position of numbers in text structure" — both useful for DS/Grok discrimination.
    """

    _NUM_RE = re.compile(r'\b\d+\b')

    def __init__(self, max_features: int = 30000, ngram_range=(1, 2), min_df: int = 2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

    def _delex(self, text: str) -> str:
        return self._NUM_RE.sub('__NUM__', text)

    def fit(self, X, y=None):
        self.tfidf_ = TfidfVectorizer(
            analyzer='word',
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b',  # \w matches _ so __NUM__ is a valid token
        )
        self.tfidf_.fit([self._delex(t) for t in X])
        return self

    def transform(self, X, y=None):
        return self.tfidf_.transform([self._delex(t) for t in X])

    def get_feature_names_out(self, input_features=None):
        return self.tfidf_.get_feature_names_out()


def build_char_tfidf_micro(cfg: Dict) -> TfidfVectorizer:
    """
    Second character-level TF-IDF with `analyzer='char'` (no word-boundary padding)
    and a tighter n-gram range (3,7).

    Captures micro punctuation/spacing patterns that distinguish DS/Grok short
    factual texts — e.g., ". " transitions, comma patterns, "The " starts.
    Complements build_char_tfidf (char_wb range 2-6).
    """
    return TfidfVectorizer(
        analyzer=cfg.get("analyzer", "char"),
        ngram_range=tuple(cfg.get("ngram_range", [3, 7])),
        max_features=cfg.get("max_features", 20000),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.95),
        sublinear_tf=cfg.get("sublinear_tf", True),
    )


def build_feature_union(config: Dict) -> FeatureUnion:
    fc  = config.get("features", {})
    wc  = fc.get("word_tfidf", {})
    cc  = fc.get("char_tfidf", {})
    mc  = fc.get("char_tfidf_micro", {})
    sc  = fc.get("stylometric", {})
    fw  = fc.get("function_word_tfidf", {})
    dgc = fc.get("ds_grok_tfidf", {})
    dlc = fc.get("delex_tfidf", {})

    trans = [
        ("word_tfidf", build_word_tfidf(wc)),
        ("char_tfidf", build_char_tfidf(cc)),
    ]
    if mc.get("enabled", True):
        trans.append(("char_tfidf_micro", build_char_tfidf_micro(mc)))
        logger.info(f"  Char micro TF-IDF (3,7): ENABLED ({mc.get('max_features', 20000)} features)")
    else:
        logger.info("  Char micro TF-IDF: disabled")

    if sc.get("enabled", True):
        trans.append(("stylometric", StyleometricPipeline()))
        logger.info("  Stylometric features: ENABLED (45 features)")
    else:
        logger.info("  Stylometric features: disabled")

    if fw.get("enabled", False):
        ngr = fw.get("ngram_range", [1, 2])
        trans.append(("function_word_tfidf", build_function_word_tfidf(fw)))
        logger.info(f"  Function-word TF-IDF: ENABLED (ngram {ngr}, {fw.get('max_features', 5000)} features)")

    if dgc.get("enabled", False):
        ngr = dgc.get("ngram_range", [1, 3])
        mf  = dgc.get("max_features", 10000)
        trans.append(("ds_grok_tfidf", DSGrokSubspaceTfidf(
            max_features=mf,
            ngram_range=tuple(ngr),
            min_df=dgc.get("min_df", 1),
        )))
        logger.info(f"  DS/Grok subspace TF-IDF: ENABLED (ngram {ngr}, {mf} features)")

    if dlc.get("enabled", False):
        ngr = dlc.get("ngram_range", [1, 2])
        mf  = dlc.get("max_features", 30000)
        trans.append(("delex_tfidf", DelexTfidfVectorizer(
            max_features=mf,
            ngram_range=tuple(ngr),
            min_df=dlc.get("min_df", 2),
        )))
        logger.info(f"  Delex TF-IDF: ENABLED (ngram {ngr}, {mf} features)")

    return FeatureUnion(transformer_list=trans)
