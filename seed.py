"""
seed.py -- turns a raw topic string into (a) a handful of representative
keywords and (b) a single sentiment bias used to initialize the swarm's
opinions.

WHY TF-IDF for a single string: scikit-learn's TfidfVectorizer is normally
used across a *corpus* of documents, where IDF down-weights words that
appear in every document. Here we only have one topic string, so IDF is
constant across terms and TF-IDF collapses to (smoothed) term frequency --
we use it anyway because it's the standard, inspectable tool for "rank
these words by importance" and keeps the pipeline honest about what it's
doing (no hand-rolled word counting, no stop-word list to maintain
ourselves). If this project grew to seed multiple topics at once, the same
TfidfVectorizer call would start doing real cross-document weighting for
free.

WHY a hardcoded sentiment word list instead of a sentiment model: the goal
is a rule you can defend on a whiteboard, not the most accurate sentiment
classifier. A small positive/negative lexicon is transparent, has zero
external dependencies, and is enough to put a directional bias on the
topic.

WHY MOST TOPICS END UP "NEUTRAL" (bias exactly 0.0), AND WHY THAT'S A
REAL LIMITATION, NOT A BUG: bias is computed only from the topic's top-5
TF-IDF keywords, and only counts if one of those keywords is an *exact*
match in POSITIVE_WORDS/NEGATIVE_WORDS. A topic like "quarterly
performance review process" has zero matches -- "performance" and
"review" are sentiment-neutral in isolation, even though the topic
*could* carry sentiment in context ("performance review" often reads as
stressful). This is the honest cost of a lexicon instead of a real
sentiment model: it can only ever recognize words it was explicitly
given, never context. When bias lands at 0.0, every agent seeds near 0,
the DeGroot-weighted update rule then converges the whole population
toward that same near-0 consensus (the rule doesn't manufacture
sentiment from nothing), and since evaluate.py's neutral band is
+-0.15, a converged population sitting at ~0 reads as "mostly neutral"
almost by construction. The fix that's actually in scope for this
project's simplicity constraint is a bigger, more general lexicon (see
below) -- there's no amount of tuning LEAN_THRESHOLD that fixes "we never
saw a word we recognize" being fed to the model.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

POSITIVE_WORDS = {
    "good", "great", "flexible", "flexibility", "support", "supportive",
    "benefit", "benefits", "improve", "improved", "improvement", "trust",
    "productive", "productivity", "happy", "positive", "welcome", "win",
    "success", "successful", "better", "beneficial", "convenient",
    "convenience", "empower", "empowering", "opportunity", "balance",
    "wellbeing", "efficient", "efficiency", "modern", "progressive",
    # Broader, domain-general positive adjectives/nouns -- added because
    # the original ~30-word list was heavily workplace-flavored and
    # missed most everyday topics entirely (see the "why is everything
    # neutral" note below).
    "excellent", "amazing", "wonderful", "outstanding", "innovative",
    "innovation", "boost", "boosts", "boosted", "growth", "thrive",
    "thriving", "safe", "safety", "secure", "security", "fair",
    "fairness", "inclusive", "inclusion", "affordable", "sustainable",
    "sustainability", "healthy", "helpful", "gain", "gains", "advance",
    "advancement", "upgrade", "upgraded", "transparent", "transparency",
    "collaborative", "collaboration", "respect", "respectful",
    "reliable", "reliability", "streamline", "streamlined", "valuable",
    "encouraging", "reward", "rewarding", "saving", "savings",
}

NEGATIVE_WORDS = {
    "bad", "worse", "worst", "mandatory", "forced", "force", "strict",
    "rigid", "restrict", "restriction", "restrictive", "commute",
    "commuting", "burden", "burnout", "stress", "stressful", "unfair",
    "distrust", "surveillance", "micromanage", "micromanagement", "cut",
    "cuts", "cutting", "layoff", "layoffs", "backward", "outdated",
    "inconvenient", "inflexible", "controlling", "punitive", "loss",
    # Same broadening as POSITIVE_WORDS above.
    "harm", "harmful", "dangerous", "risky", "threat", "threatening",
    "failure", "crisis", "declining", "worrying", "concerning",
    "frustrating", "expensive", "costly", "wasteful", "chaotic", "toxic",
    "hostile", "unsafe", "insecure", "unreliable", "broken", "disruptive",
    "punishment", "excessive", "invasive", "exploitative", "unjust",
    "injustice", "corrupt", "corruption", "greedy",
}


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract the top_n keywords from `text` by TF-IDF weight.

    Args:
        text: the raw topic string, e.g. "new company policy on remote work".
        top_n: how many keywords to return.

    Returns:
        Up to top_n lowercase keyword strings, ordered highest-weight first.
        Returns an empty list if `text` has no scorable tokens (e.g. it's
        entirely stop words).
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
    except ValueError:
        # Happens when `text` contains only stop words / no tokens at all.
        return []

    scores = tfidf_matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    ranked = sorted(zip(feature_names, scores), key=lambda pair: pair[1], reverse=True)
    return [word for word, score in ranked[:top_n] if score > 0]


def compute_bias(keywords: list[str]) -> float:
    """Turn a list of keywords into a single scalar opinion bias in [-1, 1].

    WHY this formula: bias = (positive_hits - negative_hits) / len(keywords).
    It's a plain net-sentiment ratio -- simple enough to compute by hand
    from the keyword list, and bounded in [-1, 1] by construction since a
    keyword contributes at most +1 or -1 and the denominator is the
    keyword count. Keywords matching neither list contribute 0 (they're
    treated as neutral, not dropped -- they still count toward the
    denominator, so a topic full of neutral words pulls the bias toward 0
    rather than amplifying whatever few charged words happen to be there).

    Args:
        keywords: output of extract_keywords().

    Returns:
        0.0 if `keywords` is empty (no information -> neutral prior).
    """
    if not keywords:
        return 0.0

    pos_hits = sum(1 for kw in keywords if kw in POSITIVE_WORDS)
    neg_hits = sum(1 for kw in keywords if kw in NEGATIVE_WORDS)
    return (pos_hits - neg_hits) / len(keywords)


def seed_from_topic(topic: str, top_n: int = 5) -> tuple[list[str], float]:
    """Convenience wrapper: topic string -> (keywords, bias).

    This is the function simulate.py / main.py actually call; it exists so
    callers don't need to know that keyword extraction and bias
    computation are two separate steps.
    """
    keywords = extract_keywords(topic, top_n=top_n)
    bias = compute_bias(keywords)
    return keywords, bias
