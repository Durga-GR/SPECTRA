import re
import math
import numpy as np


def analyze_text(text):
    """
    Improved AI text detector using multiple linguistic heuristics:

    1. Burstiness: Human writing has high sentence-length variation (bursty).
       AI tends to produce uniformly medium-length sentences.
    2. Perplexity proxy: Rare word usage — AI overuses common/safe words.
    3. Repetition patterns: AI often repeats phrases/structures.
    4. Punctuation diversity: Humans use dashes, ellipses, brackets etc.
    5. Discourse markers: AI overuses certain transition phrases.
    6. Vocabulary richness (type-token ratio on content words).
    """
    text = text.strip()
    words = text.split()

    if len(words) < 20:
        return {"label": "Text Too Short to Analyze", "score": 0.0}

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    if len(sentences) < 2:
        return {"label": "Text Too Short to Analyze", "score": 0.0}

    # ── Feature 1: Burstiness (sentence-length variance) ─────────────────────
    # High variance = human-like; AI tends to cluster around 15-25 words/sentence
    sent_lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(sent_lengths)
    std_len = np.std(sent_lengths)
    # Coefficient of variation
    burstiness = std_len / (mean_len + 1e-6)
    # Human: burstiness > 0.4; AI: < 0.25
    burstiness_score = _clamp((burstiness - 0.20) / 0.40)

    # ── Feature 2: Punctuation richness ──────────────────────────────────────
    # Humans use em-dashes, ellipses, semicolons, parentheses; AI rarely does
    special_punct = len(re.findall(r'[;:\-\—\…\(\)\[\]]', text))
    punct_density = special_punct / max(len(sentences), 1)
    # Human: >1.5 per sentence; AI: <0.5
    punct_score = _clamp((punct_density - 0.3) / 2.0)

    # ── Feature 3: Repeated n-gram penalty ───────────────────────────────────
    # AI repeats 3-gram patterns more than humans
    tokens = re.findall(r'\b\w+\b', text.lower())
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    if trigrams:
        unique_ratio = len(set(trigrams)) / len(trigrams)
    else:
        unique_ratio = 1.0
    # Human: unique_ratio > 0.90; AI: < 0.80
    repetition_score = _clamp((unique_ratio - 0.75) / 0.20)

    # ── Feature 4: AI filler phrases ─────────────────────────────────────────
    # Phrases that AI models tend to overuse
    ai_phrases = [
        r'\bin conclusion\b', r'\bfurthermore\b', r'\bmoreover\b',
        r'\bit is worth noting\b', r'\bit is important to\b',
        r'\bin summary\b', r'\bto summarize\b', r'\boverall\b',
        r'\bin today\'s world\b', r'\bin the realm of\b',
        r'\bdelve into\b', r'\bfacilitate\b', r'\bunderscores?\b',
        r'\bultimately\b', r'\bit is essential\b', r'\bseamlessly\b',
        r'\bcomprehensive\b', r'\brobust\b', r'\binvaluable\b',
        r'\bpivotal\b', r'\blandscape\b', r'\btailored\b',
    ]
    phrase_hits = sum(
        1 for p in ai_phrases if re.search(p, text, re.IGNORECASE)
    )
    # Each hit reduces human probability; normalize over text length
    phrase_density = phrase_hits / max(len(sentences), 1)
    # Human: <0.1 hits/sentence; AI: >0.3
    phrase_penalty = _clamp(1.0 - (phrase_density - 0.05) / 0.40)

    # ── Feature 5: Vocabulary richness (content words) ───────────────────────
    stopwords = set("""
        the a an and or but in on at to for of is are was were be been
        being have has had do does did will would could should may might
        shall this that these those i you he she it we they me him her us them
        my your his its our their what which who how when where why
    """.split())
    content_words = [w for w in tokens if w not in stopwords and len(w) > 2]
    if content_words:
        vocab_richness = len(set(content_words)) / len(content_words)
    else:
        vocab_richness = 0.5
    # Human: >0.70; AI: tends toward 0.55–0.65 (diverse but not too diverse)
    vocab_score = _clamp((vocab_richness - 0.50) / 0.35)

    # ── Weighted combination ──────────────────────────────────────────────────
    weights = [0.30, 0.20, 0.20, 0.20, 0.10]
    human_score = (
        weights[0] * burstiness_score +
        weights[1] * punct_score +
        weights[2] * repetition_score +
        weights[3] * phrase_penalty +
        weights[4] * vocab_score
    )

    threshold = 0.45
    if human_score >= threshold:
        label = "Likely Human Written"
        confidence = 0.50 + human_score * 0.50
    else:
        label = "Likely AI Generated"
        confidence = 0.50 + (1 - human_score) * 0.50

    return {
        "label": label,
        "score": round(_clamp(confidence), 2),
        "details": {
            "burstiness": round(float(burstiness), 3),
            "vocab_richness": round(float(vocab_richness), 3),
            "ai_phrase_hits": phrase_hits,
        }
    }


def _clamp(v):
    return float(max(0.0, min(1.0, v)))
