"""Lightweight relevance scoring for paper search.

Query-type-aware matching: classify_group() → type-specific matcher → field_match_score.
Supports multi-group AND queries via "+" syntax.
"""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

# ============================================================
# Constants
# ============================================================

STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "with", "and", "or",
    "by", "from", "via", "using", "based", "towards", "toward",
}

# --- Domain token map: morphological variants → noun stem ---
# Principle: keep nouns as nouns, don't map to verbs.
DOMAIN_TOKEN_MAP = {
    # Broad task morphological variants → noun stem
    "detector": "detect", "detectors": "detect", "detecting": "detect",
    "detected": "detect", "detects": "detect", "detection": "detect",
    "detections": "detect",
    "segment": "segment", "segments": "segment", "segmentation": "segment",
    "segmenting": "segment", "segmented": "segment",
    "classifier": "classify", "classifiers": "classify", "classification": "classify",
    "classifying": "classify", "classified": "classify",
    "recognize": "recognize", "recognizer": "recognize", "recognition": "recognize",
    "recognizing": "recognize", "recognized": "recognize",
    "track": "track", "tracker": "track", "trackers": "track",
    "tracking": "track", "tracked": "track",
    "localize": "localize", "localizer": "localize", "localization": "localize",
    "localisation": "localize", "localizing": "localize", "localized": "localize",
    "generate": "generate", "generator": "generate", "generators": "generate",
    "generation": "generate", "generating": "generate", "generative": "generate",
    "generated": "generate",
    "estimate": "estimate", "estimator": "estimate", "estimators": "estimate",
    "estimation": "estimate", "estimating": "estimate", "estimated": "estimate",
    "predict": "predict", "predictor": "predict", "predictors": "predict",
    "prediction": "predict", "predicting": "predict", "predicted": "predict",
    "reconstruct": "reconstruct", "reconstruction": "reconstruct",
    "reconstructing": "reconstruct", "reconstructed": "reconstruct",
    # Fixed-phrase-head morphological variants → noun stem
    "selector": "select", "selectors": "select", "selection": "select",
    "selecting": "select", "selected": "select", "selects": "select",
    "fuse": "fuse", "fusion": "fuse", "fusing": "fuse",
    "fused": "fuse", "fuses": "fuse",
    "distill": "distill", "distiller": "distill", "distillation": "distill",
    "distilling": "distill", "distilled": "distill",
    "adapt": "adapt", "adapter": "adapt", "adapters": "adapt",
    "adaptation": "adapt", "adapting": "adapt", "adapted": "adapt", "adapts": "adapt",
    "calibrate": "calibrate", "calibrator": "calibrate", "calibration": "calibrate",
    "calibrating": "calibrate", "calibrated": "calibrate",
    "align": "align", "alignment": "align",
    "aligning": "align", "aligned": "align", "aligns": "align",
    "prune": "prune", "pruning": "prune", "pruned": "prune", "prunes": "prune",
    "compress": "compress", "compressor": "compress", "compression": "compress",
    "compressing": "compress", "compressed": "compress",
    "quantize": "quantize", "quantizer": "quantize", "quantization": "quantize",
    "quantizing": "quantize", "quantized": "quantize",
    "retrieve": "retrieve", "retriever": "retrieve", "retrieval": "retrieve",
    "retrieving": "retrieve", "retrieved": "retrieve", "retrieves": "retrieve",
    # Extraction variants (common in CV/ML)
    "extract": "extract", "extractor": "extract", "extraction": "extract",
    "extracting": "extract", "extracted": "extract", "extracts": "extract",
    # Noun forms kept as nouns (NOT stemmed to verbs)
    "conv": "conv", "convs": "conv", "convolution": "conv",
    "convolutions": "conv", "convolutional": "conv",
    "transformer": "transformer", "transformers": "transformer",
    "architecture": "architecture", "architectures": "architecture",
    "representation": "representation", "representations": "representation",
    # General plurals / simple stems
    "object": "object", "objects": "object",
    "category": "category", "categories": "category",
    "image": "image", "images": "image",
    "video": "video", "videos": "video",
    "class": "class", "classes": "class",
    "feature": "feature", "features": "feature",
    "network": "network", "networks": "network",
    "model": "model", "models": "model",
    "layer": "layer", "layers": "layer",
    "dataset": "dataset", "datasets": "dataset",
    "method": "method", "methods": "method",
    "study": "study", "studies": "study",
    "box": "box", "boxes": "box",
    "augment": "augment", "augmentation": "augment",
    "augmenting": "augment", "augmented": "augment",
}

# --- Generic terms (NOT specific/intent-bearing) ---
# Principle: minimal set. Only mark words that truly carry no independent retrieval intent.
# Deliberately excluded: detect, domain, state, vocabulary, semantic, instance,
#   supervised, real, time, end, world, open, feature, selection, fusion, etc.
GENERIC_TERMS = {
    "object", "image", "video", "network", "model", "learning", "vision", "visual",
    "based", "using", "via", "with", "method", "approach", "framework", "system",
    "towards", "toward", "improved", "efficient", "novel", "deep", "neural", "scale",
    "large", "small", "new", "data", "dataset", "scene",
    "zero", "shot", "one", "stage", "two", "step", "multi", "task", "training", "inference",
    "cross", "self", "weakly", "few", "low", "high",
    "level", "aware", "guided", "driven", "enhanced", "robust", "adaptive", "dynamic",
    "spatial", "temporal",
}

# --- Broad task heads: if query's head token maps to one of these, → broad_topic ---
BROAD_TASK_HEADS = {
    "detect", "segment", "classify", "recognize", "track",
    "generate", "estimate", "predict", "reconstruct", "localize",
}

# --- Fixed phrase heads: narrow, specific task/action nouns ---
FIXED_PHRASE_HEADS = {
    "selection", "fusion", "distillation", "adaptation", "calibration",
    "alignment", "pruning", "compression", "quantization", "retrieval",
}

# --- Known model/method names (CV/ML specific, conservative) ---
MODEL_NAMES = {
    "yolo", "detr", "sam", "vit", "clip", "dino", "mamba",
    "fpn", "rcnn", "ssd", "unet", "resnet", "densenet", "swin", "convnext",
}

# --- Acronyms that should NOT be classified as model_name ---
NON_MODEL_ACRONYMS = {
    "ai", "ml", "cv", "nlp", "llm", "vlm", "mllm",
    "rgb", "rgbd", "bev", "lidar", "uav", "sar", "mri", "ct", "xray",
    "2d", "3d",
}

# --- Literal variants: canonical form → {known surface forms} ---
# Deliberately excluded: state-of-the-art/SOTA (too broad, low retrieval value).
LITERAL_VARIANTS = {
    "real-time": {"real-time", "realtime", "real time"},
    "end-to-end": {"end-to-end", "end to end", "e2e"},
    "low-light": {"low-light", "low light"},
    "test-time": {"test-time", "test time"},
    "open-vocabulary": {"open-vocabulary", "open vocabulary", "open-vocab"},
    "open-world": {"open-world", "open world"},
}

# --- Match modes ---
MATCH_THRESHOLDS = {
    "strict": 0.75,
    "normal": 0.58,
    "loose": 0.42,
}

# --- Field weights by group type ---
FIELD_WEIGHTS = {
    "model_name":   {"title": 1.00, "keywords": 0.85, "abstract": 0.55},
    "literal":      {"title": 1.00, "keywords": 0.85, "abstract": 0.55},
    "fixed_phrase": {"title": 1.00, "keywords": 0.80, "abstract": 0.50},
    "broad_topic":  {"title": 1.00, "keywords": 0.85, "abstract": 0.60},
    "phrase":       {"title": 1.00, "keywords": 0.85, "abstract": 0.55},
}


# ============================================================
# Text normalization (unchanged from original)
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_token(token: str) -> str:
    t = token.lower().strip()
    if len(t) < 2:
        return t

    # Model family normalization (preserved for backward compat)
    if t.startswith("yolo") or "yolo" in t:
        return "yolo"
    if t in ("detr", "detrs", "detr-like") or t.startswith("detr"):
        return "detr"
    if t in ("vit", "visiontransformer"):
        return "vit"
    if t in ("sam", "segmentanything"):
        return "sam"
    if t in ("clip",):
        return "clip"
    if t in ("dino",):
        return "dino"
    if t in ("mamba",):
        return "mamba"

    if t in DOMAIN_TOKEN_MAP:
        return DOMAIN_TOKEN_MAP[t]

    # Conservative suffix rules (no er/or deletion)
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 4:
        t = t[:-1]
    if t.endswith("ing") and len(t) >= 5:
        stem = t[:-3]
        if len(stem) >= 3:
            return stem
    if t.endswith("ed") and len(t) >= 5:
        stem = t[:-2]
        if len(stem) >= 3:
            return stem

    return t


def tokenize(text: str) -> list[str]:
    norm = normalize_text(text)
    tokens = norm.split()
    return [normalize_token(t) for t in tokens if len(t) >= 2 and t not in STOPWORDS]


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


# ============================================================
# Legacy functions (kept for compatibility, may be removed later)
# ============================================================

def compact_query(query: str) -> str:
    return normalize_text(query).replace(" ", "")


def is_short_specific_query(query: str) -> bool:
    q = compact_query(query)
    tokens = tokenize(query)
    return len(tokens) == 1 and len(q) <= 8 and q not in GENERIC_TERMS


def is_realtime_query(query: str) -> bool:
    q = normalize_text(query)
    compact = q.replace(" ", "")
    return q in {"real time", "realtime", "rt"} or compact in {"realtime", "rt"}


def _special_real_time_score(text: str) -> float:
    raw = text.lower()
    norm = normalize_text(text)
    if "real-time" in raw or "realtime" in raw or "real time" in norm:
        return 1.0
    if re.search(r"\brt[-\s]?[a-z0-9]", raw):
        return 1.0
    return 0.0


# ============================================================
# Scoring signals (unchanged)
# ============================================================

def phrase_score(query: str, text: str) -> float:
    q = normalize_text(query)
    t = normalize_text(text)
    return 1.0 if (q and q in t) else 0.0


def token_hit(qt: str, text_tokens: set[str]) -> bool:
    if qt in text_tokens:
        return True
    for tt in text_tokens:
        if len(qt) >= 3 and len(tt) >= 3 and qt in tt:
            return True
    return False


def token_coverage_score(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    q_unique = set(q_tokens)
    t_set = set(tokenize(text))
    return sum(1 for qt in q_unique if token_hit(qt, t_set)) / len(q_unique)


def exact_token_field_score(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    if not q_tokens or not text:
        return 0.0
    return 1.0 if token_hit(q_tokens[0], set(tokenize(text))) else 0.0


def query_specific_tokens(query: str) -> set[str]:
    return {t for t in set(tokenize(query)) if t not in GENERIC_TERMS}


def specific_tokens_score(query: str, text: str) -> float:
    specific = query_specific_tokens(query)
    if not specific:
        return 1.0
    t_tokens = set(tokenize(text))
    return sum(1 for t in specific if token_hit(t, t_tokens)) / len(specific)


def fuzzy_score(query: str, text: str) -> float:
    if is_short_specific_query(query):
        return 0.0
    q = normalize_text(query)
    t = normalize_text(text)
    cq = q.replace(" ", "")
    if len(cq) <= 5:
        return 0.0
    if len(q) < 4 or len(t) < 4:
        return 0.0
    partial = fuzz.partial_ratio(q, t) / 100.0
    token_set = fuzz.token_set_ratio(q, t) / 100.0
    return max(partial * 0.9, token_set)


# ============================================================
# Query classification
# ============================================================

def _get_head_token(tokens: list[str]) -> str | None:
    """Return the last token (typically the head noun in English)."""
    return tokens[-1] if tokens else None


def _has_uppercase_char(s: str) -> bool:
    return any(c.isupper() for c in s)


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def _is_camel_case(s: str) -> bool:
    """True if token has mixed case (but not all-uppercase)."""
    return _has_uppercase_char(s) and not s.isupper() and not s.islower()


def classify_group(query: str) -> str:
    """Classify query into: model_name, literal, broad_topic, fixed_phrase, or phrase.

    Priority order: model_name → literal → broad_topic → fixed_phrase → phrase.
    """
    raw = query.strip()
    if not raw:
        return "phrase"

    tokens = tokenize(raw)
    norm = normalize_text(raw)
    compact = norm.replace(" ", "")

    # --- Step 1: model_name detection ---
    # Single token checks
    if len(tokens) == 1:
        t = tokens[0]
        raw_t = raw.strip()

        # All uppercase short token → model_name (unless denylisted)
        if raw_t.isupper() and 2 <= len(raw_t) <= 8:
            if raw_t.lower() not in NON_MODEL_ACRONYMS:
                return "model_name"

        # CamelCase or PascalCase → model_name
        if _is_camel_case(raw_t) and 2 <= len(raw_t) <= 20:
            if raw_t.lower() not in NON_MODEL_ACRONYMS:
                return "model_name"

        # Contains digit (e.g. YOLOv10, ResNet50) → model_name
        if _has_digit(raw_t) and not raw_t.isdigit():
            digit_ratio = sum(1 for c in raw_t if c.isdigit()) / len(raw_t)
            if digit_ratio < 0.5 and raw_t.lower() not in NON_MODEL_ACRONYMS:
                return "model_name"

        # In known model names list
        if t in MODEL_NAMES or raw_t.lower() in MODEL_NAMES:
            return "model_name"

        # Single lowercase word → not model_name (fall through)

    # Multi-token with hyphen and model-like first part
    if "-" in raw and len(tokens) >= 1:
        first_part = raw.split("-")[0].strip()
        if first_part.isupper() and 2 <= len(first_part) <= 8:
            if first_part.lower() not in NON_MODEL_ACRONYMS:
                return "model_name"
        # e.g. YOLO-World, RT-DETR
        if first_part.lower() in MODEL_NAMES:
            return "model_name"

    # --- Step 2: literal detection ---
    # Only if the hyphenated query is a single conceptual unit
    # (no non-hyphenated extra words beyond the hyphenated phrase)
    if "-" in raw:
        # Check if removing hyphens gives one compact token → single concept
        dehyphened = raw.replace("-", " ")
        dehyphened_tokens = dehyphened.split()
        # If all tokens are connected by hyphens (single concept), it's literal
        # e.g. "real-time" (1 hyphenated unit), "end-to-end" (1 unit)
        # but "few-shot object detection" has "object detection" as extra non-hyphenated words
        hyphen_parts_count = len(raw.split("-"))
        non_hyphen_words = len(dehyphened_tokens) - hyphen_parts_count + 1
        # If the query IS just one hyphenated phrase (no extra standalone words), it's literal
        if non_hyphen_words <= 1:
            return "literal"
        # Otherwise fall through (don't classify as literal — check broad_topic/fixed_phrase next)
    # Check known literal patterns (non-hyphenated forms like "real time", "end to end")
    for canonical, variants in LITERAL_VARIANTS.items():
        if norm in variants:
            return "literal"

    # --- Step 3: broad_topic detection (before fixed_phrase) ---
    if len(tokens) >= 2:
        head = _get_head_token(tokens)
        if head and head in BROAD_TASK_HEADS:
            return "broad_topic"

    # --- Step 4: fixed_phrase detection ---
    if len(tokens) >= 2:
        specific_tokens = [t for t in tokens if t not in GENERIC_TERMS]
        if len(specific_tokens) >= 2:
            return "fixed_phrase"
        if len(specific_tokens) >= 1:
            # Check if the original (unstemmed) last specific-ish token is a fixed phrase head
            # We check the original raw tokens against FIXED_PHRASE_HEADS
            raw_tokens = normalize_text(raw).split()
            last_raw = raw_tokens[-1] if raw_tokens else ""
            if last_raw in FIXED_PHRASE_HEADS:
                return "fixed_phrase"
            # Also check stemmed head
            if head and head in {"select", "fuse", "distill", "adapt", "calibrate",
                                  "align", "prune", "compress", "quantize", "retrieve"}:
                return "fixed_phrase"

    # --- Step 5: fallback ---
    return "phrase"


# ============================================================
# Type-specific matchers
# ============================================================

def model_name_match(query: str, text: str) -> float:
    """Precision-first matching for model/method names. No fuzzy."""
    if not query or not text:
        return 0.0

    q_norm = normalize_text(query)
    t_norm = normalize_text(text)
    q_compact = q_norm.replace(" ", "")
    t_tokens = set(tokenize(text))
    t_raw = text.lower()

    # For hyphenated model names (e.g. RT-DETR), check each part
    if "-" in query:
        parts = [p.strip().lower() for p in query.split("-") if p.strip()]
        all_hit = all(
            any(p in tt or tt in p for tt in t_tokens if len(p) >= 2 and len(tt) >= 2)
            for p in parts
        )
        if all_hit:
            # Exact full match
            if q_compact in t_norm.replace(" ", ""):
                return 1.0
            return 0.90
        # Partial: at least the first part (the model family) matches
        if parts:
            first = parts[0]
            if any(first in tt or tt.startswith(first) for tt in t_tokens if len(tt) >= 2):
                return 0.75
        return 0.0

    # Single-token model name
    q_lower = q_norm.lower()
    is_short = len(q_compact) <= 3

    for tt in t_tokens:
        # Exact normalized match
        if tt == q_lower:
            return 1.0
        # Prefix match: query is prefix of a longer token (YOLO → YOLOv10)
        if len(tt) > len(q_lower) and tt.startswith(q_lower):
            # For short tokens, only allow prefix if the target looks like a
            # model variant (contains digit, mixed case, or is itself a known
            # model name). Prevents "SAM" prefix-matching "sample", "ViT" matching "vital".
            if is_short:
                if _has_digit(tt) or not tt.islower() or tt in MODEL_NAMES:
                    return 0.90
                continue
            return 0.90
        # Contains match: query inside longer token (YOLO → SpikingYOLOX)
        # For short tokens (≤3 chars), skip contains to avoid false matches (SAM in sample)
        if not is_short and len(tt) > len(q_lower) and q_lower in tt:
            return 0.80

    # Raw-text check for short model names: look for camelCase/PascalCase prefix
    # e.g. "ViT" should match "ViTDet" (has interior uppercase) but not "vital" or "Sample"
    if is_short:
        raw_words = re.findall(r'\b\w+\b', text)
        for rw in raw_words:
            if rw.lower().startswith(q_lower) and rw.lower() != q_lower:
                # Must have an uppercase letter beyond the first character
                # to distinguish ViTDet (camelCase) from Sample (title case)
                if any(c.isupper() for c in rw[1:]):
                    return 0.90

    # Fallback: check raw text for exact phrase
    # For short tokens, skip substring fallback to avoid false matches
    # (SAM in "sample", ViT in "vital", etc.)
    if not is_short and q_norm in t_norm:
        return 0.80

    return 0.0


def literal_match(query: str, text: str) -> float:
    """Literal phrase matching with known variant expansion. No semantic expansion. No fuzzy."""
    if not query or not text:
        return 0.0

    q_norm = normalize_text(query)
    t_norm = normalize_text(text)
    t_raw = text.lower()

    # Build the set of variant forms to check
    variants: set[str] = set()
    # Always include the normalized form
    variants.add(q_norm)
    variants.add(q_norm.replace("-", ""))
    variants.add(q_norm.replace("-", " "))

    # Check against known literal variants
    for canonical, surface_forms in LITERAL_VARIANTS.items():
        if q_norm in surface_forms or q_norm.replace("-", " ") in surface_forms:
            variants.update(surface_forms)
            # For real-time specifically, also check RT- prefix in raw text
            if "real-time" in canonical:
                if re.search(r"\brt[-\s]?[a-z0-9]", t_raw):
                    return 1.0
            break

    # Check for exact variant match in text
    for v in variants:
        v_compact = v.replace(" ", "")
        t_compact = t_norm.replace(" ", "")
        if v_compact in t_compact:
            return 1.0
        if v in t_norm:
            return 1.0

    # Check all tokens present
    q_tokens = tokenize(query)
    if q_tokens:
        coverage = token_coverage_score(query, text)
        if coverage >= 1.0:
            return 0.85
        # Partial coverage capped low — literal phrases need all tokens
        return min(coverage, 0.50)

    return 0.0


def fixed_phrase_match(query: str, text: str) -> float:
    """High-precision matching for fixed terminology. Full phrase > all tokens > high coverage."""
    if not query or not text:
        return 0.0

    # Full phrase match
    if phrase_score(query, text) >= 1.0:
        return 1.0

    coverage = token_coverage_score(query, text)
    if coverage >= 1.0:
        return 0.85
    if coverage >= 0.75:
        return 0.75
    return min(coverage, 0.40)


def broad_topic_match(query: str, text: str) -> float:
    """Recall-oriented matching for broad research topics. Allows stemming and light fuzzy."""
    if not query or not text:
        return 0.0

    # Exact normalized phrase
    if phrase_score(query, text) >= 1.0:
        return 1.0

    coverage = token_coverage_score(query, text)
    if coverage >= 1.0:
        return 0.90
    if coverage >= 0.80:
        return 0.80
    if coverage >= 0.60:
        return 0.65

    # Light fuzzy for longer queries as a fallback
    q_norm = normalize_text(query)
    if len(q_norm) > 12:
        fuzzy = fuzzy_score(query, text)
        if fuzzy > 0:
            return max(min(coverage, 0.45), fuzzy * 0.85)

    return min(coverage, 0.45)


def phrase_match(query: str, text: str) -> float:
    """Generic matcher for unclassified queries. Between fixed_phrase and broad_topic in strictness."""
    if not query or not text:
        return 0.0

    # Exact phrase
    if phrase_score(query, text) >= 1.0:
        return 1.0

    coverage = token_coverage_score(query, text)
    if coverage >= 1.0:
        return 0.85
    if coverage >= 0.75:
        return 0.75
    if coverage >= 0.50:
        base = 0.60

        # Guard: if specific tokens are missing, cap lower
        specific = query_specific_tokens(query)
        if specific:
            sp_score = specific_tokens_score(query, text)
            if sp_score == 0.0:
                base = min(base, 0.50)
            elif sp_score < 1.0 and len(specific) >= 2:
                base = min(base, 0.68)
        return base

    # Moderate fuzzy for longer queries
    q_norm = normalize_text(query)
    if len(q_norm) > 10:
        fuzzy = fuzzy_score(query, text)
        if fuzzy > 0:
            return max(min(coverage, 0.40), fuzzy * 0.80)

    return min(coverage, 0.40)


# ============================================================
# Unified field-level matching entry point
# ============================================================

def field_match_score(query: str, text: str) -> float:
    """Score a single query against a single text field, dispatched by group type."""
    if not query or not text:
        return 0.0

    gtype = classify_group(query)

    if gtype == "model_name":
        return model_name_match(query, text)
    elif gtype == "literal":
        return literal_match(query, text)
    elif gtype == "fixed_phrase":
        return fixed_phrase_match(query, text)
    elif gtype == "broad_topic":
        return broad_topic_match(query, text)
    else:
        return phrase_match(query, text)


# ============================================================
# Paper-level scoring
# ============================================================

@dataclass
class MatchResult:
    matched: bool
    score: float
    reason: str  # "title" | "keywords" | "abstract" | "none"
    matched_query: str | None = None


def score_paper(
    query: str,
    paper,
    threshold: float | None = None,
    mode: str = "normal",
) -> MatchResult:
    """Score a single query group against a paper.

    Uses group-type-aware field weights. Authors field does NOT participate
    in keyword matching (author search is handled separately via --author).
    """
    thresh = threshold if threshold is not None else MATCH_THRESHOLDS.get(mode, 0.58)

    title_text = _safe_str(paper.title)
    keywords_text = _safe_str(" ".join(paper.keywords)) if paper.keywords else ""
    abstract_text = _safe_str(paper.abstract)

    group_type = classify_group(query)
    weights = FIELD_WEIGHTS.get(group_type, FIELD_WEIGHTS["phrase"])

    title_s = field_match_score(query, title_text) * weights["title"]
    keywords_s = field_match_score(query, keywords_text) * weights["keywords"] if keywords_text else 0.0
    abstract_s = field_match_score(query, abstract_text) * weights["abstract"] if abstract_text else 0.0

    scores = [(title_s, "title"), (keywords_s, "keywords"), (abstract_s, "abstract")]
    best_s, reason = max(scores, key=lambda x: x[0])
    if best_s == 0.0:
        reason = "none"

    return MatchResult(matched=best_s >= thresh, score=round(best_s, 4), reason=reason, matched_query=query)


# ============================================================
# Author matching (unchanged)
# ============================================================

def author_match_score(author_query: str, paper) -> float:
    """1.0 if all author query tokens appear in paper authors (order-independent)."""
    query_tokens = set(tokenize(author_query))
    if not query_tokens:
        return 0.0
    authors_text = _safe_str(" ".join(paper.authors)) if paper.authors else ""
    if not authors_text:
        return 0.0
    author_tokens = set(tokenize(authors_text))
    return 1.0 if all(token_hit(qt, author_tokens) for qt in query_tokens) else 0.0


# ============================================================
# Multi-group AND queries (unchanged)
# ============================================================

def parse_query_groups(query: str) -> list[str]:
    return [part.strip() for part in query.split("+") if part.strip()]


def is_and_query(query: str) -> bool:
    return "+" in query


@dataclass
class QueryMatchResult:
    matched: bool
    score: float
    reason: str
    matched_query: str
    group_scores: dict[str, float]
    group_reasons: dict[str, str]
    missing_groups: list[str]


def score_paper_query(
    query: str,
    paper,
    threshold: float | None = None,
    mode: str = "normal",
) -> QueryMatchResult:
    groups = parse_query_groups(query)

    if len(groups) == 1:
        r = score_paper(groups[0], paper, threshold=threshold, mode=mode)
        return QueryMatchResult(
            matched=r.matched, score=r.score, reason=r.reason,
            matched_query=query,
            group_scores={groups[0]: r.score}, group_reasons={groups[0]: r.reason},
            missing_groups=[] if r.matched else [groups[0]],
        )

    # AND query: compute effective group threshold
    is_and = len(groups) > 1
    base_threshold = threshold if threshold is not None else MATCH_THRESHOLDS.get(mode, 0.58)
    if is_and and threshold is None:
        group_threshold = max(base_threshold, MATCH_THRESHOLDS["normal"])
    else:
        group_threshold = base_threshold

    group_results: dict[str, float] = {}
    group_reasons: dict[str, str] = {}
    missing: list[str] = []

    for g in groups:
        r = score_paper(g, paper, threshold=group_threshold, mode=mode)
        group_results[g] = r.score
        group_reasons[g] = r.reason

        # Reject weak capped matches in AND mode (score <= 0.50 is always weak)
        if is_and and r.score <= 0.50:
            missing.append(f"{g}(weak)")
            continue

        # In normal-mode AND queries, reject abstract-only weak matches
        effective_mode = mode if threshold is None else "custom"
        if is_and and effective_mode == "normal":
            if r.reason == "abstract" and r.score < 0.75:
                missing.append(f"{g}(abstract-weak)")
                continue
            if r.reason == "keywords" and r.score < 0.70:
                missing.append(f"{g}(keywords-weak)")
                continue

        if not r.matched:
            missing.append(g)

    if missing:
        return QueryMatchResult(
            matched=False, score=0.0, reason="missing:" + ",".join(missing),
            matched_query=query, group_scores=group_results,
            group_reasons=group_reasons, missing_groups=missing,
        )

    return QueryMatchResult(
        matched=True, score=round(min(group_results.values()), 4),
        reason="+".join(group_reasons[g] for g in groups),
        matched_query=query, group_scores=group_results,
        group_reasons=group_reasons, missing_groups=[],
    )


_REASON_WEIGHT = {"title": 1.0, "keywords": 0.85, "abstract": 0.6, "authors": 0.4, "none": 0.0}


def is_strong_group_match(score: float, reason: str) -> bool:
    if reason == "none" or score <= 0.50:
        return False
    if reason == "abstract" and score < 0.75:
        return False
    return True


def near_miss_score(result: QueryMatchResult) -> float:
    if result.matched:
        return 0.0
    groups = list(result.group_scores.keys())
    if not groups:
        return 0.0

    weighted_sum = 0.0
    strong_count = 0
    for g in groups:
        s = result.group_scores.get(g, 0.0)
        r = result.group_reasons.get(g, "none")
        if is_strong_group_match(s, r):
            w = s * _REASON_WEIGHT.get(r, 0.5)
            if classify_group(g) == "fixed_phrase":
                w *= 1.15
            weighted_sum += w
            strong_count += 1

    if strong_count == 0:
        return 0.0
    return round(weighted_sum / len(groups), 4)
