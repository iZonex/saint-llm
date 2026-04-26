"""v0.0 text data pipeline (D0.0.2 / ADR-0018 / ADR-0019).

Pipeline order per ADR-0019:

    [per-slice loader] → [quality filter] → [LANG-05 MT-pollution] →
    [LSHBloom dedup] → [tokenize] → [pack into windows] → [shard write]

Modules:

    stage           — Stage Protocol + Pipeline composer
    quality         — QualityClassifier Protocol + LengthQualityFilter stub
    mt_pollution    — UK 3-stack MT-pollution detector (URL blocklist + Translationese + KenLM Protocols)
    dedup           — FingerprintDedup (exact, v0.0 baseline) + MinHashDedup (signature exact-match)
    tokenize        — TokenizeStage: encode doc.text into doc.token_ids

Production v0.1+ swaps in:
    - Ultra-FineWeb fastText classifier in QualityFilter
    - Real translationese fastText + KenLM in MT-pollution stack
    - LSHBloom in place of FingerprintDedup at multi-TB scale
"""

from saint_llm_data.pipeline.dedup import (
    FingerprintDedup,
    MinHashDedup,
    minhash_signature,
)
from saint_llm_data.pipeline.mt_pollution import (
    DEFAULT_BLOCKED_DOMAINS,
    PerplexityFilter,
    PerplexityModel,
    TranslationeseDetector,
    TranslationeseFilter,
    URLBlocklistFilter,
)
from saint_llm_data.pipeline.quality import (
    LengthQualityFilter,
    QualityClassifier,
    QualityFilter,
)
from saint_llm_data.pipeline.stage import Document, Pipeline, Stage
from saint_llm_data.pipeline.tokenize import TokenizeStage

__all__ = [
    "DEFAULT_BLOCKED_DOMAINS",
    "Document",
    "FingerprintDedup",
    "LengthQualityFilter",
    "MinHashDedup",
    "PerplexityFilter",
    "PerplexityModel",
    "Pipeline",
    "QualityClassifier",
    "QualityFilter",
    "Stage",
    "TokenizeStage",
    "TranslationeseDetector",
    "TranslationeseFilter",
    "URLBlocklistFilter",
    "minhash_signature",
]
