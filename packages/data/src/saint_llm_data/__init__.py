"""Saint LLM data pipeline.

Modules:
    packing             — Concat-and-chunk sequence packing with EOS separators
                          and segment IDs. (FIM packing, sample-level masking
                          live on top — follow-up modules.)

Planned (not yet implemented):
    pretrain.web        — Web corpus filtering (anti-batched-template, model-collapse mitigation)
    pretrain.code       — Code/math corpora pipeline
    pretrain.long_doc   — Long-document curation (papers, technical reports)
    pretrain.multilang  — Multilingual long-tail corpus
    posttrain.sft       — Specialist domain SFT data (math, code, agent, instruction)
    posttrain.rubric    — Rubric-guided RL data for GRM
    posttrain.tools     — Tool-use trajectories
    tokenizer           — Own BBPE 131K (per AUGMENTATIONS TOK-01..06)
"""

from saint_llm_data.chat_template import (
    ChatTemplate,
    ChatTurn,
    RenderedChat,
    render_chat,
)
from saint_llm_data.multimodal_chat import (
    MultimodalChatTurn,
    RenderedMultimodalChat,
    render_multimodal_chat,
    render_text_chat_to_multimodal,
    to_chat_turn,
)
from saint_llm_data.dataset import TextFileDataset
from saint_llm_data.fertility import (
    FertilityRecord,
    FertilityReport,
    measure_fertility,
    measure_per_language_fertility,
)
from saint_llm_data.hf_dataset import HuggingFaceTextDataset
from saint_llm_data.mix_corpus import CorpusSlice, MixedCorpus
from saint_llm_data.multimodal import (
    MultimodalExample,
    MultimodalPackedBatch,
    TokenSlots,
    encode_multimodal,
    pack_multimodal_examples,
)
from saint_llm_data.packing import PackedBatch, pack_into_batch, pack_sequences
from saint_llm_data.parquet_shards import ParquetShardDataset
from saint_llm_data.tokenizer import CharTokenizer, HFTokenizer, Tokenizer
from saint_llm_data.tokenizer_trainer import (
    DEFAULT_SPECIAL_TOKENS,
    SAINT_V0_0_FORCE_INCLUDE_CHARS,
    SAINT_V0_0_SPECIAL_TOKENS,
    train_bbpe,
)

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_SPECIAL_TOKENS",
    "SAINT_V0_0_FORCE_INCLUDE_CHARS",
    "SAINT_V0_0_SPECIAL_TOKENS",
    "CharTokenizer",
    "ChatTemplate",
    "ChatTurn",
    "CorpusSlice",
    "FertilityRecord",
    "FertilityReport",
    "HFTokenizer",
    "HuggingFaceTextDataset",
    "MixedCorpus",
    "MultimodalChatTurn",
    "MultimodalExample",
    "MultimodalPackedBatch",
    "PackedBatch",
    "ParquetShardDataset",
    "RenderedChat",
    "RenderedMultimodalChat",
    "TextFileDataset",
    "TokenSlots",
    "Tokenizer",
    "encode_multimodal",
    "measure_fertility",
    "measure_per_language_fertility",
    "pack_into_batch",
    "pack_multimodal_examples",
    "pack_sequences",
    "render_chat",
    "render_multimodal_chat",
    "render_text_chat_to_multimodal",
    "to_chat_turn",
    "train_bbpe",
]
