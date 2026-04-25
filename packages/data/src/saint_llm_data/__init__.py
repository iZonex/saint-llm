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

from saint_llm_data.packing import PackedBatch, pack_into_batch, pack_sequences

__version__ = "0.0.1"

__all__ = ["PackedBatch", "pack_into_batch", "pack_sequences"]
