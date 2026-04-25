"""Saint LLM data pipeline.

Modules:
    pretrain.web        — Web corpus filtering (anti-batched-template, model-collapse mitigation)
    pretrain.code       — Code/math corpora pipeline
    pretrain.long_doc   — Long-document curation (papers, technical reports)
    pretrain.multilang  — Multilingual long-tail corpus
    posttrain.sft       — Specialist domain SFT data (math, code, agent, instruction)
    posttrain.rubric    — Rubric-guided RL data for GRM
    posttrain.tools     — Tool-use trajectories
    tokenizer           — DeepSeek-V3 BPE 128K + V4 special tokens (DSML, think, Quick Instruction)
    packing             — FIM packing, sample-level masking, multi-source sequence packing
"""

__version__ = "0.0.1"
