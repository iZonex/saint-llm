"""Saint LLM evaluation harness.

Modules:
    perplexity  — causal-LM PPL on token-id batches (single + streaming variants)

Suites (planned):
    knowledge   — MMLU/MMLU-Pro/MMLU-Redux/MMMLU/C-Eval/CMMLU/MultiLoKo/SimpleQA-V/SuperGPQA/FACTS/TriviaQA/AGIEval
    reasoning   — BBH/DROP/HellaSwag/CLUEWSC/WinoGrande/GPQA-D/HLE/HMMT/IMOAnswerBench/Apex
    code_math   — BigCodeBench/HumanEval/GSM8K/MATH/MGSM/CMath/LiveCodeBench/Codeforces/PutnamBench
    long_ctx    — LongBench-V2/MRCR-1M/CorpusQA
    agent       — Terminal-Bench-2.0/SWE-{Verified,Pro,Multilingual}/BrowseComp/MCPAtlas/Tool-Decathlon/GDPval-AA
"""

from saint_llm_eval.perplexity import compute_perplexity, compute_perplexity_streaming

__version__ = "0.0.1"

__all__ = ["compute_perplexity", "compute_perplexity_streaming"]
