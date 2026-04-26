"""lm-eval-harness compatible LM adapter for SaintLLM.

Implements the three methods every ``lm_eval.api.model.LM`` subclass
must provide:

* :meth:`SaintLLMHarnessLM.loglikelihood` — score the conditional log-prob
  of a continuation given a context. Used by tasks like HellaSwag,
  ARC-Challenge, MMLU, TruthfulQA.
* :meth:`SaintLLMHarnessLM.loglikelihood_rolling` — total log-prob of a
  full text under the model. Used by perplexity-style benchmarks.
* :meth:`SaintLLMHarnessLM.generate_until` — greedy generation that
  stops on the first occurrence of any ``until`` string. Used by
  open-ended tasks (GSM8K, IFEval).

This class is **standalone** — it does NOT inherit from
``lm_eval.api.model.LM`` so the eval package has no hard dependency on
``lm-eval``. To plug into the real lm-eval-harness, subclass:

    >>> from lm_eval.api.model import LM
    >>> class MyHarnessLM(SaintLLMHarnessLM, LM):
    ...     pass

The methods accept any object with an ``args: tuple`` attribute (the
lm-eval ``Instance`` shape), so a plain dataclass / SimpleNamespace
works for tests.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F
from saint_llm_core.model import SaintLLM
from saint_llm_data.tokenizer import Tokenizer
from saint_llm_inference.generate import greedy_decode
from torch import Tensor


class _Request(Protocol):
    """Duck-type matching ``lm_eval.api.instance.Instance``."""

    args: tuple[Any, ...]


class SaintLLMHarnessLM:
    """Adapter for lm-eval-harness style scoring + generation."""

    def __init__(
        self,
        *,
        model: SaintLLM,
        tokenizer: Tokenizer,
        device: torch.device | str = "cpu",
        max_length: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_length = max_length

    @property
    def eot_token_id(self) -> int:
        """End-of-turn / end-of-text token (matches lm-eval convention)."""
        return self.tokenizer.eos_token_id

    @torch.no_grad()
    def loglikelihood(self, requests: list[_Request]) -> list[tuple[float, bool]]:
        """For each ``(context, continuation)``: ``(logp, is_greedy)``."""
        results: list[tuple[float, bool]] = []
        was_training = self.model.training
        self.model.eval()
        try:
            for req in requests:
                ctx, cont = req.args
                results.append(self._score_pair(str(ctx), str(cont)))
        finally:
            self.model.train(was_training)
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[_Request]) -> list[float]:
        """Total log-probability of each full text under the model."""
        results: list[float] = []
        was_training = self.model.training
        self.model.eval()
        try:
            for req in requests:
                (text,) = req.args
                results.append(self._score_rolling(str(text)))
        finally:
            self.model.train(was_training)
        return results

    @torch.no_grad()
    def generate_until(self, requests: list[_Request]) -> list[str]:
        """Greedy generation stopping at first occurrence of any ``until``."""
        results: list[str] = []
        was_training = self.model.training
        self.model.eval()
        try:
            for req in requests:
                ctx, gen_kwargs = req.args
                if not isinstance(gen_kwargs, dict):
                    gen_kwargs = {}
                results.append(self._generate_one(str(ctx), gen_kwargs))
        finally:
            self.model.train(was_training)
        return results

    # ---- internals ---------------------------------------------------

    def _score_pair(self, context: str, continuation: str) -> tuple[float, bool]:
        ctx_ids = self.tokenizer.encode(context)
        cont_ids = self.tokenizer.encode(continuation)
        if not cont_ids:
            return 0.0, True
        full = ctx_ids + cont_ids
        if len(full) > self.max_length:
            full = full[-self.max_length:]
        n_cont = min(len(cont_ids), len(full) - 1)
        if n_cont <= 0:
            return 0.0, True

        tokens = torch.tensor([full], device=self.device, dtype=torch.long)
        logits = self.model(tokens)["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a Tensor")

        pred_logits = logits[0, -(n_cont + 1) : -1, :]
        target_ids = torch.tensor(full[-n_cont:], device=self.device, dtype=torch.long)
        log_probs = F.log_softmax(pred_logits.to(torch.float32), dim=-1)
        gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        cont_logp = float(gathered.sum().item())

        greedy_ids = log_probs.argmax(dim=-1)
        is_greedy = bool((greedy_ids == target_ids).all().item())
        return cont_logp, is_greedy

    def _score_rolling(self, text: str) -> float:
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            return 0.0
        ids = ids[: self.max_length]
        tokens = torch.tensor([ids], device=self.device, dtype=torch.long)
        logits = self.model(tokens)["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a Tensor")
        pred = logits[0, :-1, :].to(torch.float32)
        log_probs = F.log_softmax(pred, dim=-1)
        targets = torch.tensor(ids[1:], device=self.device, dtype=torch.long)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())

    def _generate_one(self, context: str, gen_kwargs: dict[str, Any]) -> str:
        until: list[str] = list(gen_kwargs.get("until", []) or [])
        max_gen_toks: int = int(gen_kwargs.get("max_gen_toks", 256))

        ctx_ids = self.tokenizer.encode(context)
        ctx_ids = ctx_ids[-self.max_length:]
        prompt = torch.tensor([ctx_ids], device=self.device, dtype=torch.long)

        generated = greedy_decode(self.model, prompt, max_new_tokens=max_gen_toks)
        new_ids = generated[0, len(ctx_ids):].tolist()
        text = self.tokenizer.decode(new_ids)

        for stop in until:
            if not stop:
                continue
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
        return text
