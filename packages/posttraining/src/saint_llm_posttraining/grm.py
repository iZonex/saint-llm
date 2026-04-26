"""GRM — Generative Reward Model (LM-as-judge).

A *Generative Reward Model* uses an LM to score a (prompt, completion)
pair by reading the conversation and emitting a verdict. We don't
parse free-form judge text — that's brittle. Instead we extract the
softmax probability the judge places on a small set of *score tokens*
at a fixed position right after the prompt. Two common shapes:

* **Binary judge** — score tokens ``["good", "bad"]``, score values
  ``(1.0, 0.0)``. Reward = ``p(good)``. Equivalent to a sigmoid head
  but stays in the LM's vocabulary.
* **Likert judge** — score tokens ``["1", "2", "3", "4", "5"]``,
  score values ``(0.0, 0.25, 0.5, 0.75, 1.0)``. Reward = expected
  value over the Likert distribution. Smoother gradient, less
  collapse than argmax.

This module ships the math (judge + extract + expected-value reduction)
and an adapter (:class:`GRMRewardFn`) that satisfies the
:class:`RewardFn` Protocol consumed by :mod:`grpo_trainer` and
:mod:`tree_grpo_trainer`. A mock judge in the tests verifies the
extraction without spinning up a real LM.

The judge is treated as a frozen scorer at inference; its parameters
are not differentiated through. Training the judge itself is a
separate recipe (e.g. preference SFT on chosen/rejected pairs); this
module only consumes a trained judge.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_BINARY_TEMPLATE = (
    "Prompt:\n{prompt}\n\nResponse:\n{completion}\n\n"
    "Was the response correct? Answer with one word: "
)


@dataclass(frozen=True)
class GRMConfig:
    """Generative reward model configuration.

    Attributes:
        prompt_template: format string with ``{prompt}`` and
            ``{completion}`` placeholders. The judge sees this exact
            text rendered, then is scored on the token positions
            immediately after.
        score_token_ids: tuple of vocab IDs the judge picks among.
            For binary ``("good", "bad")``; for Likert
            ``("1", "2", "3", "4", "5")``. The judge's softmax over
            these specific IDs (renormalized) defines the score
            distribution.
        score_values:    parallel float tuple. ``score_values[k]`` is
            the scalar reward when the judge fully chooses
            ``score_token_ids[k]``. The reward returned by
            :func:`judge_completion` is the expected value:
            ``sum_k p_k * score_values[k]``.
        temperature:     softmax temperature applied to the judge's
            logits over the score-token subset. Default 1.0.
    """

    prompt_template: str
    score_token_ids: tuple[int, ...]
    score_values: tuple[float, ...]
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if len(self.score_token_ids) != len(self.score_values):
            raise ValueError(
                "score_token_ids and score_values must have the same length; "
                f"got {len(self.score_token_ids)} vs {len(self.score_values)}",
            )
        if not self.score_token_ids:
            raise ValueError("at least one score token is required")
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive; got {self.temperature}",
            )


def judge_completion(
    judge: torch.nn.Module,
    tokens: Tensor,
    cfg: GRMConfig,
) -> Tensor:
    """Score a batch of fully-rendered judge prompts.

    Args:
        judge:  the judge LM. Called as ``judge(tokens)`` returning
            ``{"logits": (B, T, V)}``. Expected to be in eval mode and
            wrapped in :func:`torch.no_grad` by the caller (we don't
            modify autograd state here).
        tokens: ``(B, T)`` long — the rendered judge prompt for each
            example. The score is extracted from the **last position**
            of each row (the position the judge would emit next).
        cfg:    :class:`GRMConfig`.

    Returns:
        ``(B,)`` float reward tensor — expected value of the score
        distribution per example. Always finite; if the score-token
        subset is degenerate (every score-token logit is ``-inf``)
        the renormalization clamps via softmax so reward is at least
        the minimum ``score_values`` entry.
    """
    if tokens.dim() != 2:
        raise ValueError(
            f"tokens must be (B, T); got shape {tuple(tokens.shape)}",
        )
    out = judge(tokens)
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("judge logits must be a Tensor")
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be (B, T, V); got {tuple(logits.shape)}",
        )

    last_logits = logits[:, -1, :]  # (B, V)
    score_idx = torch.tensor(
        cfg.score_token_ids, device=last_logits.device, dtype=torch.long,
    )
    score_logits = last_logits.index_select(-1, score_idx)
    probs = F.softmax(score_logits / cfg.temperature, dim=-1)
    score_values = torch.tensor(
        cfg.score_values, device=last_logits.device, dtype=probs.dtype,
    )
    return (probs * score_values).sum(dim=-1)


class GRMRewardFn:
    """Adapter that wraps a judge LM as a :class:`RewardFn`-shaped callable.

    The :class:`RewardFn` Protocol takes ``(prompt: str, completion:
    str) -> float``; this class renders the judge prompt via
    ``cfg.prompt_template``, encodes via the supplied ``encode``
    callable, runs :func:`judge_completion`, and returns the scalar
    reward.

    Args:
        judge:   judge LM module.
        encode:  callable ``str -> list[int]`` mapping rendered text
            to token IDs (your tokenizer's ``encode``).
        cfg:     :class:`GRMConfig`.
        device:  optional device to place the judge inputs on. If None,
            uses the device of the first parameter found in ``judge``.
    """

    def __init__(
        self,
        judge: torch.nn.Module,
        encode: Callable[[str], Sequence[int]],
        cfg: GRMConfig,
        *,
        device: torch.device | None = None,
    ) -> None:
        self._judge = judge
        self._encode = encode
        self._cfg = cfg
        if device is not None:
            self._device = device
        else:
            try:
                self._device = next(judge.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")

    @torch.no_grad()
    def __call__(self, prompt: str, completion: str) -> float:
        text = self._cfg.prompt_template.format(
            prompt=prompt, completion=completion,
        )
        ids = list(self._encode(text))
        if not ids:
            raise ValueError("encoded judge prompt is empty")
        tokens = torch.tensor(
            [ids], dtype=torch.long, device=self._device,
        )
        was_training = self._judge.training
        self._judge.eval()
        try:
            reward = judge_completion(self._judge, tokens, self._cfg)
        finally:
            self._judge.train(was_training)
        return float(reward.squeeze().item())


def binary_grm_config(
    *,
    good_token_id: int,
    bad_token_id: int,
    prompt_template: str = DEFAULT_BINARY_TEMPLATE,
    temperature: float = 1.0,
) -> GRMConfig:
    """Convenience: build a binary good/bad :class:`GRMConfig`.

    Reward is ``p(good)`` per example; scores are 1.0 / 0.0.
    """
    return GRMConfig(
        prompt_template=prompt_template,
        score_token_ids=(good_token_id, bad_token_id),
        score_values=(1.0, 0.0),
        temperature=temperature,
    )


def likert_grm_config(
    *,
    score_token_ids: Sequence[int],
    prompt_template: str,
    temperature: float = 1.0,
) -> GRMConfig:
    """Convenience: build a Likert :class:`GRMConfig` with evenly-spaced rewards.

    For ``K`` score tokens, score values are ``[0, 1/(K-1), 2/(K-1), ..., 1]``.
    Mirrors the typical 5-point ``"1"..."5"`` Likert scale mapped to
    ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
    """
    k = len(score_token_ids)
    if k < 2:
        raise ValueError(f"likert needs at least 2 score tokens; got {k}")
    values = tuple(i / (k - 1) for i in range(k))
    return GRMConfig(
        prompt_template=prompt_template,
        score_token_ids=tuple(score_token_ids),
        score_values=values,
        temperature=temperature,
    )
