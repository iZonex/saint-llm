"""Caption reward functions for caption / VQA RL.

The GRPO + Tree-GRPO trainers consume any callable matching the
:class:`RewardFn` Protocol — ``(prompt: str, completion: str) -> float``.
This module supplies a few small reward components specific to
caption-style training where the supervision is one or more *target
captions* rather than a binary judge:

* :func:`unigram_overlap_reward` — F1 over whitespace-tokenized
  unigrams. Cheap, robust, language-agnostic. Equivalent to a
  smoothed BLEU-1 when summed over multiple references.
* :func:`length_target_reward` — soft length penalty: 1.0 when the
  completion length matches the target band, decaying linearly to 0
  outside it.
* :func:`constant_reward` — convenience for ablations / stub tests.
* :func:`composite_caption_reward` — weighted sum over a list of
  ``(weight, RewardFn)`` pairs. Returns a :class:`RewardFn`-shaped
  callable.

The :class:`PerExampleCaptionRewardFn` adapter binds a *list of target
captions* (one per training prompt) to a stateless reward function so
the trainer can index into it by prompt order — same access pattern
as :func:`saint_llm_posttraining.score_rollouts`.

These are deliberately simple. For LM-as-judge rewards see :mod:`grm`;
for stylized references see e.g. CIDEr / SPICE (out of scope here).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

RewardCallable = Callable[[str, str], float]


def _tokenize_unigrams(text: str) -> list[str]:
    """Lowercase whitespace tokenization. Empty strings drop out."""
    return [tok for tok in text.lower().split() if tok]


def unigram_overlap_reward(
    prompt: str,
    completion: str,
    *,
    target: str,
    mode: str = "f1",
) -> float:
    """Unigram-overlap reward: precision / recall / F1.

    Args:
        prompt:     ignored (RewardFn Protocol). The reference text
            is provided via ``target`` instead.
        completion: model output to score.
        target:     reference caption.
        mode:       one of ``"f1"`` (default), ``"precision"``,
            ``"recall"``.

    Returns:
        Reward in ``[0, 1]``. Empty completion or empty target
        returns 0.0.
    """
    pred_toks = _tokenize_unigrams(completion)
    ref_toks = _tokenize_unigrams(target)
    if not pred_toks or not ref_toks:
        return 0.0
    pred_set = set(pred_toks)
    ref_set = set(ref_toks)
    overlap = pred_set & ref_set
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(ref_set)
    if mode == "precision":
        return precision
    if mode == "recall":
        return recall
    if mode == "f1":
        return 2 * precision * recall / (precision + recall)
    raise ValueError(f"unknown mode {mode!r}; use 'f1' / 'precision' / 'recall'")


def length_target_reward(
    prompt: str,
    completion: str,
    *,
    target_min: int,
    target_max: int,
    decay_tokens: int = 10,
) -> float:
    """Soft length-band reward.

    1.0 when the completion's whitespace-token count is in
    ``[target_min, target_max]``. Outside the band, reward decays
    linearly to 0 over ``decay_tokens`` tokens of distance.

    Args:
        target_min / target_max: inclusive bounds of the "perfect" band.
        decay_tokens: tokens of slack on each side before reward hits 0.
            Setting ``decay_tokens=0`` collapses to a hard rect window.
    """
    if target_min < 0 or target_max < target_min:
        raise ValueError(
            "target_min must be >= 0 and target_max >= target_min; got "
            f"({target_min}, {target_max})",
        )
    if decay_tokens < 0:
        raise ValueError(f"decay_tokens must be >= 0; got {decay_tokens}")
    n = len(_tokenize_unigrams(completion))
    if target_min <= n <= target_max:
        return 1.0
    if decay_tokens == 0:
        return 0.0
    distance = (target_min - n) if n < target_min else (n - target_max)
    return max(0.0, 1.0 - distance / decay_tokens)


def constant_reward(_prompt: str, _completion: str, *, value: float = 0.0) -> float:
    """Stub: returns a fixed value regardless of inputs."""
    return float(value)


@dataclass(frozen=True)
class _Component:
    weight: float
    fn: RewardCallable


def composite_caption_reward(
    components: Sequence[tuple[float, RewardCallable]],
) -> RewardCallable:
    """Build a :class:`RewardFn` that returns a weighted sum.

    Args:
        components: sequence of ``(weight, reward_fn)`` pairs. The
            returned callable computes
            ``sum(w * fn(prompt, completion) for (w, fn) in components)``.

    Weights are applied as-is — caller decides whether to normalize.
    """
    parts = tuple(_Component(weight=float(w), fn=f) for w, f in components)

    def _call(prompt: str, completion: str) -> float:
        return sum(c.weight * c.fn(prompt, completion) for c in parts)

    return _call


class PerExampleCaptionRewardFn:
    """Per-prompt target reference adapter.

    Many caption datasets supply one (or several) reference captions
    per prompt. The :class:`RewardFn` Protocol's signature
    ``(prompt, completion) -> float`` doesn't carry the target, so we
    bind a parallel list of references and pick the matching one by
    prompt key.

    Usage::

        rfn = PerExampleCaptionRewardFn(
            prompt_to_targets={"describe.png": ["a cat", "small cat"]},
            scorer=lambda c, t: unigram_overlap_reward(prompt="", completion=c, target=t),
            multi_target_reduce="max",
        )
        reward = rfn("describe.png", "a small cat")  # -> max F1 across both targets

    Args:
        prompt_to_targets:    map from prompt string to list of target
            captions.
        scorer:               callable ``(completion, target) -> float``.
            One target at a time — :class:`PerExampleCaptionRewardFn`
            handles the multi-target reduction.
        multi_target_reduce:  ``"max"`` (default) takes the best
            target's score; ``"mean"`` averages across targets.
        missing_prompt_value: reward to return when the prompt isn't
            in the map. Defaults to 0.0 (skips training signal).
    """

    def __init__(
        self,
        *,
        prompt_to_targets: dict[str, Sequence[str]],
        scorer: Callable[[str, str], float],
        multi_target_reduce: str = "max",
        missing_prompt_value: float = 0.0,
    ) -> None:
        if multi_target_reduce not in ("max", "mean"):
            raise ValueError(
                f"multi_target_reduce must be 'max' or 'mean'; got "
                f"{multi_target_reduce!r}",
            )
        self._targets = prompt_to_targets
        self._scorer = scorer
        self._reduce = multi_target_reduce
        self._missing = float(missing_prompt_value)

    def __call__(self, prompt: str, completion: str) -> float:
        targets = self._targets.get(prompt)
        if not targets:
            return self._missing
        scores = [self._scorer(completion, t) for t in targets]
        if self._reduce == "max":
            return max(scores)
        return sum(scores) / len(scores)
