"""ChatSession — stateful interactive chat driver.

Closes the gap between low-level decoders (:func:`stream_greedy_decode`,
:func:`stream_top_p_sample`) and "I want to talk to my model in a
loop." Wraps:

* a model
* a :class:`Tokenizer`
* a :class:`ChatTemplate`
* a turn history (list of :class:`ChatTurn`)

Public API:

* :meth:`ChatSession.add_user` — append a user turn.
* :meth:`ChatSession.add_system` — append/replace the system seed.
* :meth:`ChatSession.stream_response` — render the prompt with a
  generation marker, stream tokens through the model, decode each
  yielded token, and yield decoded text fragments. When the stream
  ends (EOS or max-new-tokens), the final assistant turn is appended
  to history automatically.
* :meth:`ChatSession.respond` — convenience: collect the full text
  and return it (non-streaming).

The session does not own batching, KV cache reuse, or persistent
storage — those layers compose around it. Reasoning / effort tier:
pass ``effort_tier`` to ``stream_response`` to inject the
``<|effort:N|>`` marker into the generation prompt; the model still
has to be RL-trained to actually emit a thinking block and respect
the budget.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from saint_llm_core.model import SaintLLM
from saint_llm_data.chat_template import ChatTemplate, ChatTurn, render_chat
from saint_llm_data.tokenizer import Tokenizer
from torch import Tensor

from saint_llm_inference.streaming import (
    stream_greedy_decode,
    stream_top_p_sample,
)


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling knobs for :meth:`ChatSession.stream_response`.

    Attributes:
        max_new_tokens: cap on response length.
        temperature:    0.0 -> greedy; > 0 -> top-p sampling.
        top_p:          nucleus probability when sampling.
        top_k:          optional top-k cap.
        seed:           if set, seeds a fresh Generator per call for
            reproducibility.
    """

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    seed: int | None = None


class ChatSession:
    """Stateful chat session.

    Args:
        model:     a :class:`SaintLLM` (or any module with the same
            ``forward`` dict).
        tokenizer: any :class:`Tokenizer`.
        template:  formatting overrides; defaults to the standard
            saint-llm role markers.
        system:    optional system-prompt seed appended to history at
            construction.
        eos_token: optional EOS token override. ``None`` uses the
            tokenizer's ``eos_token_id``.
    """

    def __init__(
        self,
        *,
        model: SaintLLM,
        tokenizer: Tokenizer,
        template: ChatTemplate | None = None,
        system: str = "",
        eos_token: int | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._template = template if template is not None else ChatTemplate()
        self._eos = eos_token if eos_token is not None else tokenizer.eos_token_id
        self.history: list[ChatTurn] = []
        if system:
            self.history.append(ChatTurn(role="system", content=system))

    # ----- mutation -------------------------------------------------

    def add_user(self, text: str) -> None:
        if not text:
            raise ValueError("user message must be non-empty")
        self.history.append(ChatTurn(role="user", content=text))

    def add_system(self, text: str) -> None:
        """Append a fresh system turn — does not replace prior turns."""
        if not text:
            raise ValueError("system message must be non-empty")
        self.history.append(ChatTurn(role="system", content=text))

    def reset(self) -> None:
        """Drop all turns. The session can then be reused."""
        self.history = []

    # ----- generation ----------------------------------------------

    def stream_response(
        self,
        *,
        cfg: GenerationConfig | None = None,
        effort_tier: int | None = None,
    ) -> Iterator[str]:
        """Stream the next assistant response as decoded text fragments.

        Each yielded value is the decoded text for one model token.
        After the iterator finishes, the assistant turn is appended to
        :attr:`history` so subsequent calls see it.

        Args:
            cfg:         :class:`GenerationConfig`. Defaults to a
                temperature-0.7 nucleus-0.9 setup.
            effort_tier: optional effort tier (0..4) injected into the
                generation prompt.
        """
        if not self.history:
            raise ValueError("history is empty; add_user() before responding")
        gen_cfg = cfg if cfg is not None else GenerationConfig()

        # Render the prompt with a generation marker so the model picks
        # up at the assistant turn. Inject the effort tier if provided.
        prompt_turns = list(self.history)
        if effort_tier is not None:
            # Wrap the last assistant turn (if any) with the desired tier
            # so add_generation_prompt picks it up; otherwise rely on
            # render_chat's default behavior.
            prompt_turns = self._with_effort_hint(prompt_turns, effort_tier)
        rendered = render_chat(
            prompt_turns, self._tokenizer,
            template=self._template, add_generation_prompt=True,
        )
        prompt_ids = torch.tensor(
            [rendered.token_ids], dtype=torch.long,
            device=next(self._model.parameters()).device,
        )

        was_training = self._model.training
        self._model.eval()
        try:
            collected_ids: list[int] = []
            stream = self._stream(prompt_ids, gen_cfg)
            for tok in stream:
                tok_id = int(tok.squeeze().item())
                if tok_id == self._eos:
                    break
                collected_ids.append(tok_id)
                yield self._tokenizer.decode([tok_id])
        finally:
            self._model.train(was_training)

        full_response = self._tokenizer.decode(collected_ids).strip()
        self.history.append(ChatTurn(role="assistant", content=full_response))

    def respond(
        self,
        *,
        cfg: GenerationConfig | None = None,
        effort_tier: int | None = None,
    ) -> str:
        """Non-streaming convenience: drain :meth:`stream_response` and join."""
        return "".join(
            self.stream_response(cfg=cfg, effort_tier=effort_tier),
        )

    # ----- internal -------------------------------------------------

    def _stream(
        self, prompt_ids: Tensor, cfg: GenerationConfig,
    ) -> Iterator[Tensor]:
        if cfg.temperature < 1.0e-8:
            return stream_greedy_decode(
                self._model,
                prompt_ids,
                max_new_tokens=cfg.max_new_tokens,
                eos_token=self._eos,
            )
        generator = (
            torch.Generator(device=prompt_ids.device).manual_seed(cfg.seed)
            if cfg.seed is not None
            else None
        )
        return stream_top_p_sample(
            self._model,
            prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            p=cfg.top_p,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            eos_token=self._eos,
            generator=generator,
        )

    @staticmethod
    def _with_effort_hint(
        turns: list[ChatTurn], effort_tier: int,
    ) -> list[ChatTurn]:
        """Annotate the most recent assistant turn (or seed one) with effort.

        :func:`render_chat` carries the tier into the generation prompt
        when called with ``add_generation_prompt=True``. To target a
        fresh response we pop a placeholder assistant turn (no content)
        with the desired tier — it gets dropped before send.
        """
        # The simplest approach: append a "dummy" assistant turn with no
        # content but the chosen tier, then strip it before encoding. We
        # achieve that by inserting it then immediately removing — but
        # render_chat's last_assistant_effort lookup is what we need.
        # Cleanest: just append an empty assistant turn so the renderer
        # records the tier and the generation prompt picks it up. The
        # empty content emits no visible tokens.
        out = list(turns)
        out.append(ChatTurn(role="assistant", content="", effort_tier=effort_tier))
        return out
