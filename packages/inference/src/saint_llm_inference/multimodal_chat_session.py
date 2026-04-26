"""MultimodalChatSession — interactive chat with image / audio attachments.

Extends :class:`ChatSession` to handle :class:`MultimodalChatTurn`
inputs. Image features are spliced at the user-turn position via
``<|image_pad|>`` placeholders; audio features go inside an
``<|audio_start|>...<|audio_end|>`` bracket. Streaming generation
passes the combined feature tensors to ``model.forward`` each step
(constant across the response — placeholders only live in the
prompt).

Workflow::

    session = MultimodalChatSession(
        model=model, tokenizer=tok, slots=slots, system="...",
    )
    session.add_user_with_image("describe", image_features)
    for chunk in session.stream_response():
        print(chunk, end="", flush=True)

The vision / audio encoders run **outside** this session — callers
hand in pre-computed feature tensors. That split keeps the encoder
pipeline (often heavy, often cached) decoupled from the LM-side chat
loop.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from saint_llm_core.model import SaintLLM
from saint_llm_data.chat_template import ChatTemplate
from saint_llm_data.multimodal import TokenSlots
from saint_llm_data.multimodal_chat import (
    MultimodalChatTurn,
    render_multimodal_chat,
)
from saint_llm_data.tokenizer import Tokenizer
from torch import Tensor

from saint_llm_inference.chat_session import GenerationConfig
from saint_llm_inference.streaming import (
    stream_greedy_decode,
    stream_top_p_sample,
)


class MultimodalChatSession:
    """Stateful interactive chat with image / audio attachments.

    Mirrors :class:`ChatSession`'s API. Differences:

    * History entries are :class:`MultimodalChatTurn` (carries
      ``image_features`` / ``audio_features`` per turn).
    * :meth:`add_user_with_image` / :meth:`add_user_with_audio`
      attach pre-computed feature tensors to a user turn.
    * Generation passes the concatenated features through
      ``model.forward`` each step.
    """

    def __init__(
        self,
        *,
        model: SaintLLM,
        tokenizer: Tokenizer,
        slots: TokenSlots,
        template: ChatTemplate | None = None,
        system: str = "",
        eos_token: int | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._slots = slots
        self._template = template if template is not None else ChatTemplate()
        self._eos = eos_token if eos_token is not None else tokenizer.eos_token_id
        self.history: list[MultimodalChatTurn] = []
        if system:
            self.history.append(MultimodalChatTurn(role="system", content=system))

    # ----- mutation -------------------------------------------------

    def add_user(self, text: str) -> None:
        if not text:
            raise ValueError("user message must be non-empty")
        self.history.append(MultimodalChatTurn(role="user", content=text))

    def add_user_with_image(
        self, text: str, image_features: Tensor | tuple[Tensor, ...],
    ) -> None:
        """Append a user turn with one or more image-feature tensors."""
        if not text:
            raise ValueError("user message must be non-empty")
        feats = (
            (image_features,)
            if isinstance(image_features, Tensor)
            else tuple(image_features)
        )
        if not feats:
            raise ValueError("at least one image feature tensor is required")
        self.history.append(MultimodalChatTurn(
            role="user", content=text, image_features=feats,
        ))

    def add_user_with_audio(
        self, text: str, audio_features: Tensor | tuple[Tensor, ...],
    ) -> None:
        """Append a user turn with one or more audio-feature tensors."""
        if not text:
            raise ValueError("user message must be non-empty")
        feats = (
            (audio_features,)
            if isinstance(audio_features, Tensor)
            else tuple(audio_features)
        )
        if not feats:
            raise ValueError("at least one audio feature tensor is required")
        self.history.append(MultimodalChatTurn(
            role="user", content=text, audio_features=feats,
        ))

    def add_system(self, text: str) -> None:
        if not text:
            raise ValueError("system message must be non-empty")
        self.history.append(MultimodalChatTurn(role="system", content=text))

    def reset(self) -> None:
        self.history = []

    # ----- generation ----------------------------------------------

    def stream_response(
        self,
        *,
        cfg: GenerationConfig | None = None,
        effort_tier: int | None = None,
    ) -> Iterator[str]:
        """Stream the next assistant response as decoded text fragments."""
        if not self.history:
            raise ValueError("history is empty; add_user() before responding")
        gen_cfg = cfg if cfg is not None else GenerationConfig()

        prompt_turns = list(self.history)
        if effort_tier is not None:
            prompt_turns = self._with_effort_hint(prompt_turns, effort_tier)
        rendered = render_multimodal_chat(
            prompt_turns, self._tokenizer, self._slots,
            template=self._template, add_generation_prompt=True,
        )
        device = next(self._model.parameters()).device
        prompt_ids = torch.tensor(
            [rendered.token_ids], dtype=torch.long, device=device,
        )
        vision = (
            rendered.vision_features.to(device)
            if rendered.vision_features is not None
            else None
        )
        audio = (
            rendered.audio_features.to(device)
            if rendered.audio_features is not None
            else None
        )

        was_training = self._model.training
        self._model.eval()
        try:
            collected_ids: list[int] = []
            stream = self._stream(prompt_ids, gen_cfg, vision, audio)
            for tok in stream:
                tok_id = int(tok.squeeze().item())
                if tok_id == self._eos:
                    break
                collected_ids.append(tok_id)
                yield self._tokenizer.decode([tok_id])
        finally:
            self._model.train(was_training)

        full_response = self._tokenizer.decode(collected_ids).strip()
        self.history.append(MultimodalChatTurn(
            role="assistant", content=full_response,
        ))

    def respond(
        self,
        *,
        cfg: GenerationConfig | None = None,
        effort_tier: int | None = None,
    ) -> str:
        return "".join(self.stream_response(cfg=cfg, effort_tier=effort_tier))

    # ----- internal -------------------------------------------------

    def _stream(
        self,
        prompt_ids: Tensor,
        cfg: GenerationConfig,
        vision: Tensor | None,
        audio: Tensor | None,
    ) -> Iterator[Tensor]:
        if cfg.temperature < 1.0e-8:
            return stream_greedy_decode(
                self._model,
                prompt_ids,
                max_new_tokens=cfg.max_new_tokens,
                eos_token=self._eos,
                vision_features=vision,
                audio_features=audio,
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
            vision_features=vision,
            audio_features=audio,
        )

    @staticmethod
    def _with_effort_hint(
        turns: list[MultimodalChatTurn], effort_tier: int,
    ) -> list[MultimodalChatTurn]:
        out = list(turns)
        out.append(MultimodalChatTurn(
            role="assistant", content="", effort_tier=effort_tier,
        ))
        return out
