"""Multimodal chat rendering — chat template + image / audio placeholders.

:class:`ChatTemplate` handles role markers for text-only conversations.
:func:`build_multimodal_prompt_ids` (in saint_llm_inference) splices
in ``<|image_pad|>`` / audio-bracket tokens at the *whole-prompt*
level. This module bridges them: each :class:`MultimodalChatTurn`
carries optional ``image_features`` / ``audio_features`` tensors;
:func:`render_multimodal_chat` walks the turns and weaves the
placeholders inside the chat-template role markers (so the model
sees ``<|user|>describe this <|image_pad|>...<|/user|>``), and
returns the parallel feature tensors stacked in token order.

The output of :func:`render_multimodal_chat` is consumed by:

* :class:`saint_llm_inference.MultimodalChatSession` — interactive
  multimodal chat over a model.
* multimodal SFT training (the loss_mask is the same shape as the
  rendered tokens, so masked-CE works directly).

Features are concatenated in **rendering order**: image features
from turn 1, then audio features from turn 1, then image features
from turn 2, etc. This matches the model's expectation that the
flat ``vision_features`` tensor lines up positionally with the
``<|image_pad|>`` slots in the rendered tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from saint_llm_data.chat_template import ChatTemplate, ChatTurn, RenderedChat
from saint_llm_data.multimodal import TokenSlots
from saint_llm_data.tokenizer import Tokenizer


@dataclass(frozen=True)
class MultimodalChatTurn:
    """One chat turn with optional image / audio attachments.

    Attributes:
        role:            "system" / "user" / "assistant".
        content:         visible text.
        thinking:        CoT block (assistant only).
        effort_tier:     reasoning effort tier 0..4 (assistant only).
        image_features:  tuple of ``(n_patches_i, vision_dim)`` tensors
            — one per image attached to the turn. Each tensor's row
            count determines how many ``<|image_pad|>`` slots get
            inserted in the rendered tokens.
        audio_features:  tuple of ``(n_audio_tokens_j, audio_dim)``
            tensors — one per audio clip. Each clip is bracketed by
            ``<|audio_start|>...<|audio_end|>`` with body slots filled
            via the start ID (matches :func:`encode_multimodal`'s
            convention).
    """

    role: str
    content: str
    thinking: str = ""
    effort_tier: int | None = None
    image_features: tuple[Tensor, ...] = ()
    audio_features: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class RenderedMultimodalChat:
    """Output of :func:`render_multimodal_chat`.

    Attributes:
        token_ids:        full rendered token sequence.
        loss_mask:        parallel loss mask (1 on response, 0 on
            prompt / system / placeholders / role markers).
        vision_features:  ``(N_image_tokens, vision_dim)`` tensor
            stacked in render order. ``None`` when no images.
        audio_features:   ``(N_audio_tokens, audio_dim)`` tensor
            stacked in render order. ``None`` when no audio.
        text:             rendered text (debug aid).
    """

    token_ids: list[int]
    loss_mask: list[int]
    vision_features: Tensor | None = None
    audio_features: Tensor | None = None
    text: str = ""

    def to_torch(
        self, *, device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Convert ``token_ids`` / ``loss_mask`` to ``(1, T)`` tensors."""
        toks = torch.tensor(
            [self.token_ids], dtype=torch.long, device=device,
        )
        mask = torch.tensor(
            [self.loss_mask], dtype=torch.long, device=device,
        )
        return toks, mask


@dataclass
class _RenderState:
    """Mutable scratch buffer used during rendering."""

    token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    vision: list[Tensor] = field(default_factory=list)
    audio: list[Tensor] = field(default_factory=list)
    text: list[str] = field(default_factory=list)


def render_multimodal_chat(
    turns: Sequence[MultimodalChatTurn],
    tokenizer: Tokenizer,
    slots: TokenSlots,
    *,
    template: ChatTemplate | None = None,
    add_generation_prompt: bool = False,
    append_eos: bool = False,
    mask_thinking: bool = True,
) -> RenderedMultimodalChat:
    """Render a multimodal chat into tokens + features.

    Args:
        turns:                 sequence of :class:`MultimodalChatTurn`.
        tokenizer:             any :class:`Tokenizer`.
        slots:                 :class:`TokenSlots` with image_pad /
            audio_start / audio_end IDs.
        template:              chat formatting overrides.
        add_generation_prompt: append ``assistant_prefix`` (+ effort)
            so the model can continue. Set for inference, leave off
            for SFT.
        append_eos:            add EOS after the last turn.
        mask_thinking:         True → CoT content is loss-active.

    Returns:
        :class:`RenderedMultimodalChat` with stacked features in
        rendering order.
    """
    tmpl = template if template is not None else ChatTemplate()
    state = _RenderState()

    if tmpl.bos_text:
        _append_text(state, tokenizer, tmpl.bos_text, loss_active=False)

    last_assistant_effort: int | None = None

    for turn in turns:
        if turn.role == "system":
            _append_text(
                state, tokenizer,
                tmpl.system_prefix + turn.content + tmpl.system_suffix,
                loss_active=False,
            )
        elif turn.role == "user":
            _append_text(state, tokenizer, tmpl.user_prefix, loss_active=False)
            if turn.content:
                _append_text(state, tokenizer, turn.content, loss_active=False)
            _splice_modalities(state, turn, slots)
            _append_text(state, tokenizer, tmpl.user_suffix, loss_active=False)
        elif turn.role == "assistant":
            last_assistant_effort = turn.effort_tier
            _append_text(state, tokenizer, tmpl.assistant_prefix, loss_active=False)
            if turn.effort_tier is not None:
                _append_text(
                    state, tokenizer,
                    tmpl.effort_prefix_template.format(tier=turn.effort_tier),
                    loss_active=False,
                )
            if turn.thinking:
                _append_text(
                    state, tokenizer,
                    tmpl.think_start_marker + turn.thinking + tmpl.think_end_marker,
                    loss_active=mask_thinking,
                )
            _append_text(
                state, tokenizer,
                turn.content + tmpl.assistant_suffix,
                loss_active=True,
            )
        else:
            raise ValueError(f"unknown role: {turn.role!r}")

    if add_generation_prompt:
        text = tmpl.assistant_prefix
        if last_assistant_effort is not None:
            text += tmpl.effort_prefix_template.format(tier=last_assistant_effort)
        _append_text(state, tokenizer, text, loss_active=False)

    if append_eos:
        state.token_ids.append(tokenizer.eos_token_id)
        last_role = turns[-1].role if turns else None
        state.loss_mask.append(1 if last_role == "assistant" else 0)
        state.text.append("<eos>")

    vision = torch.cat(state.vision, dim=0) if state.vision else None
    audio = torch.cat(state.audio, dim=0) if state.audio else None
    return RenderedMultimodalChat(
        token_ids=state.token_ids,
        loss_mask=state.loss_mask,
        vision_features=vision,
        audio_features=audio,
        text="".join(state.text),
    )


def to_chat_turn(turn: MultimodalChatTurn) -> ChatTurn:
    """Strip modality attachments to get the underlying text :class:`ChatTurn`."""
    return ChatTurn(
        role=turn.role,
        content=turn.content,
        thinking=turn.thinking,
        effort_tier=turn.effort_tier,
    )


def render_text_chat_to_multimodal(
    chat: RenderedChat,
) -> RenderedMultimodalChat:
    """Adapter: pass a text-only :class:`RenderedChat` through with no features."""
    return RenderedMultimodalChat(
        token_ids=chat.token_ids,
        loss_mask=chat.loss_mask,
        vision_features=None,
        audio_features=None,
        text=chat.text,
    )


# ---------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------


def _append_text(
    state: _RenderState,
    tokenizer: Tokenizer,
    text: str,
    *,
    loss_active: bool,
) -> None:
    if not text:
        return
    ids = tokenizer.encode(text)
    state.token_ids.extend(ids)
    state.loss_mask.extend([1 if loss_active else 0] * len(ids))
    state.text.append(text)


def _splice_modalities(
    state: _RenderState,
    turn: MultimodalChatTurn,
    slots: TokenSlots,
) -> None:
    for img in turn.image_features:
        if img.dim() != 2:
            raise ValueError(
                f"image_features tensors must be 2D (n_patches, vision_dim); "
                f"got shape {tuple(img.shape)}",
            )
        n_patches = int(img.shape[0])
        state.token_ids.extend([slots.image_pad] * n_patches)
        state.loss_mask.extend([0] * n_patches)
        state.text.append("<image_pad>" * n_patches)
        state.vision.append(img)
    for clip in turn.audio_features:
        if clip.dim() != 2:
            raise ValueError(
                f"audio_features tensors must be 2D (n_tokens, audio_dim); "
                f"got shape {tuple(clip.shape)}",
            )
        n_tokens = int(clip.shape[0])
        # Bracket: audio_start + n_tokens body + audio_end. Body slots
        # use the start ID (matches encode_multimodal's convention so
        # the model splices features at positions in [start, end]).
        n_slots = 1 + n_tokens + 1
        state.token_ids.append(slots.audio_start)
        state.token_ids.extend([slots.audio_start] * n_tokens)
        state.token_ids.append(slots.audio_end)
        state.loss_mask.extend([0] * n_slots)
        state.text.append(f"<audio:{n_tokens}>")
        # Pad start/end positions with zero-row vectors so the flat
        # feature tensor lines up with every slot the model splices at.
        # The vision_dim is taken from the clip itself.
        zero = clip.new_zeros((1, clip.shape[1]))
        state.audio.extend([zero, clip, zero])
