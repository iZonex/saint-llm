"""Chat template — render ``list[Message]`` -> token IDs with role markers.

The Agent runtime works in :class:`Message` objects; the model sees
flat token sequences. This module bridges the two with an explicit
chat-formatting convention that:

* Wraps each turn in role markers (``<|system|>...<|/system|>``,
  ``<|user|>...<|/user|>``, ``<|assistant|>...<|/assistant|>``).
* Optionally inserts a thinking block
  (``<|effort:N|><|think_start|>...<|think_end|>``) ahead of the
  visible assistant response when CoT reasoning is enabled.
* Exposes a parallel ``loss_mask`` for SFT — 1 on the assistant
  visible response (and EOS), 0 on system / user / role markers /
  thinking content. Whether thinking content is loss-active is
  controlled by ``mask_thinking`` (default True = include it in
  the loss).

The role markers themselves are *string templates*. A BBPE tokenizer
that has these tokens reserved emits each marker as one token; a
naive byte-level tokenizer (CharTokenizer in tests) emits them as
multiple tokens — both work, only token efficiency differs.

This is the v0.0 chat convention. Production switches to a
tokenizer-specific reserved-id template once the BBPE training is
done; the renderer's API stays the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from saint_llm_data.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatTemplate:
    """Formatting strings for chat rendering.

    Attributes:
        system_prefix / system_suffix: bracket the system turn.
        user_prefix / user_suffix:     bracket each user turn.
        assistant_prefix:              opens an assistant turn.
        assistant_suffix:              closes an assistant turn.
        think_start_marker / think_end_marker: bracket a thinking
            block. Used when ``with_thinking=True`` in
            :func:`render_chat`.
        effort_prefix_template:        rendered before the thinking
            block — must include ``{tier}`` placeholder.
        bos_text:                      optional text prepended to the
            full conversation. Empty by default; some tokenizers
            handle BOS via the encoder instead.
    """

    system_prefix: str = "<|system|>\n"
    system_suffix: str = "\n<|/system|>\n"
    user_prefix: str = "<|user|>\n"
    user_suffix: str = "\n<|/user|>\n"
    assistant_prefix: str = "<|assistant|>\n"
    assistant_suffix: str = "\n<|/assistant|>\n"
    think_start_marker: str = "<|think_start|>"
    think_end_marker: str = "<|think_end|>"
    effort_prefix_template: str = "<|effort:{tier}|>"
    bos_text: str = ""


@dataclass(frozen=True)
class ChatTurn:
    """One turn for the chat renderer.

    Attributes:
        role:           "system" / "user" / "assistant".
        content:        the visible turn text (what the user / model
            said). For assistants with a thinking block, this is
            the *answer* — see :attr:`thinking` for the CoT.
        thinking:       optional CoT text. When set on an assistant
            turn, the renderer wraps it in
            ``<|think_start|>...<|think_end|>`` before the visible
            content.
        effort_tier:    optional integer tier (0..4). Renders as
            ``<|effort:N|>`` directly before the thinking block.
            Ignored on non-assistant turns.
    """

    role: Role
    content: str
    thinking: str = ""
    effort_tier: int | None = None


@dataclass(frozen=True)
class RenderedChat:
    """Output of :func:`render_chat`.

    Attributes:
        token_ids:  list of token IDs for the full rendered conversation.
        loss_mask:  parallel list — 1 on positions that should
            contribute to SFT loss (assistant visible content + EOS,
            optionally thinking), 0 elsewhere.
        text:       the underlying string the tokenizer encoded. Useful
            for debugging.
    """

    token_ids: list[int]
    loss_mask: list[int]
    text: str


def render_chat(
    turns: Sequence[ChatTurn],
    tokenizer: Tokenizer,
    *,
    template: ChatTemplate | None = None,
    add_generation_prompt: bool = False,
    append_eos: bool = False,
    mask_thinking: bool = True,
) -> RenderedChat:
    """Render a chat into ``(token_ids, loss_mask, text)``.

    Args:
        turns:                   sequence of :class:`ChatTurn`.
        tokenizer:               any :class:`Tokenizer`.
        template:                formatting overrides; default uses
            the standard saint-llm marker set.
        add_generation_prompt:   if True, append ``assistant_prefix``
            (and the effort tier from the last *assistant* turn if
            any) at the end so the model can continue the next
            assistant turn. Set this for **inference** prompting;
            leave False for SFT (where the assistant turn is part
            of the training data).
        append_eos:              if True, append the tokenizer's EOS
            after the last turn. Stays loss-active iff the last turn
            is an assistant turn.
        mask_thinking:           True (default) → thinking content
            counts toward the loss; False → loss-masked. Mask it out
            during early SFT if you don't yet have curated reasoning
            traces; turn it on once you do.

    Returns:
        :class:`RenderedChat` with parallel ``token_ids`` /
        ``loss_mask`` / ``text``.
    """
    tmpl = template if template is not None else ChatTemplate()

    token_ids: list[int] = []
    loss_mask: list[int] = []
    text_parts: list[str] = []

    if tmpl.bos_text:
        ids = tokenizer.encode(tmpl.bos_text)
        token_ids.extend(ids)
        loss_mask.extend([0] * len(ids))
        text_parts.append(tmpl.bos_text)

    last_assistant_effort: int | None = None

    for turn in turns:
        if turn.role == "system":
            token_ids, loss_mask = _append_segment(
                token_ids, loss_mask, tokenizer,
                text=tmpl.system_prefix + turn.content + tmpl.system_suffix,
                loss_active=False, text_parts=text_parts,
            )
        elif turn.role == "user":
            token_ids, loss_mask = _append_segment(
                token_ids, loss_mask, tokenizer,
                text=tmpl.user_prefix + turn.content + tmpl.user_suffix,
                loss_active=False, text_parts=text_parts,
            )
        elif turn.role == "assistant":
            last_assistant_effort = turn.effort_tier
            # Open assistant turn — markers are loss-masked.
            token_ids, loss_mask = _append_segment(
                token_ids, loss_mask, tokenizer,
                text=tmpl.assistant_prefix, loss_active=False,
                text_parts=text_parts,
            )
            # Optional effort tier marker — also loss-masked (the
            # model will be trained to emit it via the EffortRouter
            # head, not via the LM head).
            if turn.effort_tier is not None:
                token_ids, loss_mask = _append_segment(
                    token_ids, loss_mask, tokenizer,
                    text=tmpl.effort_prefix_template.format(tier=turn.effort_tier),
                    loss_active=False, text_parts=text_parts,
                )
            # Optional thinking block.
            if turn.thinking:
                token_ids, loss_mask = _append_segment(
                    token_ids, loss_mask, tokenizer,
                    text=tmpl.think_start_marker + turn.thinking + tmpl.think_end_marker,
                    loss_active=mask_thinking,
                    text_parts=text_parts,
                )
            # Visible response — always loss-active.
            token_ids, loss_mask = _append_segment(
                token_ids, loss_mask, tokenizer,
                text=turn.content + tmpl.assistant_suffix,
                loss_active=True, text_parts=text_parts,
            )
        else:
            raise ValueError(f"unknown role: {turn.role!r}")

    if add_generation_prompt:
        text = tmpl.assistant_prefix
        if last_assistant_effort is not None:
            # If the last assistant turn declared an effort tier, hint
            # the same one when continuing. Caller can override by
            # post-editing the resulting tokens.
            text += tmpl.effort_prefix_template.format(tier=last_assistant_effort)
        token_ids, loss_mask = _append_segment(
            token_ids, loss_mask, tokenizer,
            text=text, loss_active=False, text_parts=text_parts,
        )

    if append_eos:
        token_ids.append(tokenizer.eos_token_id)
        # EOS is loss-active iff the last user-visible turn was an
        # assistant turn (so the model learns to stop). Otherwise mask
        # it out — there's no teacher signal to follow.
        last_role = turns[-1].role if turns else None
        loss_mask.append(1 if last_role == "assistant" else 0)
        text_parts.append("<eos>")

    return RenderedChat(
        token_ids=token_ids,
        loss_mask=loss_mask,
        text="".join(text_parts),
    )


def _append_segment(
    token_ids: list[int],
    loss_mask: list[int],
    tokenizer: Tokenizer,
    *,
    text: str,
    loss_active: bool,
    text_parts: list[str],
) -> tuple[list[int], list[int]]:
    ids = tokenizer.encode(text)
    token_ids.extend(ids)
    loss_mask.extend([1 if loss_active else 0] * len(ids))
    text_parts.append(text)
    return token_ids, loss_mask
