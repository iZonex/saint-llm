"""Audio encoder integration: Whisper-large-v3 wrapper + Voxtral-style 12.5Hz downsample.

Per AUGMENTATIONS.md:
    MM-A-01  Whisper-large-v3 encoder, frozen v0.1
    MM-A-02  Voxtral-style continuous audio embeds (input)
    MM-A-03  4× MLP downsample 50Hz → 12.5Hz, project to V4 d_model
    MM-A-04  MTP head extended to "depth-K mode" — config flag in MTPConfig (K=1 default;
             K>1 enables Moshi-style parallel codebook prediction in v0.2)
    MM-A-05  Mimi/Moshi-style discrete codec output — deferred-v0.2, slots reserved in tokenizer

Pipeline (v0.1, audio-IN → text-OUT only):
    log-mel (B, n_mel, T_100Hz)
        → Whisper encoder (2× pool) → (B, T_50Hz, encoder_dim=1280)
        → AudioTokenizer downsample (4× via reshape+MLP) → (B, T_12.5Hz, encoder_dim)
        → caller flattens to (n_audio_tokens, encoder_dim)
        → SaintLLM.forward(..., audio_features=...) replaces audio-region tokens

For 30 s @ 16 kHz audio: 3000 mel frames → 1500 (50 Hz) → 375 (12.5 Hz) audio tokens.
Fits comfortably in V4's 1M context window.
"""

from __future__ import annotations

import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn


class AudioEncoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    encoder_name: str = "openai/whisper-large-v3"
    encoder_hidden_dim: int = 1280
    n_mel_bins: int = 128
    sample_rate: int = 16000
    chunk_duration_sec: int = 30
    downsample_factor: int = 4
    freeze: bool = True

    @property
    def output_dim(self) -> int:
        return self.encoder_hidden_dim

    @property
    def output_rate_hz(self) -> float:
        # Whisper hop is 10 ms (100 Hz mel); encoder pools 2×; we further downsample by factor.
        return 100.0 / (2 * self.downsample_factor)


class FakeWhisperEncoder(nn.Module):
    """Zero-dependency stand-in for Whisper-large-v3.

    Mimics Whisper's 2× temporal pooling. Input (B, n_mel, T) → output (B, T//2, encoder_dim).
    """

    def __init__(self, cfg: AudioEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv1d(
            cfg.n_mel_bins, cfg.encoder_hidden_dim, kernel_size=3, stride=2, padding=1,
        )

    def forward(self, mel: Tensor) -> Tensor:
        out = self.proj(mel)  # (B, encoder_dim, T//2)
        return out.transpose(1, 2).contiguous()  # (B, T//2, encoder_dim)


class WhisperLargeV3Wrapper(nn.Module):
    """Real Whisper-large-v3 encoder. Lazy-loads from HuggingFace on first forward.

    Requires ~3 GB for weights. For unit tests use FakeWhisperEncoder. For integration use
    `@pytest.mark.gpu @pytest.mark.slow`.
    """

    def __init__(self, cfg: AudioEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._encoder: nn.Module | None = None

    def _ensure_loaded(self) -> nn.Module:
        if self._encoder is not None:
            return self._encoder
        from transformers import WhisperModel

        full = WhisperModel.from_pretrained(self.cfg.encoder_name)
        encoder = full.encoder
        if self.cfg.freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)
            encoder.eval()
        self._encoder = encoder
        return encoder

    def forward(self, mel: Tensor) -> Tensor:
        encoder = self._ensure_loaded()
        outputs = encoder(mel)
        return outputs.last_hidden_state


class AudioTokenizer(nn.Module):
    """Wraps any audio encoder + Voxtral-style temporal downsample to 12.5 Hz.

    The downsample fuses `downsample_factor` consecutive frames via reshape + 2-layer MLP
    (factor*D → D → D). Output entries directly fill <|audio_*|> token slots in the LLM stream.
    """

    def __init__(self, cfg: AudioEncoderConfig, encoder: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        if cfg.downsample_factor > 1:
            d = cfg.encoder_hidden_dim
            self.downsample: nn.Module = nn.Sequential(
                nn.Linear(d * cfg.downsample_factor, d, bias=False),
                nn.GELU(),
                nn.Linear(d, d, bias=False),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, mel: Tensor) -> Tensor:
        encoded = self.encoder(mel)  # (B, T_enc, D)
        if self.cfg.downsample_factor <= 1:
            return encoded
        b, t, d = encoded.shape
        pad = (self.cfg.downsample_factor - t % self.cfg.downsample_factor) % self.cfg.downsample_factor
        if pad > 0:
            encoded = F.pad(encoded, (0, 0, 0, pad))
            t = encoded.shape[1]
        grouped = encoded.reshape(b, t // self.cfg.downsample_factor, self.cfg.downsample_factor * d)
        return self.downsample(grouped)
