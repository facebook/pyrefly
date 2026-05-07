# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
Tacotron2 text-to-speech model from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/tacotron2/model.py
         pytorch/benchmark/torchbenchmark/models/tacotron2/layers.py

Port notes:
- Encoder: 3 shape-preserving Conv1d layers (kernel_size=5, padding=2) + BatchNorm1d
  + BiLSTM. The BiLSTM output dim = 2 * hidden_size = encoder_embedding_dim.
  Uses the inference path (no pack_padded_sequence).
- Postnet: 5 Conv1d layers with BatchNorm1d. Shape-preserving on NMel channels
  (first conv goes 80→512, middle 3 stay 512→512, last goes 512→80).
- Prenet: 2 Linear layers with dropout (always on, even at eval).
- LocationLayer: Conv1d + Linear for location-sensitive attention.
- Attention: Additive (Bahdanau) attention with location sensitivity.
- DecoderStep: Single decoder step using LSTMCell. The full autoregressive
  loop is excluded — only the teacher-forcing forward shape flow is shown.
- Tacotron2: Full model orchestrating Encoder → DecoderStep → Postnet.
- ConvNorm and LinearNorm: thin wrappers around Conv1d/Linear with default args.
  Inlined at call sites since they add no shape information. ConvNorm and
  LinearNorm classes are included below for completeness.
- TacotronSTFT / STFT: mel spectrogram extraction from waveforms. Uses
  torch.stft (returns unrefined Tensor) and mel filterbank matrix multiplication.
  Included with annotation fallbacks for shapeless ops.

Key patterns exercised:
- BiLSTM encoder: LSTM with bidirectional=True, output dim = 2 * hidden_size
- Conv1d stack (shape-preserving): kernel_size=5, stride=1, padding=2
- LSTMCell: single timestep RNN cell, returns (h, c) tuple
- Additive attention: query + processed memory + location features → alignment
- Teacher-forcing decoder step: single-step shape flow
- STFT / mel spectrogram extraction (shapeless ops with annotations)
- ConvNorm / LinearNorm wrapper classes
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Prenet
# ============================================================================


class Prenet[NMel](nn.Module):
    """Decoder prenet: 2 Linear layers with always-on dropout.

    (B, NMel) → (B, 256)
    """

    def __init__(self, n_mel: Dim[NMel]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_mel, 256)
        self.fc2 = nn.Linear(256, 256)

    def forward[*Bs](self, x: Tensor[*Bs, NMel]) -> Tensor[*Bs, 256]:
        h = F.dropout(F.relu(self.fc1(x)), p=0.5, training=True)
        assert_type(h, Tensor[*Bs, 256])
        return F.dropout(F.relu(self.fc2(h)), p=0.5, training=True)


# ============================================================================
# Postnet
# ============================================================================


class Postnet[NMel](nn.Module):
    """Post-processing network: 5 Conv1d layers, shape-preserving on NMel.

    Architecture (with kernel_size=5, padding=2, all shape-preserving on T):
        Conv1d(NMel, 512, 5, padding=2) → BN → tanh
        Conv1d(512, 512, 5, padding=2) → BN → tanh  (×3)
        Conv1d(512, NMel, 5, padding=2) → BN

    (B, NMel, T) → (B, NMel, T)
    """

    def __init__(self, n_mel: Dim[NMel]) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mel, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, n_mel, kernel_size=5, stride=1, padding=2)
        self.bn5 = nn.BatchNorm1d(n_mel)

    def forward[B, T](self, x: Tensor[B, NMel, T]) -> Tensor[B, NMel, T]:
        h = F.dropout(torch.tanh(self.bn1(self.conv1(x))), 0.5, self.training)
        assert_type(h, Tensor[B, 512, T])
        h = F.dropout(torch.tanh(self.bn2(self.conv2(h))), 0.5, self.training)
        assert_type(h, Tensor[B, 512, T])
        h = F.dropout(torch.tanh(self.bn3(self.conv3(h))), 0.5, self.training)
        assert_type(h, Tensor[B, 512, T])
        h = F.dropout(torch.tanh(self.bn4(self.conv4(h))), 0.5, self.training)
        assert_type(h, Tensor[B, 512, T])
        out = F.dropout(self.bn5(self.conv5(h)), 0.5, self.training)
        assert_type(out, Tensor[B, NMel, T])
        return out


# ============================================================================
# Encoder
# ============================================================================


class Encoder[EmbDim](nn.Module):
    """Tacotron2 encoder: Conv1d stack + BiLSTM.

    Architecture:
        3 × (Conv1d(EmbDim, EmbDim, 5, padding=2) → BN → ReLU → dropout)
        → transpose (B, EmbDim, T) → (B, T, EmbDim)
        → BiLSTM(input=EmbDim, hidden=EmbDim//2, bidirectional=True)
        → (B, T, EmbDim)

    The BiLSTM output = 2 × hidden_size = 2 × (EmbDim//2) = EmbDim.

    (B, EmbDim, T) → (B, T, EmbDim)
    """

    def __init__(self, embed_dim: Dim[EmbDim]) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        # BiLSTM: hidden_size = EmbDim//2, bidirectional → num_directions=2
        # Output: (B, T, hidden_size * 2) = (B, T, EmbDim)
        self.lstm = nn.LSTM(
            embed_dim,
            embed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward[B, T](self, x: Tensor[B, EmbDim, T]) -> Tensor[B, T, EmbDim]:
        # Conv stack (shape-preserving on T)
        h = F.dropout(F.relu(self.bn1(self.conv1(x))), 0.5, self.training)
        assert_type(h, Tensor[B, EmbDim, T])
        h = F.dropout(F.relu(self.bn2(self.conv2(h))), 0.5, self.training)
        assert_type(h, Tensor[B, EmbDim, T])
        h = F.dropout(F.relu(self.bn3(self.conv3(h))), 0.5, self.training)
        assert_type(h, Tensor[B, EmbDim, T])
        # Transpose to (B, T, EmbDim) for LSTM
        h_t = h.transpose(1, 2)
        assert_type(h_t, Tensor[B, T, EmbDim])
        # BiLSTM
        outputs, _h_n, _c_n = self.lstm(h_t)
        # 2 * (EmbDim // 2) = EmbDim can't be proven algebraically
        assert_type(outputs, Tensor[B, T, EmbDim])  # type: ignore[assert-type]
        return outputs  # type: ignore[bad-return]


# ============================================================================
# Location-Sensitive Attention
# ============================================================================


class LocationLayer(nn.Module):
    """Location processing for attention: Conv1d + Linear.

    (B, 2, T) → Conv1d(2, 32, kernel_size=31, padding=15) → (B, 32, T)
    → transpose → (B, T, 32) → Linear(32, 128) → (B, T, 128)
    """

    def __init__(self) -> None:
        super().__init__()
        self.location_conv = nn.Conv1d(2, 32, kernel_size=31, stride=1, padding=15)
        self.location_dense = nn.Linear(32, 128, bias=False)

    def forward[B, T](
        self, attention_weights_cat: Tensor[B, 2, T]
    ) -> Tensor[B, T, 128]:
        processed = self.location_conv(attention_weights_cat)
        assert_type(processed, Tensor[B, 32, T])
        processed_t = processed.transpose(1, 2)
        assert_type(processed_t, Tensor[B, T, 32])
        return self.location_dense(processed_t)


class Attention[EmbDim](nn.Module):
    """Location-sensitive additive attention.

    Computes alignment energies from:
    - query (decoder hidden state): (B, 1024)
    - processed_memory (pre-projected encoder output): (B, T, 128)
    - attention_weights_cat (prev + cumulative weights): (B, 2, T)

    Returns:
    - attention_context: (B, EmbDim) — weighted sum of encoder outputs
    - attention_weights: (B, T) — alignment probabilities
    """

    def __init__(self, embed_dim: Dim[EmbDim]) -> None:
        super().__init__()
        self.query_layer = nn.Linear(1024, 128, bias=False)
        self.memory_layer = nn.Linear(embed_dim, 128, bias=False)
        self.v = nn.Linear(128, 1, bias=False)
        self.location_layer = LocationLayer()

    def forward[B, T](
        self,
        query: Tensor[B, 1024],
        memory: Tensor[B, T, EmbDim],
        processed_memory: Tensor[B, T, 128],
        attention_weights_cat: Tensor[B, 2, T],
    ) -> tuple[Tensor[B, EmbDim], Tensor[B, T]]:
        # Project query: (B, 1024) → (B, 1, 128)
        processed_query = self.query_layer(query.unsqueeze(1))
        assert_type(processed_query, Tensor[B, 1, 128])
        # Location features: (B, 2, T) → (B, T, 128)
        processed_attention = self.location_layer(attention_weights_cat)
        assert_type(processed_attention, Tensor[B, T, 128])
        # Additive attention energy
        energies = self.v(
            torch.tanh(processed_query + processed_attention + processed_memory)
        )
        assert_type(energies, Tensor[B, T, 1])
        energies_squeezed = energies.squeeze(-1)
        assert_type(energies_squeezed, Tensor[B, T])
        attention_weights = F.softmax(energies_squeezed, dim=1)
        assert_type(attention_weights, Tensor[B, T])
        # Context: (B, 1, T) @ (B, T, EmbDim) → (B, 1, EmbDim) → (B, EmbDim)
        context = torch.bmm(attention_weights.unsqueeze(1), memory)
        assert_type(context, Tensor[B, 1, EmbDim])
        context_squeezed = context.squeeze(1)
        assert_type(context_squeezed, Tensor[B, EmbDim])
        return context_squeezed, attention_weights


# ============================================================================
# Decoder Step
# ============================================================================


class DecoderStep[EmbDim](nn.Module):
    """Single decoder step using LSTMCell (non-autoregressive view).

    One step of the autoregressive decoder, exposing the shape flow:
        prenet_out (B, 256) + context (B, EmbDim) → cat → (B, 256 + EmbDim)
        → attention_rnn (LSTMCell) → (B, 1024)
        + context → cat → (B, 1024 + EmbDim)
        → decoder_rnn (LSTMCell) → (B, 1024)
        + context → cat → (B, 1024 + EmbDim)
        → linear_projection → (B, 80)  [mel output]
        → gate_layer → (B, 1)  [stop token]
    """

    def __init__(self, embed_dim: Dim[EmbDim]) -> None:
        super().__init__()
        self.attention_rnn = nn.LSTMCell(256 + embed_dim, 1024)
        self.decoder_rnn = nn.LSTMCell(1024 + embed_dim, 1024)
        self.linear_projection = nn.Linear(1024 + embed_dim, 80)
        self.gate_layer = nn.Linear(1024 + embed_dim, 1)

    def forward[B](
        self,
        prenet_out: Tensor[B, 256],
        attention_context: Tensor[B, EmbDim],
        attn_h: Tensor[B, 1024],
        attn_c: Tensor[B, 1024],
        dec_h: Tensor[B, 1024],
        dec_c: Tensor[B, 1024],
    ) -> tuple[
        Tensor[B, 80],
        Tensor[B, 1],
        Tensor[B, 1024],
        Tensor[B, 1024],
        Tensor[B, 1024],
        Tensor[B, 1024],
    ]:
        # Attention RNN
        cell_input = torch.cat((prenet_out, attention_context), dim=1)
        assert_type(cell_input, Tensor[B, 256 + EmbDim])
        attn_h, attn_c = self.attention_rnn(cell_input, (attn_h, attn_c))
        assert_type(attn_h, Tensor[B, 1024])
        assert_type(attn_c, Tensor[B, 1024])
        # Decoder RNN
        dec_input = torch.cat((attn_h, attention_context), dim=1)
        assert_type(dec_input, Tensor[B, 1024 + EmbDim])
        dec_h, dec_c = self.decoder_rnn(dec_input, (dec_h, dec_c))
        assert_type(dec_h, Tensor[B, 1024])
        assert_type(dec_c, Tensor[B, 1024])
        # Projection
        proj_input = torch.cat((dec_h, attention_context), dim=1)
        assert_type(proj_input, Tensor[B, 1024 + EmbDim])
        mel_out = self.linear_projection(proj_input)
        assert_type(mel_out, Tensor[B, 80])
        gate_out = self.gate_layer(proj_input)
        assert_type(gate_out, Tensor[B, 1])
        return mel_out, gate_out, attn_h, attn_c, dec_h, dec_c


# ============================================================================
# Tacotron2 (full model, teacher-forcing forward)
# ============================================================================


class Tacotron2[NSymbols](nn.Module):
    """Tacotron2 TTS model.

    tokens (B, T_text) → Embedding → (B, T_text, 512)
      → transpose → (B, 512, T_text) → Encoder → (B, T_text, 512)
      → [Decoder steps produce mel frames] → (B, 80, T_mel)
      → Postnet → residual → (B, 80, T_mel)

    This port shows the component shapes. The autoregressive decoder loop
    is not fully typed (it requires variable-length list accumulation).
    """

    def __init__(self, n_symbols: Dim[NSymbols], max_decoder_steps: int = 1000) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, 512)
        self.encoder = Encoder(512)
        self.prenet = Prenet(80)
        self.attention = Attention(512)
        self.decoder_step = DecoderStep(512)
        self.postnet = Postnet(80)
        self.max_decoder_steps = max_decoder_steps

    def forward[B, T](
        self, tokens: Tensor[B, T], teacher_forcing_mel: Tensor | None = None
    ) -> Tensor:
        """Run Tacotron2 end-to-end.

        Original autoregressive decoder loop: at each step, feed the previous
        mel frame through prenet, compute attention context over encoder output,
        run decoder RNNs, produce next mel frame + stop token.

        Args:
            tokens: (B, T) input token IDs
            teacher_forcing_mel: (B, 80, T_mel) ground truth mel for teacher forcing.
                If None, uses autoregressive (predicted) output.

        Returns: (B, 80, T_mel) mel spectrogram after postnet residual.
            T_mel is data-dependent (stop token or max_decoder_steps).
        """
        # Encode text
        embedded = self.embedding(tokens)
        assert_type(embedded, Tensor[B, T, 512])
        encoded = self.encoder(embedded.transpose(1, 2))
        assert_type(encoded, Tensor[B, T, 512])
        # Initialize decoder state
        b = tokens.shape[0]
        attn_h: Tensor[B, 1024] = torch.zeros(b, 1024)
        attn_c: Tensor[B, 1024] = torch.zeros(b, 1024)
        dec_h: Tensor[B, 1024] = torch.zeros(b, 1024)
        dec_c: Tensor[B, 1024] = torch.zeros(b, 1024)
        # Initial context, mel input, and attention state
        t = tokens.shape[1]
        attention_context: Tensor[B, 512] = torch.zeros(b, 512)
        decoder_input: Tensor[B, 80] = torch.zeros(b, 80)
        attention_weights: Tensor[B, T] = torch.zeros(b, t)
        attention_weights_cum: Tensor[B, T] = torch.zeros(b, t)
        processed_memory = self.attention.memory_layer(encoded)
        assert_type(processed_memory, Tensor[B, T, 128])
        # Autoregressive decoder loop
        mel_outputs: list[Tensor[B, 80]] = []
        for step in range(self.max_decoder_steps):
            prenet_out = self.prenet(decoder_input)
            assert_type(prenet_out, Tensor[B, 256])
            # Attention: compute context from encoder output
            attention_weights_cat = torch.cat(
                (attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)),
                dim=1,
            )
            assert_type(attention_weights_cat, Tensor[B, 2, T])
            attention_context, attention_weights = self.attention(
                attn_h, encoded, processed_memory, attention_weights_cat
            )
            assert_type(attention_context, Tensor[B, 512])
            attention_weights_cum = attention_weights_cum + attention_weights
            # Decoder step
            mel_out, gate_out, attn_h, attn_c, dec_h, dec_c = self.decoder_step(
                prenet_out, attention_context, attn_h, attn_c, dec_h, dec_c
            )
            assert_type(mel_out, Tensor[B, 80])
            mel_outputs.append(mel_out)
            # Stop condition: gate > 0.5 (sigmoid threshold)
            if teacher_forcing_mel is None and torch.sigmoid(gate_out).min() > 0.5:
                break
            # Next input: teacher forcing or autoregressive
            if (
                teacher_forcing_mel is not None
                and step < teacher_forcing_mel.shape[2] - 1
            ):
                decoder_input = teacher_forcing_mel[:, :, step + 1]
            else:
                decoder_input = mel_out
        # Stack mel frames: list of (B, 80) → (B, 80, T_mel)
        mel_stack = torch.stack(mel_outputs, dim=2)
        # Postnet residual
        mel_post = mel_stack + self.postnet(mel_stack)
        return mel_post


# ============================================================================
# ConvNorm / LinearNorm (thin wrappers, included for completeness)
# ============================================================================


class ConvNorm[InC, OutC](nn.Module):
    """Thin wrapper around Conv1d with Xavier uniform initialization.

    Original: tacotron2/layers.py ConvNorm class.
    Used throughout the original code for consistent initialization.
    Inlined at call sites in the main model classes above.
    """

    def __init__(
        self,
        in_channels: Dim[InC],
        out_channels: Dim[OutC],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        nn.init.xavier_uniform_(self.conv.weight)

    def forward[B, T](self, x: Tensor[B, InC, T]) -> Tensor[B, OutC, T]:
        return self.conv(x)


class LinearNorm[InF, OutF](nn.Module):
    """Thin wrapper around Linear with Xavier uniform initialization.

    Original: tacotron2/layers.py LinearNorm class.
    """

    def __init__(self, in_features: Dim[InF], out_features: Dim[OutF]) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward[B](self, x: Tensor[B, InF]) -> Tensor[B, OutF]:
        return self.linear(x)


# ============================================================================
# STFT / Audio Processing
# ============================================================================


class STFT(nn.Module):
    """Short-Time Fourier Transform module.

    Original: tacotron2/layers.py STFT class.

    Computes STFT on a waveform using torch.stft (which returns complex-valued
    output). The magnitudes are extracted. torch.stft returns unrefined Tensor
    in our type system, so results are annotated.

    filter_length=1024, hop_length=256, win_length=1024 (defaults).
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window: Tensor = nn.Buffer(torch.hann_window(win_length))

    def transform(self, y: Tensor) -> Tensor:
        """Compute STFT magnitudes.

        (B, WaveLen) → (B, n_fft//2+1, n_frames)

        torch.stft returns shapeless Tensor in our stubs.
        """
        # torch.stft returns complex output, from which we take magnitudes
        stft_out: Tensor = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        magnitudes: Tensor = torch.abs(stft_out)
        return magnitudes


class TacotronSTFT(nn.Module):
    """Mel spectrogram extraction for Tacotron2.

    Original: tacotron2/layers.py TacotronSTFT class.

    Computes mel spectrograms from raw waveforms:
        waveform → STFT → magnitudes → mel filterbank → log mel spectrogram

    The mel filterbank is a (n_mel_channels, n_fft//2+1) matrix that projects
    linear-frequency magnitudes to mel-frequency bins.
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        sampling_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
    ) -> None:
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        # Mel filterbank: (n_mel_channels, n_fft//2+1)
        # In the original, this comes from librosa.filters.mel.
        # Here we register a placeholder buffer of the right shape.
        n_fft_bins = filter_length // 2 + 1
        self.mel_basis: Tensor = nn.Buffer(torch.randn(n_mel_channels, n_fft_bins))

    def spectral_normalize(self, magnitudes: Tensor) -> Tensor:
        """Dynamic range compression: log(clamp(x, min=1e-5))."""
        return torch.log(torch.clamp(magnitudes, min=1e-5))

    def mel_spectrogram(self, y: Tensor) -> Tensor:
        """Compute mel spectrogram from waveform.

        (B, WaveLen) → (B, n_mel_channels, n_frames)

        Uses annotation fallback since STFT output shapes are not tracked.
        """
        magnitudes = self.stft_fn.transform(y)
        # mel_basis: (n_mel, n_fft_bins), magnitudes: (B, n_fft_bins, n_frames)
        mel_output: Tensor = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


# ============================================================================
# Smoke tests
# ============================================================================


def test_prenet():
    """Test prenet: (B, 80) → (B, 256)."""
    prenet = Prenet(80)
    x: Tensor[4, 80] = torch.randn(4, 80)
    out = prenet(x)
    assert_type(out, Tensor[4, 256])


def test_postnet():
    """Test postnet: (B, 80, T) → (B, 80, T) shape-preserving."""
    postnet = Postnet(80)
    x: Tensor[4, 80, 200] = torch.randn(4, 80, 200)
    out = postnet(x)
    assert_type(out, Tensor[4, 80, 200])


def test_encoder():
    """Test encoder: (B, 512, T) → (B, T, 512) with BiLSTM."""
    enc = Encoder(512)
    x: Tensor[4, 512, 50] = torch.randn(4, 512, 50)
    out = enc(x)
    assert_type(out, Tensor[4, 50, 512])


def test_location_layer():
    """Test location layer: (B, 2, T) → (B, T, 128)."""
    loc = LocationLayer()
    x: Tensor[4, 2, 50] = torch.randn(4, 2, 50)
    out = loc(x)
    assert_type(out, Tensor[4, 50, 128])


def test_attention():
    """Test attention: query + memory → context + weights."""
    attn = Attention(512)
    query: Tensor[4, 1024] = torch.randn(4, 1024)
    memory: Tensor[4, 50, 512] = torch.randn(4, 50, 512)
    processed_memory: Tensor[4, 50, 128] = torch.randn(4, 50, 128)
    weights_cat: Tensor[4, 2, 50] = torch.randn(4, 2, 50)
    context, weights = attn(query, memory, processed_memory, weights_cat)
    assert_type(context, Tensor[4, 512])
    assert_type(weights, Tensor[4, 50])


def test_decoder_step():
    """Test single decoder step shape flow."""
    dec = DecoderStep(512)
    prenet_out: Tensor[4, 256] = torch.randn(4, 256)
    ctx: Tensor[4, 512] = torch.randn(4, 512)
    attn_h: Tensor[4, 1024] = torch.zeros(4, 1024)
    attn_c: Tensor[4, 1024] = torch.zeros(4, 1024)
    dec_h: Tensor[4, 1024] = torch.zeros(4, 1024)
    dec_c: Tensor[4, 1024] = torch.zeros(4, 1024)
    mel, gate, attn_h2, attn_c2, dec_h2, dec_c2 = dec(
        prenet_out, ctx, attn_h, attn_c, dec_h, dec_c
    )
    assert_type(mel, Tensor[4, 80])
    assert_type(gate, Tensor[4, 1])


def test_conv_norm():
    """Test ConvNorm wrapper: properly types output channels."""
    conv = ConvNorm(80, 512, kernel_size=5, padding=2)
    x: Tensor[4, 80, 200] = torch.randn(4, 80, 200)
    out = conv(x)
    assert_type(out, Tensor[4, 512, 200])


def test_linear_norm():
    """Test LinearNorm wrapper: properly types output features."""
    ln = LinearNorm(256, 128)
    x: Tensor[4, 256] = torch.randn(4, 256)
    out = ln(x)
    assert_type(out, Tensor[4, 128])
