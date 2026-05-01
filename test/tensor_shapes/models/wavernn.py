# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: pytorch/audio (torchaudio/models/wavernn.py)
#
# This adaptation adds tensor shape type annotations for pyrefly.

# ## Inventory
# - [x] ResBlock.__init__ — Dims: n_freq
# - [x] ResBlock.forward
# - [x] MelResNet.__init__ — Dims: n_freq, n_hidden, n_output, kernel_size; int: n_res_block
# - [x] MelResNet.forward
# - [x] Stretch2d.__init__ — Dims: time_scale, freq_scale
# - [x] Stretch2d.forward
# - [x] UpsampleNetwork.__init__ — Dims: n_freq, n_hidden, n_output, kernel_size; int: n_res_block, upsample_scales
# - [x] UpsampleNetwork.forward
# - [x] WaveRNN.__init__ — Dims: n_classes, n_rnn, n_fc, n_freq, n_hidden, n_output, kernel_size; int: n_res_block, hop_length
# - [x] WaveRNN.forward
# - [x] WaveRNN.infer

import math
from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class ResBlock[NF = 128](nn.Module):
    def __init__(self, n_freq: Dim[NF] = 128) -> None:
        super().__init__()
        self.resblock_model = nn.Sequential(
            nn.Conv1d(
                in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False
            ),
            nn.BatchNorm1d(n_freq),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False
            ),
            nn.BatchNorm1d(n_freq),
        )

    def forward[B, L](self, specgram: Tensor[B, NF, L]) -> Tensor[B, NF, L]:
        residual = self.resblock_model(specgram)
        assert_type(residual, Tensor[B, NF, L])
        out = residual + specgram
        assert_type(out, Tensor[B, NF, L])
        return out


class MelResNet[NF = 128, NH = 128, NO = 128, K = 5](nn.Module):
    def __init__(
        self,
        n_res_block: int = 10,
        n_freq: Dim[NF] = 128,
        n_hidden: Dim[NH] = 128,
        n_output: Dim[NO] = 128,
        kernel_size: Dim[K] = 5,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(
            in_channels=n_freq,
            out_channels=n_hidden,
            kernel_size=kernel_size,
            bias=False,
        )
        self.bn_in = nn.BatchNorm1d(n_hidden)
        self.relu_in = nn.ReLU(inplace=True)
        self.res_blocks = nn.ModuleList(
            [ResBlock(n_hidden) for _ in range(n_res_block)]
        )
        self.conv_out = nn.Conv1d(
            in_channels=n_hidden, out_channels=n_output, kernel_size=1
        )

    def forward[B, L](
        self, specgram: Tensor[B, NF, L]
    ) -> Tensor[B, NO, (1 + L) + (-1 * K)]:
        x = self.conv_in(specgram)
        assert_type(x, Tensor[B, NH, (1 + L) + (-1 * K)])
        x = self.bn_in(x)
        assert_type(x, Tensor[B, NH, (1 + L) + (-1 * K)])
        x = self.relu_in(x)
        assert_type(x, Tensor[B, NH, (1 + L) + (-1 * K)])
        for block in self.res_blocks:
            x = block(x)
        assert_type(x, Tensor[B, NH, (1 + L) + (-1 * K)])
        x = self.conv_out(x)
        assert_type(x, Tensor[B, NO, (1 + L) + (-1 * K)])
        return x


class Stretch2d[TS, FS](nn.Module):
    def __init__(self, time_scale: Dim[TS], freq_scale: Dim[FS]) -> None:
        super().__init__()
        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def forward[B, C, F, T](
        self, specgram: Tensor[B, C, F, T]
    ) -> Tensor[B, C, F * FS, T * TS]:
        x = specgram.repeat_interleave(self.freq_scale, -2)
        assert_type(x, Tensor[B, C, F * FS, T])
        x = x.repeat_interleave(self.time_scale, -1)
        assert_type(x, Tensor[B, C, F * FS, T * TS])
        return x


class UpsampleNetwork[NF = 128, NH = 128, NO = 128, K = 5](nn.Module):
    def __init__(
        self,
        upsample_scales: list[int],
        n_res_block: int = 10,
        n_freq: Dim[NF] = 128,
        n_hidden: Dim[NH] = 128,
        n_output: Dim[NO] = 128,
        kernel_size: Dim[K] = 5,
    ) -> None:
        super().__init__()

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        self.total_scale: int = total_scale

        self.indent = (kernel_size - 1) // 2 * total_scale
        self.resnet = MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.resnet_stretch = Stretch2d(total_scale, 1)

        up_layers: list[nn.Module] = []
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(1, scale * 2 + 1),
                padding=(0, scale),
                bias=False,
            )
            torch.nn.init.constant_(conv.weight, 1.0 / (scale * 2 + 1))
            up_layers.append(stretch)
            up_layers.append(conv)
        self.upsample_layers = nn.Sequential(*up_layers)

    def forward[B, T](
        self, specgram: Tensor[B, NF, T]
    ) -> tuple[Tensor[B, NF, Any], Tensor[B, NO, Any]]:
        resnet_output = self.resnet(specgram)
        assert_type(resnet_output, Tensor[B, NO, (1 + T) + (-1 * K)])
        resnet_output_4d = resnet_output.unsqueeze(1)
        assert_type(resnet_output_4d, Tensor[B, 1, NO, (1 + T) + (-1 * K)])
        resnet_output_4d = self.resnet_stretch(resnet_output_4d)
        # total_scale is int from dynamic loop product → time dim becomes Any
        assert_type(resnet_output_4d, Tensor[B, 1, NO, Any])
        resnet_out = resnet_output_4d.squeeze(1)
        assert_type(resnet_out, Tensor[B, NO, Any])

        specgram_4d = specgram.unsqueeze(1)
        assert_type(specgram_4d, Tensor[B, 1, NF, T])
        upsampling_raw = self.upsample_layers(specgram_4d)
        assert_type(upsampling_raw, Tensor)  # Sequential(*list) erases types
        upsampling_sliced = upsampling_raw.squeeze(1)[:, :, self.indent : -self.indent]
        assert_type(upsampling_sliced, Tensor)  # contagion from Sequential(*list)
        # Annotation fallback: NF freq channels preserved through upsampling
        # Receipt: Sequential(*list) with dynamic layer count; NF is bridge dim
        upsampling_output: Tensor[B, NF, Any] = upsampling_sliced
        assert_type(upsampling_output, Tensor[B, NF, Any])

        return upsampling_output, resnet_out


class WaveRNN[NC, NR = 512, NFC = 512, NF = 128, NH = 128, NO = 128, K = 5](nn.Module):
    def __init__(
        self,
        upsample_scales: list[int],
        n_classes: Dim[NC],
        hop_length: int,
        n_res_block: int = 10,
        n_rnn: Dim[NR] = 512,
        n_fc: Dim[NFC] = 512,
        kernel_size: Dim[K] = 5,
        n_freq: Dim[NF] = 128,
        n_hidden: Dim[NH] = 128,
        n_output: Dim[NO] = 128,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self._pad = (kernel_size - 1 if kernel_size % 2 else kernel_size) // 2
        self.n_rnn = n_rnn
        self.n_aux = n_output // 4
        self.hop_length = hop_length
        self.n_classes = n_classes
        self.n_bits: int = int(math.log2(n_classes))

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        if total_scale != hop_length:
            raise ValueError(
                f"Expected: total_scale == hop_length, but found {total_scale} != {hop_length}"
            )

        self.upsample = UpsampleNetwork(
            upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size
        )
        self.fc = nn.Linear(n_freq + self.n_aux + 1, n_rnn)

        self.rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True)
        self.rnn2 = nn.GRU(n_rnn + self.n_aux, n_rnn, batch_first=True)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(n_rnn + self.n_aux, n_fc)
        self.fc2 = nn.Linear(n_fc + self.n_aux, n_fc)
        self.fc3 = nn.Linear(n_fc, n_classes)

    def forward[B, T](
        self, waveform: Tensor[B, 1, T], specgram: Tensor[B, 1, NF, Any]
    ) -> Tensor[B, 1, T, NC]:
        if waveform.size(1) != 1:
            raise ValueError("Require the input channel of waveform is 1")
        if specgram.size(1) != 1:
            raise ValueError("Require the input channel of specgram is 1")

        waveform_2d = waveform.squeeze(1)
        assert_type(waveform_2d, Tensor[B, T])
        specgram_3d = specgram.squeeze(1)
        assert_type(specgram_3d, Tensor[B, NF, Any])

        batch_size = waveform_2d.size(0)
        h1 = torch.zeros(
            1,
            batch_size,
            self.n_rnn,
            dtype=waveform_2d.dtype,
            device=waveform_2d.device,
        )
        assert_type(h1, Tensor[1, B, NR])
        h2 = torch.zeros(
            1,
            batch_size,
            self.n_rnn,
            dtype=waveform_2d.dtype,
            device=waveform_2d.device,
        )
        assert_type(h2, Tensor[1, B, NR])

        specgram_up, aux = self.upsample(specgram_3d)
        assert_type(specgram_up, Tensor[B, NF, Any])
        assert_type(aux, Tensor[B, NO, Any])
        specgram_up_t = specgram_up.transpose(1, 2)
        assert_type(specgram_up_t, Tensor[B, Any, NF])
        aux_t = aux.transpose(1, 2)
        assert_type(aux_t, Tensor[B, Any, NO])

        aux_idx = [self.n_aux * i for i in range(5)]
        a1 = aux_t[:, :, aux_idx[0] : aux_idx[1]]
        assert_type(a1, Tensor[B, Any, Any])  # slice of Any time and aux dims
        a2 = aux_t[:, :, aux_idx[1] : aux_idx[2]]
        assert_type(a2, Tensor[B, Any, Any])
        a3 = aux_t[:, :, aux_idx[2] : aux_idx[3]]
        assert_type(a3, Tensor[B, Any, Any])
        a4 = aux_t[:, :, aux_idx[3] : aux_idx[4]]
        assert_type(a4, Tensor[B, Any, Any])

        x = torch.cat((waveform_2d.unsqueeze(-1), specgram_up_t, a1), dim=-1)
        assert_type(x, Tensor[B, T, Any])  # cat with Any-dim aux slices
        x = self.fc(x)
        assert_type(x, Tensor[B, T, NR])
        res = x
        assert_type(res, Tensor[B, T, NR])
        x, _ = self.rnn1(x, h1)
        assert_type(x, Tensor[B, T, NR])

        x = x + res
        assert_type(x, Tensor[B, T, NR])
        res = x
        assert_type(res, Tensor[B, T, NR])
        x = torch.cat((x, a2), dim=-1)
        assert_type(x, Tensor[B, T, Any])  # cat with Any-dim aux slice
        x, _ = self.rnn2(x, h2)
        assert_type(x, Tensor[B, T, NR])

        x = x + res
        assert_type(x, Tensor[B, T, NR])
        x = torch.cat((x, a3), dim=-1)
        assert_type(x, Tensor[B, T, Any])  # cat with Any-dim aux slice
        x = self.fc1(x)
        assert_type(x, Tensor[B, T, NFC])
        x = self.relu1(x)
        assert_type(x, Tensor[B, T, NFC])

        x = torch.cat((x, a4), dim=-1)
        assert_type(x, Tensor[B, T, Any])  # cat with Any-dim aux slice
        x = self.fc2(x)
        assert_type(x, Tensor[B, T, NFC])
        x = self.relu2(x)
        assert_type(x, Tensor[B, T, NFC])
        x = self.fc3(x)
        assert_type(x, Tensor[B, T, NC])

        result = x.unsqueeze(1)
        assert_type(result, Tensor[B, 1, T, NC])
        return result

    def infer[B](
        self, specgram: Tensor[B, NF, Any], lengths: Tensor[B] | None = None
    ) -> tuple[Tensor, Tensor[B] | None]:
        device = specgram.device
        dtype = specgram.dtype

        specgram_padded = F.pad(specgram, (self._pad, self._pad))
        assert_type(specgram_padded, Tensor)  # F.pad on Any-dim input

        specgram_up_raw, aux_raw = self.upsample(specgram_padded)
        # Input is bare (from F.pad) → B binds to Unknown, but NF/NO preserved
        assert_type(specgram_up_raw, Tensor[Any, NF, Any])
        assert_type(aux_raw, Tensor[Any, NO, Any])
        # Annotation fallback: recover B from method type param
        # Receipt: bare F.pad input loses B binding; NF/NO from class params
        specgram_up: Tensor[B, NF, Any] = specgram_up_raw
        assert_type(specgram_up, Tensor[B, NF, Any])
        aux: Tensor[B, NO, Any] = aux_raw
        assert_type(aux, Tensor[B, NO, Any])
        if lengths is not None:
            lengths = lengths * self.upsample.total_scale

        output: list[Tensor] = []
        b_size, _, seq_len = specgram_up.size()

        h1 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        assert_type(h1, Tensor[1, B, NR])
        h2 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        assert_type(h2, Tensor[1, B, NR])
        x = torch.zeros((b_size, 1), device=device, dtype=dtype)
        assert_type(x, Tensor[B, 1])

        aux_split = [aux[:, self.n_aux * i : self.n_aux * (i + 1), :] for i in range(4)]

        for i in range(seq_len):
            m_t = specgram_up[:, :, i]
            assert_type(m_t, Tensor[B, NF])

            a1_t, a2_t, a3_t, a4_t = [a[:, :, i] for a in aux_split]
            assert_type(a1_t, Tensor[B, Any])  # aux slice with dynamic idx
            assert_type(a2_t, Tensor[B, Any])
            assert_type(a3_t, Tensor[B, Any])
            assert_type(a4_t, Tensor[B, Any])

            x = torch.cat((x, m_t, a1_t), dim=1)
            assert_type(x, Tensor[B, Any])  # cat of mixed Any dims
            x = self.fc(x)
            assert_type(x, Tensor[B, NR])
            _, h1 = self.rnn1(x.unsqueeze(1), h1)
            assert_type(h1, Tensor[1, B, NR])

            x = x + h1[0]
            assert_type(x, Tensor[B, NR])
            inp = torch.cat((x, a2_t), dim=1)
            assert_type(inp, Tensor[B, Any])  # cat with Any-dim aux slice
            _, h2 = self.rnn2(inp.unsqueeze(1), h2)
            assert_type(h2, Tensor[1, B, NR])

            x = x + h2[0]
            assert_type(x, Tensor[B, NR])
            x = torch.cat((x, a3_t), dim=1)
            assert_type(x, Tensor[B, Any])  # cat with Any-dim aux slice
            x = F.relu(self.fc1(x))
            assert_type(x, Tensor[B, NFC])

            x = torch.cat((x, a4_t), dim=1)
            assert_type(x, Tensor[B, Any])  # cat with Any-dim aux slice
            x = F.relu(self.fc2(x))
            assert_type(x, Tensor[B, NFC])

            logits = self.fc3(x)
            assert_type(logits, Tensor[B, NC])

            posterior = F.softmax(logits, dim=1)
            assert_type(posterior, Tensor[B, NC])

            x = torch.multinomial(posterior, 1).float()
            assert_type(x, Tensor[B, 1])
            x = 2 * x / (2**self.n_bits - 1.0) - 1.0
            assert_type(x, Tensor[B, 1])

            output.append(x)

        # dynamic loop accumulation → stack/permute result is bare
        result = torch.stack(output).permute(1, 2, 0)
        assert_type(result, Tensor)
        return result, lengths


def _smoke_test() -> None:
    model = WaveRNN(upsample_scales=[5, 5, 8], n_classes=512, hop_length=200)
    waveform = torch.randn(2, 1, 6000)
    specgram = torch.randn(2, 1, 128, 30)
    out = model(waveform, specgram)
    assert_type(out, Tensor[2, 1, 6000, 512])
