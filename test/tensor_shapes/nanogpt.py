# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass
from typing import Any, assert_type, TYPE_CHECKING, TypedDict

import torch
import torch.nn as nn
import torch.nn.init
import torch.optim
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class LayerNorm[M](nn.Module):
    """LayerNorm but with an optional bias. Generic over normalized dimension size."""

    def __init__(self, ndim: Dim[M], bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        assert_type(self.weight, Tensor[M])
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        assert_type(self.bias, Tensor[M] | None)

    def forward[*Bs](self, input: Tensor[*Bs, M]) -> Tensor[*Bs, M]:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class GPTConfig[
    VocabSize,
    BlockSize,
    NEmbedding,
    NHead,
    NLayer,
]:
    """Configuration for GPT model, generic over key dimensions"""

    block_size: Dim[BlockSize]
    vocab_size: Dim[VocabSize]
    n_layer: Dim[NLayer]
    n_head: Dim[NHead]
    n_embd: Dim[NEmbedding]
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class CausalSelfAttention[NEmbedding, NHead, BlockSize](nn.Module):
    """Multi-head causal self-attention. Generic over embedding dim, num heads, and block size."""

    def __init__(self, config: GPTConfig[Any, BlockSize, NEmbedding, NHead, Any]):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        assert_type(self.c_attn, nn.Linear[NEmbedding, (3 * NEmbedding)])
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        assert_type(self.c_proj, nn.Linear[NEmbedding, NEmbedding])
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.bias = nn.Buffer(
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                )
            )
            assert_type(self.bias, Tensor[1, 1, BlockSize, BlockSize])

    def forward[B, T](self, x: Tensor[B, T, NEmbedding]) -> Tensor[B, T, NEmbedding]:
        b, t, c = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        assert_type(b, Dim[B])
        assert_type(t, Dim[T])
        assert_type(c, Dim[NEmbedding])

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        assert_type(x, Tensor[B, T, NEmbedding])
        c_attn = self.c_attn(x)
        assert_type(c_attn, Tensor[B, T, (3 * NEmbedding)])
        split = c_attn.split(self.n_embd, dim=2)
        assert_type(self.n_embd, Dim[NEmbedding])
        assert_type(
            split,
            tuple[
                Tensor[B, T, NEmbedding],
                Tensor[B, T, NEmbedding],
                Tensor[B, T, NEmbedding],
            ],
        )
        q, k, v = split
        assert_type(q, Tensor[B, T, NEmbedding])
        assert_type(k, Tensor[B, T, NEmbedding])
        assert_type(v, Tensor[B, T, NEmbedding])
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        assert_type(k, Tensor[B, NHead, T, (NEmbedding // NHead)])
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        assert_type(q, Tensor[B, NHead, T, (NEmbedding // NHead)])
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        assert_type(v, Tensor[B, NHead, T, (NEmbedding // NHead)])

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
            assert_type(y, Tensor[B, NHead, T, (NEmbedding // NHead)])
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            assert_type(att, Tensor[B, NHead, T, T])
            mask = self.bias[:, :, :t, :t] == 0
            assert_type(mask, Tensor[1, 1, T, T])
            att = att.masked_fill(mask, float("-inf"))
            assert_type(att, Tensor[B, NHead, T, T])
            att = F.softmax(att, dim=-1)
            assert_type(att, Tensor[B, NHead, T, T])
            att = self.attn_dropout(att)
            assert_type(att, Tensor[B, NHead, T, T])
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            assert_type(y, Tensor[B, NHead, T, (NEmbedding // NHead)])

        assert_type(y, Tensor[B, NHead, T, (NEmbedding // NHead)])
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side
        assert_type(y, Tensor[B, T, NEmbedding])

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        assert_type(y, Tensor[B, T, NEmbedding])
        return y


class MLP[NEmbedding](nn.Module):
    """Multi-layer perceptron. Generic over embedding dimension."""

    def __init__(self, config: GPTConfig[Any, Any, NEmbedding, Any, Any]):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        assert_type(self.c_fc, nn.Linear[NEmbedding, (4 * NEmbedding)])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        assert_type(self.c_proj, nn.Linear[(4 * NEmbedding), NEmbedding])
        self.dropout = nn.Dropout(config.dropout)

    def forward[B, T](self, x: Tensor[B, T, NEmbedding]) -> Tensor[B, T, NEmbedding]:
        h = self.c_fc(x)
        assert_type(h, Tensor[B, T, (4 * NEmbedding)])
        h = self.gelu(h)
        assert_type(h, Tensor[B, T, (4 * NEmbedding)])
        x = self.c_proj(h)
        assert_type(x, Tensor[B, T, NEmbedding])
        x = self.dropout(x)
        assert_type(x, Tensor[B, T, NEmbedding])
        return x


class Block[NEmbedding, NHead, BlockSize](nn.Module):
    """Transformer block with self-attention and MLP. Generic over embedding dim, num heads, and block size."""

    def __init__(self, config: GPTConfig[Any, BlockSize, NEmbedding, NHead, Any]):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        assert_type(self.ln_1, LayerNorm[NEmbedding])
        self.attn = CausalSelfAttention(config)
        assert_type(self.attn, CausalSelfAttention[NEmbedding, NHead, BlockSize])
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        assert_type(self.ln_2, LayerNorm[NEmbedding])
        self.mlp = MLP(config)
        assert_type(self.mlp, MLP[NEmbedding])

    def forward[B, T](self, x: Tensor[B, T, NEmbedding]) -> Tensor[B, T, NEmbedding]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfigArgs[
    VocabSize,
    BlockSize,
    NEmbedding,
    NHead,
    NLayer,
](TypedDict):
    """TypedDict for GPTConfig constructor arguments, generic over key dimensions"""

    n_layer: Dim[NLayer]
    n_head: Dim[NHead]
    n_embd: Dim[NEmbedding]
    vocab_size: Dim[VocabSize]
    block_size: Dim[BlockSize]
    bias: bool
    dropout: float


class TransformerModules[VocabSize, BlockSize, NEmbedding, NHead](TypedDict):
    """TypedDict defining the structure of GPT transformer modules. Generic over key dimensions."""

    wte: nn.Embedding[VocabSize, NEmbedding]  # token embeddings
    wpe: nn.Embedding[BlockSize, NEmbedding]  # position embeddings
    drop: nn.Dropout
    h: nn.ModuleList[Block[NEmbedding, NHead, BlockSize]]
    ln_f: LayerNorm[NEmbedding]  # final layer norm


class GPT[VocabSize, BlockSize, NEmbedding, NHead, NLayer](nn.Module):
    """GPT Language Model. Generic over vocabulary size, block size, embedding dim, num heads, and num layers."""

    def __init__(
        self, config: GPTConfig[VocabSize, BlockSize, NEmbedding, NHead, NLayer]
    ):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config: GPTConfig[VocabSize, BlockSize, NEmbedding, NHead, NLayer] = config

        transformer_modules: TransformerModules[
            VocabSize, BlockSize, NEmbedding, NHead
        ] = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        self.transformer = nn.ModuleDict(transformer_modules)
        assert_type(
            self.transformer,
            nn.ModuleDict[TransformerModules[VocabSize, BlockSize, NEmbedding, NHead]],
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        assert_type(self.lm_head, nn.Linear[NEmbedding, VocabSize])
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward[B, T](
        self, idx: Tensor[B, T], targets: Tensor[B, T] | None = None
    ) -> tuple[Tensor[B, T, VocabSize] | Tensor[B, 1, VocabSize], Tensor[()] | None]:
        device = idx.device
        b, t = idx.size()
        assert_type(b, Dim[B])
        assert_type(t, Dim[T])
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        assert_type(pos, Tensor[T])

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        assert_type(tok_emb, Tensor[B, T, NEmbedding])
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        assert_type(pos_emb, Tensor[T, NEmbedding])
        tok_pos_emb = tok_emb + pos_emb
        assert_type(tok_pos_emb, Tensor[B, T, NEmbedding])
        x = self.transformer.drop(tok_pos_emb)
        assert_type(x, Tensor[B, T, NEmbedding])
        for block in self.transformer.h:
            _x: Tensor[B, T, NEmbedding] = x
            x = block(_x)
        assert_type(x, Tensor[B, T, NEmbedding])

        x = self.transformer.ln_f(x)
        assert_type(x, Tensor[B, T, NEmbedding])

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            assert_type(logits, Tensor[B, T, VocabSize])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            assert_type(loss, Tensor[()])
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, (-1,), :]
            )  # note: using list [-1] to preserve the time dim
            assert_type(logits, Tensor[B, 1, VocabSize])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)

        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args: GPTConfigArgs
        if model_type == "gpt2":
            config_args = dict(
                n_layer=12,
                n_head=12,
                n_embd=768,
                vocab_size=50257,
                block_size=1024,
                bias=True,
                dropout=0.0,
            )
        elif model_type == "gpt2-medium":
            config_args = dict(
                n_layer=24,
                n_head=16,
                n_embd=1024,
                vocab_size=50257,
                block_size=1024,
                bias=True,
                dropout=0.0,
            )
        elif model_type == "gpt2-large":
            config_args = dict(
                n_layer=36,
                n_head=20,
                n_embd=1280,
                vocab_size=50257,
                block_size=1024,
                bias=True,
                dropout=0.0,
            )
        else:  # gpt2-xl
            config_args = dict(
                n_layer=48,
                n_head=25,
                n_embd=1600,
                vocab_size=50257,
                block_size=1024,
                bias=True,
                dropout=0.0,
            )

        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        model.state_dict()

        """
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        from transformers import GPT2LMHeadModel

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        """
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if use_fused:
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=True
            )
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate[B](
        self,
        idx: Tensor[B, Any],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor[B, Any]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            assert_type(idx_cond, Tensor[B, Any])
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            assert_type(logits, Tensor[B, 1, VocabSize] | Tensor[B, Any, VocabSize])
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            assert_type(logits, Tensor[B, VocabSize])
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            assert_type(logits, Tensor[B, VocabSize])
            probs = F.softmax(logits, dim=-1)
            assert_type(probs, Tensor[B, VocabSize])
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            assert_type(idx_next, Tensor[B, 1])
            # append sampled index to the running sequence and continue
            _idx: Tensor[B, Any] = idx
            idx = torch.cat((_idx, idx_next), dim=1)

        assert_type(idx, Tensor[B, Any])
        return idx
