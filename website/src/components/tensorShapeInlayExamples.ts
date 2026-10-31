/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import type { InlayHint } from './InlayCodeBlock';

export const linearModelCode = `features = torch.randn((1000, 3))
coefficients = torch.randn((3, 1))

predictions = features @ coefficients
gram_matrix = features.transpose() @ features
results = torch.concatenate((features, predictions), axis=1)`;

export const linearModelHints: readonly InlayHint[] = [
    { line: 1, after: 'features', label: ': Tensor[[1000, 3]]' },
    { line: 2, after: 'coefficients', label: ': Tensor[[3, 1]]' },
    { line: 4, after: 'predictions', label: ': Tensor[[1000, 1]]' },
    { line: 5, after: 'gram_matrix', label: ': Tensor[[3, 3]]' },
    { line: 6, after: 'results', label: ': Tensor[[1000, 4]]' },
];

export const nanoGPTBlockCode = `tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
for block in self.transformer.h:
    x = block(x)
x = self.transformer.ln_f(x)`;

export const nanoGPTBlockHints: readonly InlayHint[] = [
    { line: 1, after: 'tok_emb', label: ': Tensor[[B, T, NEmbedding]]' },
    { line: 2, after: 'pos_emb', label: ': Tensor[[T, NEmbedding]]' },
    { line: 3, after: 'x', label: ': Tensor[[B, T, NEmbedding]]' },
    { line: 5, after: 'x', label: ': Tensor[[B, T, NEmbedding]]' },
    { line: 6, after: 'x', label: ': Tensor[[B, T, NEmbedding]]' },
];

export const nanoGPTMLPCode = `def __init__(self, config: GPTConfig[Any, Any, NEmbedding, Any, Any]):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)

def forward[B: IntVar, T: IntVar](self, x: Tensor[[B, T, NEmbedding]]) -> Tensor[[B, T, NEmbedding]]:
    h = self.c_fc(x)
    h = self.gelu(h)
    x = self.c_proj(h)
    x = self.dropout(x)
    return x`;

export const nanoGPTMLPHints: readonly InlayHint[] = [
    { line: 1, after: ')', label: ' -> None' },
    { line: 9, after: 'h', label: ': Tensor[[B, T, 4 * NEmbedding]]' },
    { line: 10, after: 'h', label: ': Tensor[[B, T, 4 * NEmbedding]]' },
];

export const nanoGPTForwardSignatureCode = `def forward[B: IntVar, T: IntVar](self, x: Tensor[[B, T, NEmbedding]]) -> Tensor[[B, T, NEmbedding]]:
    b, t, c = (
        x.size()
    )  # batch size, sequence length, embedding dimensionality (n_embd)`;

export const nanoGPTForwardSignatureHints: readonly InlayHint[] = [
    { line: 2, after: 'b', label: ': Int[B]' },
    { line: 2, after: 't', label: ': Int[T]' },
    { line: 2, after: 'c', label: ': Int[NEmbedding]' },
];

export const nanoGPTAttentionQKVCode = `# calculate query, key, values for all heads in batch and move head forward to be the batch dim
c_attn = self.c_attn(x)
split = c_attn.split(self.n_embd, dim=2)
q, k, v = split
k = k.view(b, t, self.n_head, c // self.n_head).transpose(
    1, 2
)  # (B, nh, T, hs)
q = q.view(b, t, self.n_head, c // self.n_head).transpose(
    1, 2
)  # (B, nh, T, hs)
v = v.view(b, t, self.n_head, c // self.n_head).transpose(
    1, 2
)  # (B, nh, T, hs)`;

export const nanoGPTAttentionQKVHints: readonly InlayHint[] = [
    { line: 2, after: 'c_attn', label: ': Tensor[[B, T, 3 * NEmbedding]]' },
    {
        line: 3,
        after: 'split',
        label: ': tuple[Tensor[[B, T, NEmbedding]], Tensor[[B, T, NEmbedding]], Tensor[[B, T, NEmbedding]]]',
    },
    { line: 4, after: 'q', label: ': Tensor[[B, T, NEmbedding]]' },
    { line: 4, after: 'k', label: ': Tensor[[B, T, NEmbedding]]' },
    { line: 4, after: 'v', label: ': Tensor[[B, T, NEmbedding]]' },
    {
        line: 5,
        after: 'k',
        label: ': Tensor[[B, NHead, T, NEmbedding // NHead]]',
    },
    {
        line: 8,
        after: 'q',
        label: ': Tensor[[B, NHead, T, NEmbedding // NHead]]',
    },
    {
        line: 11,
        after: 'v',
        label: ': Tensor[[B, NHead, T, NEmbedding // NHead]]',
    },
];

export const nanoGPTAttentionBodyCode = `# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
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
else:
    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    mask = self.bias[:, :, :t, :t] == 0
    att = att.masked_fill(mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

y = (
    y.transpose(1, 2).contiguous().view(b, t, c)
)  # re-assemble all head outputs side by side

# output projection
y = self.resid_dropout(self.c_proj(y))
return y`;

export const nanoGPTAttentionBodyHints: readonly InlayHint[] = [
    {
        line: 4,
        after: 'y',
        label: ': Tensor[[B, NHead, T, NEmbedding // NHead]]',
    },
    { line: 14, after: 'att', label: ': Tensor[[B, NHead, T, T]]' },
    { line: 15, after: 'mask', label: ': Tensor[[1, 1, T, T]]' },
    { line: 16, after: 'att', label: ': Tensor[[B, NHead, T, T]]' },
    { line: 17, after: 'att', label: ': Tensor[[B, NHead, T, T]]' },
    { line: 18, after: 'att', label: ': Tensor[[B, NHead, T, T]]' },
    {
        line: 19,
        after: 'y',
        label: ': Tensor[[B, NHead, T, NEmbedding // NHead]]',
    },
    { line: 21, after: 'y', label: ': Tensor[[B, T, NEmbedding]]' },
    { line: 26, after: 'y', label: ': Tensor[[B, T, NEmbedding]]' },
];
