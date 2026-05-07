# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Real-World PyTorch Workflows with Symbolic Dimensions
# Demonstrates practical multi-operation patterns that benefit from symbolic shape tracking

from typing import assert_type

import torch
import torch.linalg
import torch.nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# Demo 1: Generic Batch Processing Pipeline
# ============================================================================


def preprocess_batch[B, D](
    batch: Tensor[B, D], mean: Tensor[D], std: Tensor[D]
) -> Tensor[B, D]:
    """Normalize batch - works for ANY batch size"""
    # Subtract mean (broadcasting)
    centered: Tensor[B, D] = batch - mean

    # Divide by std (broadcasting)
    normalized: Tensor[B, D] = centered / std

    return normalized


def process_features[B](
    features: Tensor[B, 768], weights: Tensor[768, 512], bias: Tensor[512]
) -> Tensor[B, 512]:
    """Linear layer with generic batch size"""
    # Matrix multiply: [B, 768] @ [768, 512] → [B, 512]
    linear: Tensor[B, 512] = torch.matmul(features, weights)

    # Add bias (broadcasting)
    with_bias: Tensor[B, 512] = linear + bias

    # Activation
    activated: Tensor[B, 512] = torch.relu(with_bias)

    return activated


def test_batch_pipeline():
    """End-to-end batch processing with multiple operations"""
    # Batch of 32 samples
    batch: Tensor[32, 768] = torch.randn(32, 768)
    mean: Tensor[768] = torch.randn(768)
    std: Tensor[768] = torch.randn(768)

    # Preprocessing
    normalized = preprocess_batch(batch, mean, std)
    assert_type(normalized, Tensor[32, 768])

    # Feature processing
    weights: Tensor[768, 512] = torch.randn(768, 512)
    bias: Tensor[512] = torch.randn(512)
    output = process_features(normalized, weights, bias)
    assert_type(output, Tensor[32, 512])


# ============================================================================
# Demo 2: Concatenating Multiple Data Sources
# ============================================================================


def concat_datasets[N1, N2, N3, D](
    dataset1: Tensor[N1, D], dataset2: Tensor[N2, D], dataset3: Tensor[N3, D]
) -> Tensor[(N1 + N2) + N3, D]:
    """Concatenate three datasets with different sizes"""
    # First concat: N1 + N2
    combined12: Tensor[N1 + N2, D] = torch.cat([dataset1, dataset2], dim=0)

    # Second concat: (N1 + N2) + N3
    all_data: Tensor[(N1 + N2) + N3, D] = torch.cat([combined12, dataset3], dim=0)

    return all_data


def test_concat_multiple_sources():
    """Concatenating multiple data sources"""
    data1: Tensor[100, 768] = torch.randn(100, 768)
    data2: Tensor[50, 768] = torch.randn(50, 768)
    data3: Tensor[75, 768] = torch.randn(75, 768)

    combined = concat_datasets(data1, data2, data3)
    # Type: (100 + 50) + 75 = 225
    assert_type(combined, Tensor[225, 768])


# ============================================================================
# Demo 3: Attention Mechanism with Variable Sequence Length
# ============================================================================


def scaled_dot_product_attention[B, T, D](
    queries: Tensor[B, T, D], keys: Tensor[B, T, D], values: Tensor[B, T, D]
) -> Tensor[B, T, D]:
    """Attention with variable sequence length T"""
    # Compute scores: [B, T, D] @ [B, D, T] → [B, T, T]
    keys_transposed: Tensor[B, D, T] = keys.transpose(1, 2)
    scores: Tensor[B, T, T] = torch.matmul(queries, keys_transposed)

    # In practice, would scale and apply softmax here
    # For demo, simplified to show shape flow
    attention_weights: Tensor[B, T, T] = scores

    # Apply attention: [B, T, T] @ [B, T, D] → [B, T, D]
    output: Tensor[B, T, D] = torch.matmul(attention_weights, values)

    return output


def test_attention_mechanism():
    """Attention with variable sequence length"""
    # batch_size=2, seq_len=128, d_model=64
    q: Tensor[2, 128, 64] = torch.randn(2, 128, 64)
    k: Tensor[2, 128, 64] = torch.randn(2, 128, 64)
    v: Tensor[2, 128, 64] = torch.randn(2, 128, 64)

    output = scaled_dot_product_attention(q, k, v)
    assert_type(output, Tensor[2, 128, 64])


# ============================================================================
# Demo 4: CNN Feature Extraction Pipeline
# ============================================================================


def cnn_features[B, H, W](
    images: Tensor[B, 3, H, W],
) -> Tensor[B, 64, H, W]:
    """CNN feature extraction preserving spatial dimensions"""
    # First conv (padding=1 preserves size with kernel=3)
    weight1: Tensor[32, 3, 3, 3] = torch.randn(32, 3, 3, 3)
    conv1: Tensor[B, 32, H, W] = F.conv2d(images, weight1, stride=1, padding=1)
    relu1: Tensor[B, 32, H, W] = torch.relu(conv1)

    # Second conv (padding=1 preserves size)
    weight2: Tensor[64, 32, 3, 3] = torch.randn(64, 32, 3, 3)
    conv2: Tensor[B, 64, H, W] = F.conv2d(relu1, weight2, stride=1, padding=1)
    relu2: Tensor[B, 64, H, W] = torch.relu(conv2)

    return relu2


def flatten_for_classifier[B, C, H, W](
    features: Tensor[B, C, H, W],
) -> Tensor[B * C * H * W]:
    """Flatten CNN features for classification"""
    return features.flatten()


def test_cnn_pipeline():
    """Complete CNN feature extraction and classification"""
    # Batch of 4 images, 224x224
    images: Tensor[4, 3, 224, 224] = torch.randn(4, 3, 224, 224)

    # Extract features (preserves spatial dimensions)
    features = cnn_features(images)
    assert_type(features, Tensor[4, 64, 224, 224])

    # Flatten for classifier
    flattened = flatten_for_classifier(features)
    # Type: 4 * 64 * 224 * 224 = 12,845,056
    assert_type(flattened, Tensor[12845056])


# ============================================================================
# Demo 5: Sequence Processing with Dynamic Lengths
# ============================================================================


def process_sequences[B, T, D](
    sequences: Tensor[B, T, D], embeddings: Tensor[D, D]
) -> Tensor[B, T, D]:
    """Process variable-length sequences"""
    # Project each token: [B, T, D] → [B, T, D]
    # This is broadcasting over T dimension
    projected: Tensor[B, T, D] = torch.matmul(sequences, embeddings)

    # Activation (layer_norm signature complex, skip for demo)
    activated: Tensor[B, T, D] = torch.relu(projected)

    return activated


def pool_sequences[B, T, D](sequences: Tensor[B, T, D]) -> Tensor[B, D]:
    """Pool over sequence dimension"""
    # Mean over sequence dimension
    pooled: Tensor[B, D] = torch.mean(sequences, dim=1)
    return pooled


def test_sequence_processing():
    """Variable sequence length processing"""
    # Batch of 8, sequence length 50, embedding dim 512
    sequences: Tensor[8, 50, 512] = torch.randn(8, 50, 512)
    embeddings: Tensor[512, 512] = torch.randn(512, 512)

    # Process sequences
    processed = process_sequences(sequences, embeddings)
    assert_type(processed, Tensor[8, 50, 512])

    # Pool to fixed size
    pooled = pool_sequences(processed)
    assert_type(pooled, Tensor[8, 512])


# ============================================================================
# Demo 6: Image Augmentation and Batching
# ============================================================================


def augment_batch[N](images: Tensor[N, 3, 224, 224]) -> Tensor[N * 2, 3, 224, 224]:
    """Augment doubles batch size using tile"""
    # Tile to double batch size
    augmented: Tensor[N * 2, 3, 224, 224] = images.tile((2, 1, 1, 1))

    return augmented


def test_augmentation():
    """Data augmentation doubling batch size"""
    images: Tensor[16, 3, 224, 224] = torch.randn(16, 3, 224, 224)

    augmented = augment_batch(images)
    # Type: 16 * 2 = 32
    assert_type(augmented, Tensor[32, 3, 224, 224])


# ============================================================================
# Demo 7: Multi-Scale Feature Fusion
# ============================================================================


def fuse_features[B, C, H, W](
    high_res: Tensor[B, C, H, W], low_res: Tensor[B, C, H, W]
) -> Tensor[B, C, H, W]:
    """Fuse features at same resolution"""
    # Element-wise addition
    fused: Tensor[B, C, H, W] = high_res + low_res

    # Normalize
    normalized: Tensor[B, C, H, W] = F.relu(fused)

    return normalized


def test_feature_fusion():
    """Multi-scale feature fusion"""
    high: Tensor[1, 64, 56, 56] = torch.randn(1, 64, 56, 56)
    low: Tensor[1, 64, 56, 56] = torch.randn(1, 64, 56, 56)

    fused = fuse_features(high, low)
    assert_type(fused, Tensor[1, 64, 56, 56])


# ============================================================================
# Demo 8: Batch Matmul for Multiple Queries
# ============================================================================


def batched_queries[B, N, D](
    queries: Tensor[B, N, D], database: Tensor[B, 1000, D]
) -> Tensor[B, N, 1000]:
    """Compute similarity scores for N queries against database"""
    # Transpose database: [B, 1000, D] → [B, D, 1000]
    db_t: Tensor[B, D, 1000] = database.transpose(1, 2)

    # Batch matmul: [B, N, D] @ [B, D, 1000] → [B, N, 1000]
    scores: Tensor[B, N, 1000] = torch.matmul(queries, db_t)

    return scores


def test_batched_queries():
    """Batched query matching"""
    queries: Tensor[4, 10, 128] = torch.randn(4, 10, 128)
    database: Tensor[4, 1000, 128] = torch.randn(4, 1000, 128)

    scores = batched_queries(queries, database)
    assert_type(scores, Tensor[4, 10, 1000])


# ============================================================================
# Demo 9: Dynamic Reshaping for Different Tasks
# ============================================================================


def spatial_to_sequence[B](spatial_features: Tensor[B, 256, 7, 7]) -> Tensor:
    """Convert spatial features to sequence (flatten spatial dimensions)"""
    # Permute may not fully support symbolic yet, returning shapeless
    transposed = spatial_features.permute(0, 2, 3, 1)

    # Flatten
    sequence = transposed.flatten(0, 2)

    return sequence


def test_spatial_to_sequence():
    """Reshaping spatial features to sequence"""
    features: Tensor[2, 256, 7, 7] = torch.randn(2, 256, 7, 7)

    sequence = spatial_to_sequence(features)
    # Permute returns shapeless currently
    assert_type(sequence, Tensor)


# ============================================================================
# Demo 10: Multi-Head Attention (Simplified)
# ============================================================================


def split_heads[B, T](
    x: Tensor[B, T, 512],
    num_heads: int,  # 8 heads
) -> Tensor[B * 8, T, 512]:
    """Split into multiple attention heads (simplified using tile)"""
    # Real transformer uses reshape, but for demo use tile
    tiled: Tensor[B * 8, T, 512] = x.tile((8, 1, 1))
    return tiled


def test_multi_head_split():
    """Splitting for multi-head attention (simplified)"""
    x: Tensor[2, 128, 512] = torch.randn(2, 128, 512)

    # Tile to simulate multi-head (simplified demo)
    heads = split_heads(x, 8)
    # Type: 2 * 8 = 16
    assert_type(heads, Tensor[16, 128, 512])


# ============================================================================
# Demo 11: Gather Different Sized Batches
# ============================================================================


def gather_batches[N, M, K, D](
    batch1: Tensor[N, D], batch2: Tensor[M, D], batch3: Tensor[K, D]
) -> Tensor[(N + M) + K, D]:
    """Gather multiple batches into single batch"""
    combined: Tensor[(N + M) + K, D] = torch.cat([batch1, batch2, batch3], dim=0)
    return combined


def process_combined[Total, D](
    combined: Tensor[Total, D],
) -> Tensor[Total, D]:
    """Process combined batch"""
    normalized: Tensor[Total, D] = F.normalize(combined, dim=1)
    return normalized


def test_gather_and_process():
    """Gather and process different sized batches"""
    b1: Tensor[32, 768] = torch.randn(32, 768)
    b2: Tensor[16, 768] = torch.randn(16, 768)
    b3: Tensor[48, 768] = torch.randn(48, 768)

    # Gather: 32 + 16 + 48 = 96
    combined = gather_batches(b1, b2, b3)
    assert_type(combined, Tensor[96, 768])

    # Process
    processed = process_combined(combined)
    assert_type(processed, Tensor[96, 768])


# ============================================================================
# Demo 12: Image Grid to Patches
# ============================================================================


def flatten_images[B](images: Tensor[B, 3, 224, 224]) -> Tensor[B * 150528]:
    """Flatten images completely (3 * 224 * 224 = 150,528)"""
    return images.flatten()


def test_flatten_images():
    """Flatten batch of images"""
    images: Tensor[8, 3, 224, 224] = torch.randn(8, 3, 224, 224)

    flat = flatten_images(images)
    # Type: 8 * 150528 = 1,204,224
    assert_type(flat, Tensor[1204224])


# ============================================================================
# Demo 13: Reduction and Broadcasting Pattern
# ============================================================================


def compute_statistics[B, T, D](
    data: Tensor[B, T, D],
) -> tuple[Tensor[B, D], Tensor[B, D]]:
    """Compute mean and std over sequence dimension"""
    # Mean over T dimension: [B, T, D] → [B, D]
    mean: Tensor[B, D] = torch.mean(data, dim=1)

    # Std over T dimension: [B, T, D] → [B, D]
    std: Tensor[B, D] = torch.std(data, dim=1)

    return mean, std


def normalize_with_stats[B, T, D](
    data: Tensor[B, T, D], mean: Tensor[B, D], std: Tensor[B, D]
) -> Tensor[B, T, D]:
    """Normalize using computed statistics"""
    # Broadcast subtraction: [B, T, D] - [B, D] → [B, T, D]
    centered: Tensor[B, T, D] = data - mean.unsqueeze(1)

    # Broadcast division: [B, T, D] / [B, D] → [B, T, D]
    normalized: Tensor[B, T, D] = centered / std.unsqueeze(1)

    return normalized


def test_statistics_normalization():
    """Computing and using statistics"""
    data: Tensor[4, 50, 128] = torch.randn(4, 50, 128)

    # Compute stats
    mean, std = compute_statistics(data)
    assert_type(mean, Tensor[4, 128])
    assert_type(std, Tensor[4, 128])

    # Normalize
    normalized = normalize_with_stats(data, mean, std)
    assert_type(normalized, Tensor[4, 50, 128])


# ============================================================================
# Demo 14: Multi-Stage Downsampling
# ============================================================================


def downsample_stage[B](features: Tensor[B, 64, 56, 56]) -> Tensor:
    """Downsample with stride=2 (returns shapeless - formula needs symbolic division)"""
    weight: Tensor[128, 64, 3, 3] = torch.randn(128, 64, 3, 3)
    # Stride=2 halves spatial dimensions: 56/2 = 28
    # With symbolic, would compute: H/2, W/2
    downsampled = F.conv2d(features, weight, stride=2, padding=1)
    return downsampled


def test_downsampling():
    """Downsampling feature maps"""
    features: Tensor[1, 64, 56, 56] = torch.randn(1, 64, 56, 56)

    # Downsample (returns shapeless for now due to stride=2 with symbolic)
    down = downsample_stage(features)
    # Would be: Tensor[1, 128, 28, 28]
    # For now: Tensor (shapeless)
    assert_type(down, Tensor)


# ============================================================================
# Demo 15: Reshape for Different Downstream Tasks
# ============================================================================


def prepare_for_linear[B](features: Tensor[B, 512, 7, 7]) -> Tensor[B * 25088]:
    """Flatten CNN features (512 * 7 * 7 = 25,088)"""
    return features.flatten()


def test_prepare_for_linear():
    """Flatten CNN features to vector"""
    features: Tensor[2, 512, 7, 7] = torch.randn(2, 512, 7, 7)

    flat = prepare_for_linear(features)
    # Type: 2 * 25088 = 50,176
    assert_type(flat, Tensor[50176])


# ============================================================================
# Demo 16: Batched Matrix Operations
# ============================================================================


def batch_inverse[B](matrices: Tensor[B, 10, 10]) -> Tensor[B, 10, 10]:
    """Compute inverse for batch of matrices"""
    # torch.linalg.inv preserves all dimensions
    return torch.linalg.inv(matrices)


def batch_solve[B](A: Tensor[B, 10, 10], b: Tensor[B, 10, 5]) -> Tensor[B, 10, 5]:
    """Solve linear systems for batch"""
    # Solve preserves batch and output shape matches b
    return torch.linalg.solve(A, b)


def test_batched_linear_algebra():
    """Linear algebra with batched matrices"""
    matrices: Tensor[4, 10, 10] = torch.randn(4, 10, 10)
    b: Tensor[4, 10, 5] = torch.randn(4, 10, 5)

    # Inverse
    inv = batch_inverse(matrices)
    assert_type(inv, Tensor[4, 10, 10])

    # Solve
    x = batch_solve(matrices, b)
    assert_type(x, Tensor[4, 10, 5])


# ============================================================================
# Summary Test: Complete Pipeline
# ============================================================================


def complete_cnn_pipeline[B](
    inputs: Tensor[B, 3, 224, 224],
) -> Tensor[B * 3211264]:
    """Complete CNN pipeline: conv → relu → flatten"""
    # 1. CNN feature extraction (preserves spatial with padding=1)
    weight1: Tensor[64, 3, 3, 3] = torch.randn(64, 3, 3, 3)
    features: Tensor[B, 64, 224, 224] = F.conv2d(inputs, weight1, stride=1, padding=1)

    # 2. Activation
    activated: Tensor[B, 64, 224, 224] = torch.relu(features)

    # 3. Flatten completely
    # B * 64 * 224 * 224 = B * 3,211,264
    flattened: Tensor[B * 3211264] = activated.flatten()

    return flattened


def test_complete_cnn_pipeline():
    """End-to-end CNN pipeline"""
    images: Tensor[8, 3, 224, 224] = torch.randn(8, 3, 224, 224)

    output = complete_cnn_pipeline(images)
    # 8 * 3211264 = 25,690,112
    assert_type(output, Tensor[25690112])


# ============================================================================
# Demo 17: Transformer Attention with Einsum (NEW!)
# ============================================================================


def multi_head_attention_einsum[B, T, H, D](
    queries: Tensor[B, H, T, D],  # [batch, heads, seq_len, head_dim]
    keys: Tensor[B, H, T, D],
    values: Tensor[B, H, T, D],
) -> Tensor[B, H, T, D]:
    """Multi-head attention using einsum - elegant and efficient!"""
    # Compute attention scores using einsum
    # [B, H, T, D] x [B, H, T, D] -> [B, H, T, T]
    # 'bhid,bhjd->bhij' means: batch, head, query_pos, key_pos
    scores: Tensor[B, H, T, T] = torch.einsum("bhid,bhjd->bhij", queries, keys)

    # Apply attention to values
    # [B, H, T, T] x [B, H, T, D] -> [B, H, T, D]
    # 'bhij,bhjd->bhid' means: attend over key_pos dimension
    output: Tensor[B, H, T, D] = torch.einsum("bhij,bhjd->bhid", scores, values)

    return output


def test_multi_head_attention_einsum():
    """Transformer-style attention with einsum (4 heads, 64-dim)"""
    # B=2, H=4, T=128, D=64
    q: Tensor[2, 4, 128, 64] = torch.randn(2, 4, 128, 64)
    k: Tensor[2, 4, 128, 64] = torch.randn(2, 4, 128, 64)
    v: Tensor[2, 4, 128, 64] = torch.randn(2, 4, 128, 64)

    output = multi_head_attention_einsum(q, k, v)
    assert_type(output, Tensor[2, 4, 128, 64])


# ============================================================================
# Demo 18: Bilinear Pooling with Einsum
# ============================================================================


def bilinear_pooling[B, C, H, W](
    features_a: Tensor[B, C, H, W], features_b: Tensor[B, C, H, W]
) -> Tensor[B, C, C]:
    """Bilinear pooling for visual question answering / fine-grained recognition"""
    # Use flatten(2) to flatten spatial dimensions H, W → H*W
    # flatten(2) flattens from dimension 2 onwards: [B, C, H, W] → [B, C, H*W]
    a_flat: Tensor[B, C, H * W] = features_a.flatten(2)
    b_flat: Tensor[B, C, H * W] = features_b.flatten(2)

    # Compute outer product and sum over spatial locations
    # 'bci,bdi->bcd' computes outer product over C and D, sums over i (spatial)
    pooled: Tensor[B, C, C] = torch.einsum("bci,bdi->bcd", a_flat, b_flat)

    return pooled


def test_bilinear_pooling():
    """Bilinear pooling for multi-modal fusion"""
    # Two feature maps from different modalities
    feat_a: Tensor[2, 512, 7, 7] = torch.randn(2, 512, 7, 7)
    feat_b: Tensor[2, 512, 7, 7] = torch.randn(2, 512, 7, 7)

    # Bilinear pooling produces [B, C, C] interaction matrix
    pooled = bilinear_pooling(feat_a, feat_b)
    assert_type(pooled, Tensor[2, 512, 512])


# ============================================================================
# Demo 19: Unbind for Per-Sample Processing (NEW!)
# ============================================================================


def process_sample[H, W](image: Tensor[3, H, W]) -> Tensor[3, H, W]:
    """Process a single image"""
    # Apply some transformation (simplified)
    processed: Tensor[3, H, W] = torch.relu(image)
    return processed


def process_batch_individually[B, H, W](
    batch: Tensor[B, 3, H, W],
) -> list:
    """Process each sample in batch individually using unbind"""
    # Unbind splits batch into tuple of individual samples
    samples: tuple[Tensor[3, H, W], ...] = torch.unbind(batch, dim=0)

    # Process each sample (in practice, would use list comprehension)
    # For type checking demo, just show the pattern
    return list(samples)


def test_unbind_processing():
    """Unbind batch for per-sample processing"""
    batch: Tensor[4, 3, 224, 224] = torch.randn(4, 3, 224, 224)

    # Unbind returns tuple[Tensor[3, 224, 224], ...]
    samples = torch.unbind(batch, dim=0)
    assert_type(samples, tuple[Tensor[3, 224, 224], ...])

    # Can iterate and process each sample
    _ = process_batch_individually(batch)


# ============================================================================
# Demo 20: Tensor Network Contraction
# ============================================================================


def tensor_network_contraction[N, M, K, L](
    tensor_a: Tensor[N, M, K], tensor_b: Tensor[K, L, M], tensor_c: Tensor[L, N]
) -> Tensor[()]:
    """Complex tensor contraction (used in physics, tensor networks)"""
    # Contract tensor_a with tensor_b over indices K and M
    # 'ijk,klj->il'
    intermediate: Tensor[N, L] = torch.einsum("ijk,klj->il", tensor_a, tensor_b)

    # Contract result with tensor_c to get scalar
    # 'il,li->'
    scalar: Tensor[()] = torch.einsum("il,li->", intermediate, tensor_c)

    return scalar


def test_tensor_network():
    """Tensor network contraction to scalar"""
    a: Tensor[5, 6, 7] = torch.randn(5, 6, 7)
    b: Tensor[7, 8, 6] = torch.randn(7, 8, 6)
    c: Tensor[8, 5] = torch.randn(8, 5)

    result = tensor_network_contraction(a, b, c)
    assert_type(result, Tensor[()])


# ============================================================================
# Demo 21: Batched Outer Products with Einsum
# ============================================================================


def batched_outer_product[B, N, M](
    vectors_a: Tensor[B, N], vectors_b: Tensor[B, M]
) -> Tensor[B, N, M]:
    """Compute outer product for each sample in batch"""
    # 'bi,bj->bij' preserves batch, creates outer product for each
    outer: Tensor[B, N, M] = torch.einsum("bi,bj->bij", vectors_a, vectors_b)
    return outer


def test_batched_outer_product():
    """Outer products for batch of vector pairs"""
    a: Tensor[16, 128] = torch.randn(16, 128)
    b: Tensor[16, 256] = torch.randn(16, 256)

    # Each sample gets 128x256 outer product matrix
    outer = batched_outer_product(a, b)
    assert_type(outer, Tensor[16, 128, 256])


# ============================================================================
# Demo 22: Diagonal Extraction with Einsum
# ============================================================================


def extract_batch_diagonals[B, N](matrices: Tensor[B, N, N]) -> Tensor[B, N]:
    """Extract diagonal from each matrix in batch"""
    # 'bii->bi' extracts diagonal (repeated index)
    diagonals: Tensor[B, N] = torch.einsum("bii->bi", matrices)
    return diagonals


def batch_trace[B, N](matrices: Tensor[B, N, N]) -> Tensor[B]:
    """Compute trace for each matrix in batch"""
    # 'bii->b' sums diagonal for each batch element
    traces: Tensor[B] = torch.einsum("bii->b", matrices)
    return traces


def test_diagonal_operations():
    """Diagonal extraction and trace for batched matrices"""
    matrices: Tensor[8, 100, 100] = torch.randn(8, 100, 100)

    # Extract diagonals
    diags = extract_batch_diagonals(matrices)
    assert_type(diags, Tensor[8, 100])

    # Compute traces
    traces = batch_trace(matrices)
    assert_type(traces, Tensor[8])


# ============================================================================
# Demo 23: Cross-Attention with Einsum
# ============================================================================


def cross_attention_einsum[B, Tq, Tkv, D](
    queries: Tensor[B, Tq, D],  # Query sequence
    keys: Tensor[B, Tkv, D],  # Key-value sequence (different length!)
    values: Tensor[B, Tkv, D],
) -> Tensor[B, Tq, D]:
    """Cross-attention between sequences of different lengths"""
    # Compute attention scores: query attends to key-value sequence
    # 'bqd,bkd->bqk'
    scores: Tensor[B, Tq, Tkv] = torch.einsum("bqd,bkd->bqk", queries, keys)

    # Apply attention to values
    # 'bqk,bkd->bqd'
    output: Tensor[B, Tq, D] = torch.einsum("bqk,bkd->bqd", scores, values)

    return output


def test_cross_attention():
    """Cross-attention between different sequence lengths"""
    queries: Tensor[2, 50, 512] = torch.randn(2, 50, 512)  # 50 query tokens
    keys: Tensor[2, 100, 512] = torch.randn(2, 100, 512)  # 100 kv tokens
    values: Tensor[2, 100, 512] = torch.randn(2, 100, 512)

    output = cross_attention_einsum(queries, keys, values)
    # Output has query sequence length
    assert_type(output, Tensor[2, 50, 512])


# ============================================================================
# Demo 24: Pairwise Distance Matrix with Einsum
# ============================================================================


def pairwise_dot_products[B, N, M, D](
    points_a: Tensor[B, N, D], points_b: Tensor[B, M, D]
) -> Tensor[B, N, M]:
    """Compute dot products between all pairs of points"""
    # Compute pairwise dot products using einsum
    # 'bnd,bmd->bnm' computes dot product for each (n, m) pair
    dot_products: Tensor[B, N, M] = torch.einsum("bnd,bmd->bnm", points_a, points_b)

    return dot_products


def test_pairwise_dot_products():
    """Compute pairwise dot products between point sets"""
    points_a: Tensor[4, 100, 128] = torch.randn(4, 100, 128)  # 100 points
    points_b: Tensor[4, 200, 128] = torch.randn(4, 200, 128)  # 200 points

    # Dot product matrix: 100 x 200 for each batch
    # Used in attention, nearest neighbor search, clustering
    dot_prods = pairwise_dot_products(points_a, points_b)
    assert_type(dot_prods, Tensor[4, 100, 200])
