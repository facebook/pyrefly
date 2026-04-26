# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test nn.ModuleDict with TypedDict for typed attribute access

This tests that ModuleDict[T: TypedDict] allows:
1. Type inference from initialization: ModuleDict(typed_dict) → ModuleDict[TypedDict]
2. Attribute access returns the TypedDict field type: module_dict.key → FieldType
3. Item access also returns the TypedDict field type: module_dict["key"] → FieldType
"""

from typing import Any, assert_type, TYPE_CHECKING, TypedDict

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

# ============================================================================
# Test 1: Basic ModuleDict with TypedDict
# ============================================================================


class SimpleModules(TypedDict):
    """TypedDict defining the structure of modules"""

    linear: nn.Linear
    dropout: nn.Dropout


def test_basic_moduledict_typeddict():
    """Test basic ModuleDict with TypedDict initialization"""
    # Create typed dict of modules
    modules: SimpleModules = dict(linear=nn.Linear(10, 5), dropout=nn.Dropout(0.1))

    # Initialize ModuleDict - should infer ModuleDict[SimpleModules]
    module_dict = nn.ModuleDict(modules)

    # Attribute access should return the correct type from TypedDict
    assert_type(module_dict.linear, nn.Linear)
    assert_type(module_dict.dropout, nn.Dropout)

    # Item access should also work
    assert_type(module_dict["linear"], nn.Linear)
    assert_type(module_dict["dropout"], nn.Dropout)


# ============================================================================
# Test 2: ModuleDict with Embedding modules
# ============================================================================


class EmbeddingModules(TypedDict):
    """TypedDict for embedding modules"""

    token_emb: nn.Embedding
    position_emb: nn.Embedding


def test_embedding_moduledict():
    """Test ModuleDict with Embedding modules"""
    modules: EmbeddingModules = dict(
        token_emb=nn.Embedding(50000, 768), position_emb=nn.Embedding(1024, 768)
    )

    embeddings = nn.ModuleDict(modules)

    # Should be typed as nn.Embedding, not generic Module
    tok_emb: nn.Embedding = embeddings.token_emb
    pos_emb: nn.Embedding = embeddings.position_emb

    # Can call forward on them (since they're typed as Embedding)
    # Result shape: [32, 128, embedding_dim] where embedding_dim is unknown
    indices: Tensor[32, 128] = torch.randn(32, 128)
    assert_type(tok_emb(indices), Tensor[32, 128, Any])
    assert_type(pos_emb(indices), Tensor[32, 128, Any])


# ============================================================================
# Test 3: Complex nested structure (like GPT transformer)
# ============================================================================


class TransformerModules(TypedDict):
    """TypedDict for transformer modules (GPT-style)"""

    wte: nn.Embedding  # token embeddings
    wpe: nn.Embedding  # position embeddings
    drop: nn.Dropout
    # Note: h would be ModuleList, ln_f would be LayerNorm (custom),
    # but keeping it simple for now


def test_transformer_moduledict():
    """Test ModuleDict with transformer-style modules"""
    modules: TransformerModules = dict(
        wte=nn.Embedding(50257, 768), wpe=nn.Embedding(1024, 768), drop=nn.Dropout(0.1)
    )

    transformer = nn.ModuleDict(modules)

    # All these should be correctly typed
    assert_type(transformer.wte, nn.Embedding)
    assert_type(transformer.wpe, nn.Embedding)
    assert_type(transformer.drop, nn.Dropout)

    # Can use them with proper types
    idx: Tensor[2, 128] = torch.randn(2, 128)
    pos: Tensor[128] = torch.randn(128)

    tok_emb: Tensor = transformer.wte(idx)
    pos_emb: Tensor = transformer.wpe(pos)
    combined: Tensor = tok_emb + pos_emb
    assert_type(transformer.drop(combined), Tensor)


# ============================================================================
# Test 4: Mixed module types
# ============================================================================


class MixedModules(TypedDict):
    """TypedDict with various module types"""

    embedding: nn.Embedding
    linear1: nn.Linear
    linear2: nn.Linear
    activation: nn.GELU
    dropout: nn.Dropout


def test_mixed_moduledict():
    """Test ModuleDict with mixed module types"""
    modules: MixedModules = dict(
        embedding=nn.Embedding(1000, 128),
        linear1=nn.Linear(128, 256),
        linear2=nn.Linear(256, 128),
        activation=nn.GELU(),
        dropout=nn.Dropout(0.2),
    )

    model = nn.ModuleDict(modules)

    # Each attribute should have its specific type
    assert_type(model.embedding, nn.Embedding)
    assert_type(model.linear1, nn.Linear)
    assert_type(model.linear2, nn.Linear)
    assert_type(model.activation, nn.GELU)
    assert_type(model.dropout, nn.Dropout)

    # Build a simple forward pass
    x: Tensor[32, 10] = torch.randn(32, 10)
    embedded: Tensor = model.embedding(x)
    h1: Tensor = model.linear1(embedded)
    h1_act: Tensor = model.activation(h1)
    h1_drop: Tensor = model.dropout(h1_act)
    # Linear layer output shape is complex - just check it's a Tensor
    # The actual shape depends on the model configuration
    _ = model.linear2(h1_drop)


# ============================================================================
# Test 5: Explicit type annotation
# ============================================================================


def test_explicit_annotation():
    """Test with explicit ModuleDict type annotation"""
    modules: SimpleModules = dict(linear=nn.Linear(20, 10), dropout=nn.Dropout(0.5))

    # Explicit annotation (should match inferred type)
    module_dict: nn.ModuleDict[SimpleModules] = nn.ModuleDict(modules)

    assert_type(module_dict.linear, nn.Linear)
    assert_type(module_dict.dropout, nn.Dropout)
