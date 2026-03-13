# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Runtime tests for runnable model variants.

Verifies that the runnable models (with assert_type stripped) can be
instantiated and run forward passes at Python runtime. Tests both the
future_annotations variants (which use PEP 563 to defer annotation
evaluation) and the int_type_var variants (which use torch_shapes.TypeVar for
runtime-safe arithmetic without future annotations).
"""

import unittest

import torch


class TestNanoGPTFutureAnnotationsRuntime(unittest.TestCase):
    """Test that nanogpt_future_annotations_runnable.py can be imported, instantiated, and run."""

    def _make_config(self):
        from nanogpt_future_annotations_runnable import GPTConfig

        return GPTConfig(
            block_size=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_embd=32,
            dropout=0.0,
            bias=True,
        )

    def test_import(self):
        """Module can be imported without errors."""
        import nanogpt_future_annotations_runnable  # noqa: F401

    def test_instantiate_config(self):
        """GPTConfig dataclass can be created."""
        config = self._make_config()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.n_embd, 32)

    def test_instantiate_model(self):
        """GPT model can be instantiated."""
        from nanogpt_future_annotations_runnable import GPT

        config = self._make_config()
        model = GPT(config)
        self.assertIsNotNone(model)

    def test_forward_inference(self):
        """Forward pass without targets (inference mode)."""
        from nanogpt_future_annotations_runnable import GPT

        config = self._make_config()
        model = GPT(config)
        model.eval()

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            logits, loss = model(idx)

        # Inference mode: only last position
        self.assertEqual(logits.shape, (batch_size, 1, config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_training(self):
        """Forward pass with targets (training mode)."""
        from nanogpt_future_annotations_runnable import GPT

        config = self._make_config()
        model = GPT(config)

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, loss = model(idx, targets)

        self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, ())

    def test_generate(self):
        """Generate method produces tokens."""
        from nanogpt_future_annotations_runnable import GPT

        config = self._make_config()
        model = GPT(config)
        model.eval()

        batch_size = 1
        idx = torch.randint(0, config.vocab_size, (batch_size, 4))
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=3)

        self.assertEqual(generated.shape[0], batch_size)
        self.assertEqual(generated.shape[1], 4 + 3)  # original + generated


class TestGPTFastFutureAnnotationsRuntime(unittest.TestCase):
    """Test that gptfast_future_annotations_runnable.py can be imported and sub-modules can run."""

    def _make_config(self):
        from gptfast_future_annotations_runnable import ModelArgs

        return ModelArgs(
            block_size=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            dim=32,
            n_local_heads=4,
            intermediate_size=64,
        )

    def test_import(self):
        """Module can be imported without errors."""
        import gptfast_future_annotations_runnable  # noqa: F401

    def test_instantiate_config(self):
        """ModelArgs dataclass can be created."""
        config = self._make_config()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.dim, 32)
        self.assertEqual(config.head_dim, 8)  # dim // n_head = 32 // 4

    def test_instantiate_transformer(self):
        """Transformer model can be instantiated."""
        from gptfast_future_annotations_runnable import Transformer

        config = self._make_config()
        model = Transformer(config)
        self.assertIsNotNone(model)

    def test_rms_norm_forward(self):
        """RMSNorm forward pass works."""
        from gptfast_future_annotations_runnable import RMSNorm

        norm = RMSNorm(32)
        x = torch.randn(2, 16, 32)
        out = norm(x)
        self.assertEqual(out.shape, (2, 16, 32))

    def test_feed_forward(self):
        """FeedForward forward pass works."""
        from gptfast_future_annotations_runnable import FeedForward

        config = self._make_config()
        ff = FeedForward(config)
        x = torch.randn(2, 16, 32)
        out = ff(x)
        self.assertEqual(out.shape, (2, 16, 32))

    def test_precompute_freqs_cis(self):
        """precompute_freqs_cis produces correct shape."""
        from gptfast_future_annotations_runnable import precompute_freqs_cis

        # head_dim=8, seq_len=64
        cache = precompute_freqs_cis(seq_len=64, n_elem=8)
        self.assertEqual(cache.shape, (64, 4, 2))  # (seq_len, head_dim//2, 2)

    def test_apply_rotary_emb(self):
        """apply_rotary_emb works on a tensor."""
        from gptfast_future_annotations_runnable import (
            apply_rotary_emb,
            precompute_freqs_cis,
        )

        batch, seq_len, n_heads, head_dim = 2, 16, 4, 8
        x = torch.randn(batch, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(seq_len=seq_len, n_elem=head_dim)
        out = apply_rotary_emb(x, freqs_cis)
        self.assertEqual(out.shape, (batch, seq_len, n_heads, head_dim))

    def test_kv_cache(self):
        """KVCache can be created and updated."""
        from gptfast_future_annotations_runnable import KVCache

        cache = KVCache(
            max_batch_size=2,
            max_seq_length=64,
            n_heads=4,
            head_dim=8,
        )
        input_pos = torch.arange(16)
        k_val = torch.randn(2, 4, 16, 8, dtype=torch.bfloat16)
        v_val = torch.randn(2, 4, 16, 8, dtype=torch.bfloat16)
        k_out, v_out = cache.update(input_pos, k_val, v_val)
        self.assertEqual(k_out.shape, (2, 4, 64, 8))
        self.assertEqual(v_out.shape, (2, 4, 64, 8))


class TestNanoGPTIntTypeVarRuntime(unittest.TestCase):
    """Test that nanogpt_int_type_var_runnable.py can be imported, instantiated, and run."""

    def _make_config(self):
        from nanogpt_int_type_var_runnable import GPTConfig

        return GPTConfig(
            block_size=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_embd=32,
            dropout=0.0,
            bias=True,
        )

    def test_import(self):
        """Module can be imported without errors."""
        import nanogpt_int_type_var_runnable  # noqa: F401

    def test_instantiate_config(self):
        """GPTConfig dataclass can be created."""
        config = self._make_config()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.n_embd, 32)

    def test_instantiate_model(self):
        """GPT model can be instantiated."""
        from nanogpt_int_type_var_runnable import GPT

        config = self._make_config()
        model = GPT(config)
        self.assertIsNotNone(model)

    def test_forward_inference(self):
        """Forward pass without targets (inference mode)."""
        from nanogpt_int_type_var_runnable import GPT

        config = self._make_config()
        model = GPT(config)
        model.eval()

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            logits, loss = model(idx)

        self.assertEqual(logits.shape, (batch_size, 1, config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_training(self):
        """Forward pass with targets (training mode)."""
        from nanogpt_int_type_var_runnable import GPT

        config = self._make_config()
        model = GPT(config)

        batch_size, seq_len = 2, 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, loss = model(idx, targets)

        self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, ())

    def test_generate(self):
        """Generate method produces tokens."""
        from nanogpt_int_type_var_runnable import GPT

        config = self._make_config()
        model = GPT(config)
        model.eval()

        batch_size = 1
        idx = torch.randint(0, config.vocab_size, (batch_size, 4))
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens=3)

        self.assertEqual(generated.shape[0], batch_size)
        self.assertEqual(generated.shape[1], 4 + 3)


class TestGPTFastIntTypeVarRuntime(unittest.TestCase):
    """Test that gptfast_int_type_var_runnable.py can be imported and sub-modules can run."""

    def _make_config(self):
        from gptfast_int_type_var_runnable import ModelArgs

        return ModelArgs(
            block_size=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            dim=32,
            n_local_heads=4,
            intermediate_size=64,
        )

    def test_import(self):
        """Module can be imported without errors."""
        import gptfast_int_type_var_runnable  # noqa: F401

    def test_instantiate_config(self):
        """ModelArgs dataclass can be created."""
        config = self._make_config()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.dim, 32)
        self.assertEqual(config.head_dim, 8)

    def test_instantiate_transformer(self):
        """Transformer model can be instantiated."""
        from gptfast_int_type_var_runnable import Transformer

        config = self._make_config()
        model = Transformer(config)
        self.assertIsNotNone(model)

    def test_rms_norm_forward(self):
        """RMSNorm forward pass works."""
        from gptfast_int_type_var_runnable import RMSNorm

        norm = RMSNorm(32)
        x = torch.randn(2, 16, 32)
        out = norm(x)
        self.assertEqual(out.shape, (2, 16, 32))

    def test_feed_forward(self):
        """FeedForward forward pass works."""
        from gptfast_int_type_var_runnable import FeedForward

        config = self._make_config()
        ff = FeedForward(config)
        x = torch.randn(2, 16, 32)
        out = ff(x)
        self.assertEqual(out.shape, (2, 16, 32))

    def test_precompute_freqs_cis(self):
        """precompute_freqs_cis produces correct shape."""
        from gptfast_int_type_var_runnable import precompute_freqs_cis

        cache = precompute_freqs_cis(seq_len=64, n_elem=8)
        self.assertEqual(cache.shape, (64, 4, 2))

    def test_apply_rotary_emb(self):
        """apply_rotary_emb works on a tensor."""
        from gptfast_int_type_var_runnable import apply_rotary_emb, precompute_freqs_cis

        batch, seq_len, n_heads, head_dim = 2, 16, 4, 8
        x = torch.randn(batch, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(seq_len=seq_len, n_elem=head_dim)
        out = apply_rotary_emb(x, freqs_cis)
        self.assertEqual(out.shape, (batch, seq_len, n_heads, head_dim))

    def test_kv_cache(self):
        """KVCache can be created and updated."""
        from gptfast_int_type_var_runnable import KVCache

        cache = KVCache(
            max_batch_size=2,
            max_seq_length=64,
            n_heads=4,
            head_dim=8,
        )
        input_pos = torch.arange(16)
        k_val = torch.randn(2, 4, 16, 8, dtype=torch.bfloat16)
        v_val = torch.randn(2, 4, 16, 8, dtype=torch.bfloat16)
        k_out, v_out = cache.update(input_pos, k_val, v_val)
        self.assertEqual(k_out.shape, (2, 4, 64, 8))
        self.assertEqual(v_out.shape, (2, 4, 64, 8))


if __name__ == "__main__":
    unittest.main()
