"""Tests for KV-cache sizing and context suggestions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.context_sizing import (  # noqa: E402
    kv_bytes_per_token,
    kv_cache_bytes,
    model_max_context,
    suggest_context_size,
)


def _qwen_like() -> dict:
    """Metadata shaped like the Qwen3.8-27B GGUF this app is used with."""
    return {
        "general.architecture": "qwen35",
        "qwen35.block_count": 65,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.embedding_length": 5120,
        "qwen35.context_length": 262144,
    }


class KvMathTests(unittest.TestCase):
    def test_bytes_per_token_matches_the_architecture(self):
        # 65 layers x 4 KV heads x (256 + 256) dims x 2 bytes (f16).
        self.assertEqual(kv_bytes_per_token(_qwen_like()), 266_240)

    def test_quantized_cache_costs_less_per_token(self):
        f16 = kv_bytes_per_token(_qwen_like(), kv_type="f16")
        q8 = kv_bytes_per_token(_qwen_like(), kv_type="q8_0")
        self.assertLess(q8, f16)
        # q8_0 packs 32 values into 34 bytes, so a little over half of f16.
        self.assertAlmostEqual(q8 / f16, 34 / 64, places=3)

    def test_cache_scales_linearly_with_context(self):
        md = _qwen_like()
        self.assertEqual(kv_cache_bytes(md, 131072), 266_240 * 131072)
        self.assertEqual(kv_cache_bytes(md, 16384) * 8, kv_cache_bytes(md, 131072))

    def test_missing_kv_head_count_is_treated_as_multi_head(self):
        md = _qwen_like()
        del md["qwen35.attention.head_count_kv"]
        # Falls back to head_count (24), so 6x the grouped-query cost.
        self.assertEqual(kv_bytes_per_token(md), 266_240 * 6)

    def test_head_dim_is_derived_when_not_declared(self):
        md = _qwen_like()
        del md["qwen35.attention.key_length"]
        del md["qwen35.attention.value_length"]
        # A fractional head dimension must not be silently rounded down.
        self.assertIsNone(kv_bytes_per_token(md))
        md["qwen35.embedding_length"] = 6144
        self.assertEqual(kv_bytes_per_token(md), 65 * 4 * 512 * 2)

    def test_architecture_prefix_is_recovered_from_block_count(self):
        # Some conversions omit general.architecture; the namespaced layer
        # count is still there and is enough to find the attention keys.
        md = _qwen_like()
        del md["general.architecture"]
        self.assertEqual(kv_bytes_per_token(md), 266_240)

    def test_context_length_uses_the_shared_architecture_fallbacks(self):
        # model_max_context delegates to gguf_metadata, which knows the common
        # prefixes even when general.architecture is absent.
        self.assertEqual(model_max_context({"llama.context_length": 32768}), 32768)

    def test_unreadable_metadata_yields_none_rather_than_a_guess(self):
        self.assertIsNone(kv_bytes_per_token({}))
        self.assertIsNone(kv_bytes_per_token({"general.architecture": "qwen35"}))
        self.assertIsNone(model_max_context({}))
        self.assertIsNone(kv_cache_bytes(_qwen_like(), 0))


class SuggestionTests(unittest.TestCase):
    def test_small_memory_budgets_never_exceed_the_ceiling(self):
        for budget in (0, 1, 1024**2):
            result = suggest_context_size(_qwen_like(), 7400, 2048, budget)
            self.assertLessEqual(result.suggested_ctx, result.memory_max)

    def test_workload_drives_the_suggestion(self):
        # 7,400 prompt + 2,048 reply rounds up to 16,384, far below both the
        # model's 262,144 and what memory would allow.
        result = suggest_context_size(
            _qwen_like(),
            prompt_tokens=7400,
            max_tokens=2048,
            available_bytes=120 * 1024**3,
        )
        self.assertEqual(result.suggested_ctx, 16384)
        self.assertEqual(result.bound_by, "workload")

    def test_model_maximum_clamps_an_oversized_workload(self):
        result = suggest_context_size(
            _qwen_like(),
            prompt_tokens=900_000,
            max_tokens=1024,
            available_bytes=10 * 1024**4,
        )
        self.assertEqual(result.suggested_ctx, 262144)
        self.assertEqual(result.bound_by, "model")

    def test_memory_clamps_below_the_workload(self):
        # 2 GiB of headroom cannot hold the ~4 GiB the workload wants.
        result = suggest_context_size(
            _qwen_like(),
            prompt_tokens=7400,
            max_tokens=2048,
            available_bytes=2 * 1024**3,
        )
        self.assertEqual(result.bound_by, "memory")
        self.assertLess(result.suggested_ctx, result.workload_ctx)
        self.assertTrue(result.notes)

    def test_suggestion_survives_an_unreadable_memory_budget(self):
        result = suggest_context_size(
            _qwen_like(), prompt_tokens=7400, max_tokens=2048, available_bytes=None
        )
        self.assertEqual(result.suggested_ctx, 16384)
        self.assertIsNone(result.memory_max)
