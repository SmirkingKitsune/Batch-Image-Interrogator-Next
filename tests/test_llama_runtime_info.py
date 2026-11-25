"""Tests for managed and custom llama.cpp runtime resolution and status."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llama_provisioner import ProvisionConfig, current_platform  # noqa: E402
from core.llama_runtime_info import (  # noqa: E402
    RUNTIME_MODE_CUSTOM,
    RUNTIME_MODE_MANAGED,
    resolve_runtime_binary,
    runtime_mode,
    summarize_runtime,
)


def make_runtime(root: Path, accelerator: str = "cuda", **extra) -> Path:
    """Create a fake installed runtime with provisioner state."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    # Named for this host (llama-server.exe on Windows), which is what the
    # managed-runtime lookup searches for.
    exe_name = ProvisionConfig(provision_dir=root, platform=current_platform()).exe_name
    executable = bin_dir / exe_name
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    record = {
        "version": "b10828 (source, cuda)",
        "method": "source",
        "accelerator": accelerator,
        "executable": str(executable),
        "installed_at": "2026-09-06T15:50:12",
    }
    record.update(extra)
    (root / "active-runtime.json").write_text(json.dumps(record), encoding="utf-8")
    return executable


class TestRuntimeMode(unittest.TestCase):
    def test_empty_config_defaults_to_managed(self):
        self.assertEqual(runtime_mode({}), RUNTIME_MODE_MANAGED)

    def test_explicit_mode_is_respected(self):
        self.assertEqual(runtime_mode({"llama_runtime_mode": "custom"}), RUNTIME_MODE_CUSTOM)
        self.assertEqual(runtime_mode({"llama_runtime_mode": "managed"}), RUNTIME_MODE_MANAGED)

    def test_unknown_mode_falls_back_to_inference(self):
        config = {"llama_runtime_mode": "nonsense", "llama_binary_path": "/opt/llama-server"}
        self.assertEqual(runtime_mode(config), RUNTIME_MODE_CUSTOM)

    def test_legacy_config_with_a_path_is_treated_as_custom(self):
        # Settings saved before this option existed carried only a path. It must
        # keep launching that binary rather than being silently ignored.
        self.assertEqual(
            runtime_mode({"llama_binary_path": "/somewhere/else/llama-server"}),
            RUNTIME_MODE_CUSTOM,
        )

    def test_legacy_config_with_no_path_is_managed(self):
        self.assertEqual(runtime_mode({"llama_binary_path": ""}), RUNTIME_MODE_MANAGED)


class TestRuntimeResolution(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_custom_mode_uses_the_configured_path(self):
        config = {"llama_runtime_mode": "custom", "llama_binary_path": "/opt/custom/llama-server"}
        self.assertEqual(resolve_runtime_binary(config, self.root), "/opt/custom/llama-server")

    def test_managed_mode_ignores_a_stale_configured_path(self):
        """The point of managed mode: reprovisioning takes effect by itself.

        A stored absolute path goes stale the moment the backend is rebuilt, so
        managed mode resolves from disk instead of trusting what was saved.
        """
        executable = make_runtime(self.root)
        config = {
            "llama_runtime_mode": "managed",
            "llama_binary_path": "/an/old/path/from/last/year/llama-server",
        }
        self.assertEqual(resolve_runtime_binary(config, self.root), str(executable))

    def test_managed_mode_with_nothing_installed_resolves_empty(self):
        config = {"llama_runtime_mode": "managed"}
        self.assertEqual(resolve_runtime_binary(config, self.root), "")

    def test_custom_mode_with_no_path_resolves_empty(self):
        config = {"llama_runtime_mode": "custom", "llama_binary_path": "   "}
        self.assertEqual(resolve_runtime_binary(config, self.root), "")


class TestRuntimeSummary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_runtime_reports_not_installed(self):
        summary = summarize_runtime(self.root)
        self.assertFalse(summary["installed"])
        self.assertEqual(summary["executable"], "")

    def test_gpu_runtime_is_flagged_as_gpu(self):
        make_runtime(self.root, accelerator="cuda", cuda_architectures="121a-real")
        summary = summarize_runtime(self.root)
        self.assertTrue(summary["installed"])
        self.assertTrue(summary["is_gpu"])
        self.assertEqual(summary["accelerator"], "cuda")
        self.assertEqual(summary["cuda_architectures"], "121a-real")
        self.assertEqual(summary["fallback_from"], "", "a CUDA build is not a fallback")

    def test_metal_counts_as_gpu(self):
        make_runtime(self.root, accelerator="metal")
        self.assertTrue(summarize_runtime(self.root)["is_gpu"])

    def test_cpu_runtime_is_not_flagged_as_gpu(self):
        make_runtime(self.root, accelerator="cpu")
        summary = summarize_runtime(self.root)
        self.assertTrue(summary["installed"])
        self.assertFalse(summary["is_gpu"])

    def test_cpu_runtime_on_gpu_hardware_is_reported_as_a_fallback(self):
        # The ladder's last rung. The provisioning dialog warns once and closes;
        # this is what still says so while a slow batch is running.
        make_runtime(self.root, accelerator="cpu")
        summary = summarize_runtime(self.root, detect=lambda: "cuda")
        self.assertEqual(summary["fallback_from"], "cuda")

    def test_cpu_runtime_on_cpu_only_hardware_is_not_a_fallback(self):
        make_runtime(self.root, accelerator="cpu")
        summary = summarize_runtime(self.root, detect=lambda: "cpu")
        self.assertEqual(summary["fallback_from"], "", "CPU is the correct target here")
