"""TensorRT engine publication and reuse for cached ONNX models."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.model_cache import ModelCacheManager, precompiled_engine_dir
from core.onnx_providers import ONNXProviderSettings, ProviderPreference, TENSORRT_PROVIDER


@pytest.fixture
def cached_model(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    monkeypatch.setattr("core.model_cache.HF_CACHE_DIR", hub)
    model_id = ModelCacheManager.WD_MODELS[0]
    repo = hub / ("models--" + model_id.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("current", encoding="utf-8")
    for revision in ("stale", "current"):
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "model.onnx").touch()
    cache = tmp_path / "engines"
    monkeypatch.setitem(ONNXProviderSettings.DEFAULT_TRT_OPTIONS, "trt_engine_cache_path", str(cache))
    manager = ModelCacheManager(cache)
    settings = ONNXProviderSettings(str(tmp_path / "settings.json"))
    settings._available_providers = [TENSORRT_PROVIDER, "CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(settings, "get_session_options", lambda: "same-optimization-options")
    return settings, manager, model_id, repo / "snapshots" / "current" / "model.onnx"


def test_precompile_publishes_and_inference_reuses_the_same_snapshot(cached_model, monkeypatch):
    settings, manager, model_id, model_path = cached_model
    seen = []

    class Session:
        def __init__(self, path, *, sess_options, providers, provider_options):
            self.providers = providers
            self.cache = Path(provider_options[0]["trt_engine_cache_path"])
            seen.append((path, sess_options, providers, provider_options))

        def get_providers(self):
            return self.providers

        def disable_fallback(self):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="image", shape=["batch", 448, 448, 3], type="tensor(float)")]

        def run(self, _outputs, feed):
            assert feed["image"].shape == (1, 448, 448, 3)
            assert feed["image"].dtype == np.float32
            (self.cache / "fixture.engine").write_bytes(b"engine")
            (self.cache / "fixture.profile").write_bytes(b"profile")

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    assert manager.get_onnx_model_path(model_id) == model_path
    assert settings.precompile_tensorrt(model_path, (1, 448, 448, 3))
    assert not settings.precompile_tensorrt(model_path, (1, 448, 448, 3))

    engine_dir = precompiled_engine_dir(manager.tensorrt_cache_dir, model_path)
    assert manager.get_tensorrt_engine_info(model_id) == (True, engine_dir, len(b"engineprofile"))
    settings.preference = ProviderPreference.TENSORRT_CUDA_CPU
    assert settings.create_inference_session(str(model_path)).cache == engine_dir
    first, inference = seen
    assert first[:3] == inference[:3]
    assert first[3][1:] == inference[3][1:]
    first[3][0].pop("trt_engine_cache_path")
    inference[3][0].pop("trt_engine_cache_path")
    assert first[3][0] == inference[3][0]

    assert manager.delete_tensorrt_engine(model_id)
    assert not engine_dir.exists()
    assert settings.create_inference_session(str(model_path)).cache == manager.tensorrt_cache_dir


@pytest.mark.parametrize("fallback", [True, False])
def test_precompile_cleans_partial_files_and_rejects_fallback(cached_model, monkeypatch, fallback):
    settings, manager, model_id, model_path = cached_model

    class Session:
        def __init__(self, _path, *, sess_options, providers, provider_options):
            self.cache = Path(provider_options[0]["trt_engine_cache_path"])
            self.providers = providers[1:] if fallback else providers

        def get_providers(self):
            return self.providers

        def disable_fallback(self):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="image", shape=[1, 448, 448, 3], type="tensor(float)")]

        def run(self, _outputs, _feed):
            (self.cache / "partial.engine").touch()
            raise RuntimeError("warmup failed")

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    with pytest.raises(RuntimeError, match="fell back" if fallback else "warmup failed"):
        settings.precompile_tensorrt(model_path, (1, 448, 448, 3))
    engine_dir = precompiled_engine_dir(manager.tensorrt_cache_dir, model_path)
    assert not engine_dir.exists()
    assert not list(engine_dir.parent.glob(".compile-*"))
    assert not manager.get_tensorrt_engine_info(model_id)[0]
