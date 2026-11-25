"""Database, providers, model cache and app preferences for the bridge."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bridge.server import RpcError
from core.model_cache import ModelCacheManager
from core.onnx_providers import ProviderPreference


class SettingsFeature:
    """What the Database / Settings tab reads and changes."""

    def _init_settings(self) -> None:
        self._queue_processing = False
        self._cache_compile_job: Optional[Dict[str, Any]] = None
        self._cache_compile_cancel = threading.Event()

    # -- database -----------------------------------------------------------

    def rpc_db_stats(self) -> Dict[str, Any]:
        stats = self.database.get_statistics()
        location = self.database.get_db_location()
        try:
            size = Path(location).stat().st_size
        except OSError:
            size = 0
        return {
            **stats,
            "location": location,
            "size": size,
            "local": bool(self.database.use_local_db),
        }

    def rpc_db_set_mode(self, local: bool) -> Dict[str, Any]:
        self.database.set_local_mode(bool(local))
        if self.directory:
            self.database.switch_to_directory(str(self.directory))
        return self.rpc_db_stats()

    def rpc_db_vacuum(self) -> Dict[str, Any]:
        location = Path(self.database.get_db_location())
        before = location.stat().st_size if location.exists() else 0
        self.database.vacuum()
        after = location.stat().st_size if location.exists() else 0
        return {"before": before, "after": after, "saved": max(0, before - after)}

    def rpc_db_export(self, path: str) -> Dict[str, Any]:
        """Write images, models and interrogations to a JSON file."""
        target = Path(path or "")
        if not target.name:
            raise RpcError("Choose a file to export to.")
        location = self.database.get_db_location()
        # Read-only connection: an export must never lock the database for writers.
        uri = Path(location).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            images = [dict(row) for row in conn.execute(
                "SELECT id, file_path, file_hash, width, height, file_size FROM images"
            )]
            models = [dict(row) for row in conn.execute("SELECT * FROM models")]
            interrogations = []
            for row in conn.execute("SELECT * FROM interrogations"):
                entry = dict(row)
                for key in ("tags", "confidence_scores"):
                    if isinstance(entry.get(key), str):
                        try:
                            entry[key] = json.loads(entry[key])
                        except ValueError:
                            pass
                interrogations.append(entry)
        finally:
            conn.close()
        payload = {
            "exported_from": location,
            "images": images,
            "models": models,
            "interrogations": interrogations,
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return {"path": str(target), "images": len(images), "interrogations": len(interrogations)}

    # -- operation queue ----------------------------------------------------

    def rpc_db_queue(self) -> Dict[str, Any]:
        status = self.database.get_queue_status()
        return {**status, "processing": self._queue_processing}

    def rpc_db_process_queue(self) -> Dict[str, Any]:
        if self._queue_processing:
            raise RpcError("The queue is already being processed.", code="busy")
        self._queue_processing = True

        def run() -> None:
            try:
                success, failed = self.database.process_queued_operations()
            except Exception as exc:  # noqa: BLE001
                success, failed = 0, -1
                self.bus.publish("toast", {"level": "error", "message": f"Queue processing failed: {exc}"})
            finally:
                self._queue_processing = False
            self.bus.publish("db.queue_processed", {"success": success, "failed": failed, **self.rpc_db_queue()})

        self._spawn("db-queue", run)
        return {"processing": True}

    def rpc_db_clear_queue(self) -> Dict[str, Any]:
        cleared = self.database.clear_operation_queue()
        return {"cleared": cleared, **self.rpc_db_queue()}

    # -- providers -----------------------------------------------------------

    def _provider_status(self) -> Dict[str, Any]:
        status = self.provider_settings.get_status_info()
        preference = status["current_preference"]
        return {
            "tensorrt": status["tensorrt_available"],
            "cuda": status["cuda_available"],
            "available": status["all_providers"],
            "preference": preference.value,
            "preference_display": status["preference_display"],
            "description": status["preference_description"],
            "chain": self.provider_settings.get_provider_chain("cuda"),
            "options": [
                {"value": pref.value, "label": pref.display_name(), "description": pref.description()}
                for pref in ProviderPreference
            ],
        }

    def rpc_providers_state(self) -> Dict[str, Any]:
        return self._provider_status()

    def rpc_providers_refresh(self) -> Dict[str, Any]:
        self.provider_settings.get_available_providers(force_refresh=True)
        return self._provider_status()

    def rpc_providers_set(self, preference: str) -> Dict[str, Any]:
        self.provider_settings.set_preference(ProviderPreference.from_string(preference))
        return self._provider_status()

    # -- model cache ----------------------------------------------------------

    def _cache_manager(self) -> ModelCacheManager:
        cache_dir = Path(self.provider_settings.trt_options["trt_engine_cache_path"]).expanduser()
        return ModelCacheManager(cache_dir)

    def rpc_cache_models(self) -> Dict[str, Any]:
        manager = self._cache_manager()
        models: List[Dict[str, Any]] = []
        total_hf = total_trt = 0
        for model_type, infos in manager.get_all_models().items():
            for info in infos:
                total_hf += info.cache_size_bytes
                total_trt += info.tensorrt_engine_size
                models.append({
                    "id": info.model_id,
                    "type": model_type,
                    "name": info.display_name,
                    "cached": info.is_cached,
                    "onnx_cached": manager.get_onnx_model_path(info.model_id) is not None,
                    "size": info.cache_size_bytes,
                    "trt": info.has_tensorrt_engine,
                    "trt_size": info.tensorrt_engine_size,
                })
        gguf = []
        config = self._llama_config()
        for key in ("llama_model_path", "llama_mmproj_path"):
            path = config.get(key)
            if path:
                try:
                    gguf.append({"path": path, "name": Path(path).name, "size": Path(path).stat().st_size})
                except OSError:
                    pass
        return {
            "models": models,
            "huggingface": total_hf,
            "tensorrt": total_trt,
            "total": total_hf + total_trt,
            "gguf": gguf,
            "tensorrt_available": self.provider_settings.is_tensorrt_available(),
            "engine_caching": bool(self.provider_settings.trt_options.get("trt_engine_cache_enable")),
        }

    def _known_cache_models(self) -> Dict[str, str]:
        return dict.fromkeys(ModelCacheManager.WD_MODELS, "WD") | dict.fromkeys(
            ModelCacheManager.CAMIE_MODELS, "Camie"
        )

    def _validate_cache_selection(self, model_ids: List[str], known: Dict[str, str]) -> List[str]:
        if not isinstance(model_ids, list) or not model_ids:
            raise RpcError("Select at least one model.")
        for model_id in model_ids:
            if not isinstance(model_id, str) or model_id not in known:
                raise RpcError(f"Unknown model: {model_id}")
        return list(dict.fromkeys(model_ids))

    def _delete_cached_models(self, model_ids: List[str], tensorrt: bool) -> Dict[str, Any]:
        model_ids = self._validate_cache_selection(model_ids, self._known_cache_models())
        manager = self._cache_manager()
        with self._lock:
            if self._cache_compile_active() or self.model_loading:
                raise RpcError("Wait for model loading or TensorRT precompilation to finish.", code="busy")
            loaded = (self.model_info or {}).get("label")
            if loaded in model_ids:
                raise RpcError("Unload this model before deleting its cache.", code="busy")
            return {
                model_id: {
                    "deleted": manager.delete_hf_model_cache(model_id),
                    "tensorrt_deleted": manager.delete_tensorrt_engine(model_id) if tensorrt else False,
                }
                for model_id in model_ids
            }

    def rpc_cache_delete(self, model_id: str, tensorrt: bool = False) -> Dict[str, Any]:
        result = self._delete_cached_models([model_id], tensorrt)[model_id]
        return {**result, **self.rpc_cache_models()}

    def rpc_cache_delete_many(self, model_ids: List[str], tensorrt: bool = False) -> Dict[str, Any]:
        results = self._delete_cached_models(model_ids, tensorrt)
        return {"results": results, **self.rpc_cache_models()}

    def _cache_compile_active(self) -> bool:
        return bool(self._cache_compile_job and self._cache_compile_job["state"] in ("running", "cancelling"))

    def cache_compile_state(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._cache_compile_job) if self._cache_compile_job is not None else None

    def rpc_cache_compile(self, model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        known = self._known_cache_models()
        selected = model_ids is not None
        candidates = self._validate_cache_selection(model_ids, known) if selected else list(known)
        with self._lock:
            if self._cache_compile_active() or self.model_loading or self.interrogation_busy():
                raise RpcError("Wait for the active tagger or TensorRT job to finish.", code="busy")
            if not self.provider_settings.is_tensorrt_available():
                raise RpcError("The ONNX Runtime TensorRT execution provider is unavailable.", code="no_tensorrt")
            if not self.provider_settings.trt_options.get("trt_engine_cache_enable"):
                raise RpcError("Enable TensorRT engine caching before precompiling models.")
            manager = self._cache_manager()
            models = [(model_id, known[model_id], manager.get_onnx_model_path(model_id)) for model_id in candidates]
            missing = [model_id for model_id, _, path in models if path is None]
            if selected and missing:
                raise RpcError(f"Cached ONNX file missing for {', '.join(missing)}.", code="no_cached_onnx")
            work = [(model_id, family, path) for model_id, family, path in models if path is not None]
            if not work:
                raise RpcError("No cached ONNX models are available.", code="no_cached_onnx")

            self._cache_compile_cancel.clear()
            self._cache_compile_job = {
                "state": "running", "total": len(work), "current": None,
                "compiled": 0, "cached": 0, "errors": [],
            }
            if self._spawn("tensorrt-compile", lambda: self._run_cache_compile(work)) is None:
                self._cache_compile_job = None
                raise RpcError("The application is shutting down.")
            self._publish_cache_compile()
            return dict(self._cache_compile_job)

    def rpc_cache_cancel_compile(self) -> Dict[str, Any]:
        with self._lock:
            if self._cache_compile_active():
                self._cache_compile_cancel.set()
                self._publish_cache_compile(state="cancelling")
            return {"cancelling": self._cache_compile_cancel.is_set() and self._cache_compile_active()}

    def _publish_cache_compile(self, **changes: Any) -> None:
        # Call under the service lock to keep progress and cancellation in order.
        self._cache_compile_job = {**self._cache_compile_job, **changes}
        self.bus.publish("cache.compile", self._cache_compile_job)

    def _run_cache_compile(self, models: List[Tuple[str, str, Path]]) -> None:
        compiled = cached = 0
        errors = []
        try:
            from interrogators import CamieInterrogator, WDInterrogator

            for model_id, family, path in models:
                with self._lock:
                    if self._cache_compile_cancel.is_set() or self._stopping.is_set():
                        break
                    self._publish_cache_compile(current=model_id)
                try:
                    # Match the respective tagger's batch-of-one preprocessing.
                    if family == "WD":
                        size = WDInterrogator(model_id).target_size
                        shape = (1, size, size, 3)
                    else:
                        size = CamieInterrogator(model_id).target_size
                        shape = (1, 3, size, size)
                    if self.provider_settings.precompile_tensorrt(path, shape):
                        compiled += 1
                    else:
                        cached += 1
                except Exception as exc:  # noqa: BLE001 - continue to the next model
                    errors.append({"model_id": model_id, "message": str(exc)})
                with self._lock:
                    self._publish_cache_compile(compiled=compiled, cached=cached, errors=list(errors))
        finally:
            with self._lock:
                state = "cancelled" if compiled + cached + len(errors) < len(models) else "finished"
                self._publish_cache_compile(state=state, current=None)

    # -- app preferences ------------------------------------------------------

    def rpc_app_set_setting(self, key: str, value: Any) -> Dict[str, Any]:
        try:
            stored = self.ui_settings.set(key, value)
        except (KeyError, ValueError) as exc:
            raise RpcError(str(exc)) from exc
        return {"key": key, "value": stored}
