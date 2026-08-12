from __future__ import annotations

import asyncio
import io
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path

from fastapi import HTTPException, UploadFile

import backend.main as backend_main
from backend.inference.history import HistoryManager
from backend.inference.local_model import InferenceRuntimeError
from backend.inference.manager import ModelNotReadyError


def _result(status: str, score: float | None = None) -> dict:
    return {
        "status": status,
        "decision": status if status in {"PASS", "FAIL"} else None,
        "raw_image_score": score,
    }


class SQLiteHistoryTests(unittest.TestCase):
    def test_crud_filter_persistence_and_sql_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.db"
            manager = HistoryManager(path)
            pass_id = manager.save_session({"G01": _result("PASS", 12)}, "PASS", "m1")
            manager.save_session({"G01": _result("FAIL", 60)}, "FAIL", "m1")
            wrong_id = manager.save_session(
                {"G01": _result("WRONG_INPUT")}, "WRONG_INPUT", "m1"
            )

            self.assertEqual(manager.get_session(pass_id)["angles"]["G01"]["status"], "PASS")
            self.assertEqual(manager.get_session(wrong_id)["overall_status"], "WRONG_INPUT")
            self.assertEqual(len(manager.get_history(status="FAIL")), 1)
            stats = manager.get_stats()
            self.assertEqual(
                stats,
                {
                    "total": 3,
                    "decision_count": 2,
                    "passes": 1,
                    "faults": 1,
                    "wrong_inputs": 1,
                    "pass_rate": 50.0,
                },
            )
            self.assertEqual(HistoryManager(path).get_stats()["total"], 3)
            with closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_concurrent_writes_do_not_lose_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = HistoryManager(Path(temporary) / "history.db")
            errors = []

            def writer(index: int) -> None:
                try:
                    manager.save_session(
                        {"G01": _result("PASS", float(index))}, "PASS", "m1"
                    )
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(index,)) for index in range(24)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(manager.get_stats()["total"], 24)

    def test_unknown_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = HistoryManager(Path(temporary) / "history.db")
            with self.assertRaisesRegex(ValueError, "Unknown inspection status"):
                manager.save_session({"G01": {}}, "SYSTEM_ERROR")


class _UnavailableManager:
    runtime = None

    def inspect(self, *_args, **_kwargs):
        raise ModelNotReadyError("No model found: configure JERRYSCAN_MODEL_FOLDER")

    def get_required_angles(self):
        raise ModelNotReadyError("No model found: configure JERRYSCAN_MODEL_FOLDER")


class _FailingManager:
    runtime = None

    def inspect(self, *_args, **_kwargs):
        raise InferenceRuntimeError("PatchCore inference failed")

    def get_required_angles(self):
        return ["G01"]


class _ConcurrentManager:
    runtime = None

    def __init__(self):
        self.barrier = threading.Barrier(2)

    def get_required_angles(self):
        return ["G01", "G02"]

    def inspect(self, angle, *_args, **_kwargs):
        self.barrier.wait(timeout=2)
        return {**_result("PASS", 1.0), "angle": angle}


class _PartialManager:
    runtime = None

    def get_required_angles(self):
        return ["G01"]

    def get_configured_angles(self):
        return ["G01", "G02"]

    def get_unavailable_angles(self):
        return {
            "G02": {
                "stage": "artifact_validation",
                "error_type": "ModelFolderError",
                "detail": "Missing G02 checkpoint",
            }
        }

    def inspect(self, angle, *_args, **_kwargs):
        return {**_result("PASS", 1.0), "angle": angle}


class _FormRequest:
    def __init__(self, uploads=None):
        self.uploads = uploads if uploads is not None else {
            "G01": UploadFile(filename="G01.png", file=io.BytesIO(b"image-1")),
            "G02": UploadFile(filename="G02.png", file=io.BytesIO(b"image-2")),
        }

    async def form(self):
        return self.uploads


class _NoopAlertManager:
    def evaluate_session(self, *_args, **_kwargs):
        return None


class BackendHTTPTests(unittest.TestCase):
    def test_batch_dispatches_distinct_angles_concurrently(self):
        old_manager = backend_main.model_manager
        old_history = backend_main.history_manager
        old_alerts = backend_main.alert_manager
        try:
            with tempfile.TemporaryDirectory() as temporary:
                backend_main.model_manager = _ConcurrentManager()
                backend_main.history_manager = HistoryManager(Path(temporary) / "history.db")
                backend_main.alert_manager = _NoopAlertManager()

                response = asyncio.run(backend_main.inspect_batch(_FormRequest()))

                self.assertEqual(response["required_angles"], ["G01", "G02"])
                self.assertEqual(response["inspected_angles"], ["G01", "G02"])
                self.assertEqual(set(response["angles"]), {"G01", "G02"})
                self.assertEqual(response["overall_status"], "PASS")
        finally:
            backend_main.model_manager = old_manager
            backend_main.history_manager = old_history
            backend_main.alert_manager = old_alerts

    def test_partial_batch_requires_and_processes_only_available_angles(self):
        old_manager = backend_main.model_manager
        old_history = backend_main.history_manager
        old_alerts = backend_main.alert_manager
        try:
            with tempfile.TemporaryDirectory() as temporary:
                backend_main.model_manager = _PartialManager()
                backend_main.history_manager = HistoryManager(Path(temporary) / "history.db")
                backend_main.alert_manager = _NoopAlertManager()

                request = _FormRequest({
                    "G01": UploadFile(filename="G01.png", file=io.BytesIO(b"image-1")),
                })
                response = asyncio.run(backend_main.inspect_batch(request))

                self.assertEqual(response["required_angles"], ["G01"])
                self.assertEqual(response["inspected_angles"], ["G01"])
                self.assertEqual(response["configured_angles"], ["G01", "G02"])
                self.assertEqual(set(response["unavailable_angles"]), {"G02"})
                self.assertEqual(set(response["angles"]), {"G01"})
        finally:
            backend_main.model_manager = old_manager
            backend_main.history_manager = old_history
            backend_main.alert_manager = old_alerts

    def test_batch_processes_and_records_only_uploaded_available_angle(self):
        old_manager = backend_main.model_manager
        old_history = backend_main.history_manager
        old_alerts = backend_main.alert_manager
        try:
            with tempfile.TemporaryDirectory() as temporary:
                backend_main.model_manager = _ConcurrentManager()
                backend_main.history_manager = HistoryManager(Path(temporary) / "history.db")
                backend_main.alert_manager = _NoopAlertManager()
                # This test uses one upload, so avoid the two-party synchronization
                # used by the concurrency-specific fake manager.
                backend_main.model_manager.barrier = threading.Barrier(1)
                request = _FormRequest({
                    "G02": UploadFile(filename="G02.png", file=io.BytesIO(b"image-2")),
                })

                response = asyncio.run(backend_main.inspect_batch(request))

                self.assertEqual(response["inspected_angles"], ["G02"])
                self.assertEqual(set(response["angles"]), {"G02"})
                saved = backend_main.history_manager.get_session(response["session_id"])
                self.assertEqual(set(saved["angles"]), {"G02"})
        finally:
            backend_main.model_manager = old_manager
            backend_main.history_manager = old_history
            backend_main.alert_manager = old_alerts

    def test_empty_batch_is_http_400_and_not_persisted(self):
        old_manager = backend_main.model_manager
        old_history = backend_main.history_manager
        try:
            with tempfile.TemporaryDirectory() as temporary:
                backend_main.model_manager = _ConcurrentManager()
                backend_main.history_manager = HistoryManager(Path(temporary) / "history.db")

                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(backend_main.inspect_batch(_FormRequest({})))

                self.assertEqual(caught.exception.status_code, 400)
                self.assertIn("No images provided", str(caught.exception.detail))
                self.assertEqual(backend_main.history_manager.get_stats()["total"], 0)
        finally:
            backend_main.model_manager = old_manager
            backend_main.history_manager = old_history

    def test_batch_rejects_unavailable_uploaded_angle(self):
        old_manager = backend_main.model_manager
        old_history = backend_main.history_manager
        try:
            with tempfile.TemporaryDirectory() as temporary:
                backend_main.model_manager = _PartialManager()
                backend_main.history_manager = HistoryManager(Path(temporary) / "history.db")
                request = _FormRequest({
                    "G02": UploadFile(filename="G02.png", file=io.BytesIO(b"image-2")),
                })

                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(backend_main.inspect_batch(request))

                self.assertEqual(caught.exception.status_code, 503)
                self.assertIn("G02: Missing G02 checkpoint", str(caught.exception.detail))
                self.assertEqual(backend_main.history_manager.get_stats()["total"], 0)
        finally:
            backend_main.model_manager = old_manager
            backend_main.history_manager = old_history

    def test_batch_rejects_unknown_angle(self):
        old_manager = backend_main.model_manager
        old_history = backend_main.history_manager
        try:
            with tempfile.TemporaryDirectory() as temporary:
                backend_main.model_manager = _PartialManager()
                backend_main.history_manager = HistoryManager(Path(temporary) / "history.db")
                request = _FormRequest({
                    "G99": UploadFile(filename="G99.png", file=io.BytesIO(b"image")),
                })

                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(backend_main.inspect_batch(request))

                self.assertEqual(caught.exception.status_code, 400)
                self.assertIn("Unknown camera angle fields: G99", str(caught.exception.detail))
                self.assertEqual(backend_main.history_manager.get_stats()["total"], 0)
        finally:
            backend_main.model_manager = old_manager
            backend_main.history_manager = old_history

    def test_configured_but_unavailable_angle_is_http_503(self):
        old_manager = backend_main.model_manager
        try:
            backend_main.model_manager = _PartialManager()
            upload = UploadFile(filename="G02.png", file=io.BytesIO(b"image"))
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(backend_main.inspect_image("G02", upload))
            self.assertEqual(caught.exception.status_code, 503)
            self.assertIn("configured but unavailable", str(caught.exception.detail))
        finally:
            backend_main.model_manager = old_manager

    def _assert_http_error(self, manager, expected_status: int) -> None:
        old_manager = backend_main.model_manager
        old_history = backend_main.history_manager
        try:
            with tempfile.TemporaryDirectory() as temporary:
                backend_main.model_manager = manager
                backend_main.history_manager = HistoryManager(Path(temporary) / "history.db")
                upload = UploadFile(filename="G01.png", file=io.BytesIO(b"image"))
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(backend_main.inspect_image("G01", upload))
                self.assertEqual(caught.exception.status_code, expected_status)
                self.assertEqual(backend_main.history_manager.get_stats()["total"], 0)
        finally:
            backend_main.model_manager = old_manager
            backend_main.history_manager = old_history

    def test_missing_model_is_http_503_and_not_persisted(self):
        self._assert_http_error(_UnavailableManager(), 503)

    def test_runtime_failure_is_http_500_and_not_persisted(self):
        self._assert_http_error(_FailingManager(), 500)


if __name__ == "__main__":
    unittest.main()
