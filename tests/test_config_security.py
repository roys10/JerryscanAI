from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.inference.config import ConfigManager


class ConfigSecurityTests(unittest.TestCase):
    def test_smtp_password_is_environment_only_and_never_persisted_or_returned(self):
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SMTP_PASSWORD": "unit-test-value"}, clear=False
        ):
            path = Path(directory) / "settings.json"
            manager = ConfigManager(str(path))
            public = manager.update({"smtp": {"server": "smtp.example", "port": 587, "user": "sender", "password": "submitted-value"}})
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("password", stored["smtp"])
            self.assertNotIn("password", public["smtp"])
            self.assertTrue(public["smtp"]["password_configured"])
            self.assertEqual(manager.get("smtp")["password"], os.environ["SMTP_PASSWORD"])


if __name__ == "__main__":
    unittest.main()
