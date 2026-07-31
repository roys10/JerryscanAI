from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LabSettings:
    workspace: Path
    registry_file: Path
    imports_dir: Path
    cache_dir: Path
    results_dir: Path
    preprocessing_configs_dir: Path

    @classmethod
    def from_environment(cls) -> "LabSettings":
        workspace = Path(
            os.getenv("JERRYSCAN_LAB_WORKSPACE", PROJECT_ROOT / ".model_lab")
        ).expanduser().resolve()
        return cls(
            workspace=workspace,
            registry_file=workspace / "registry.json",
            imports_dir=workspace / "imports",
            cache_dir=workspace / "preprocessing_cache",
            results_dir=workspace / "comparisons",
            preprocessing_configs_dir=PROJECT_ROOT
            / "training"
            / "preprocessing"
            / "configs",
        )

    def ensure_writable_directories(self) -> None:
        for path in (
            self.workspace,
            self.imports_dir,
            self.cache_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

