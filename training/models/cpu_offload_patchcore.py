"""PatchCore training variant that offloads temporary embeddings to system RAM.

The final coreset and checkpoint are the same kind of artifacts as standard
Anomalib PatchCore. Only the temporary location used while collecting the full
pre-coreset embedding pool changes.
"""

from __future__ import annotations

import logging
from collections.abc import MutableSequence

import torch
from anomalib.models import Patchcore
from anomalib.models.components import KCenterGreedy
from lightning.pytorch.utilities.types import STEP_OUTPUT


logger = logging.getLogger(__name__)


def offload_latest_embedding(
    embedding_store: MutableSequence[torch.Tensor],
) -> None:
    """Detach and move the most recently collected embedding to CPU memory."""
    if not embedding_store:
        raise ValueError("Embedding store is empty after the PatchCore training step.")
    embedding_store[-1] = embedding_store[-1].detach().to("cpu")


def build_coreset_from_cpu_embeddings(
    embedding_store: MutableSequence[torch.Tensor],
    sampling_ratio: float,
    target_device: torch.device,
) -> torch.Tensor:
    """Consolidate CPU embeddings and run the standard coreset on the target device."""
    if not embedding_store:
        raise ValueError("Embedding store is empty. Cannot perform coreset selection.")
    if any(embedding.device.type != "cpu" for embedding in embedding_store):
        raise ValueError("CPU-offload mode received an embedding stored outside CPU memory.")

    embedding_count = sum(embedding.shape[0] for embedding in embedding_store)
    embedding_dimension = embedding_store[0].shape[1]
    byte_count = sum(
        embedding.nelement() * embedding.element_size()
        for embedding in embedding_store
    )
    consolidation_message = (
        f"CPU-offloaded embedding pool: {embedding_count} rows x "
        f"{embedding_dimension} features ({byte_count / 1024**3:.2f} GiB)."
    )
    logger.info(consolidation_message)
    print(consolidation_message)

    embedding_bank_cpu = torch.vstack(tuple(embedding_store))
    embedding_store.clear()

    if target_device.type == "cuda":
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(target_device)
        cuda_message = (
            "CUDA memory before coreset transfer: "
            f"{free_bytes / 1024**3:.2f} GiB free / "
            f"{total_bytes / 1024**3:.2f} GiB total."
        )
        logger.info(cuda_message)
        print(cuda_message)

    embedding_bank = embedding_bank_cpu.to(target_device)
    del embedding_bank_cpu
    sampler = KCenterGreedy(
        embedding=embedding_bank,
        sampling_ratio=sampling_ratio,
    )
    coreset = sampler.sample_coreset()
    result_message = (
        f"Final PatchCore memory bank: selected {coreset.shape[0]} of "
        f"{embedding_count} embeddings."
    )
    logger.info(result_message)
    print(result_message)
    return coreset


class CpuOffloadPatchcore(Patchcore):
    """Anomalib PatchCore with CPU storage for pre-coreset embeddings."""

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        """Run the standard step, then immediately offload its embedding."""
        output = super().training_step(batch, *args, **kwargs)
        offload_latest_embedding(self.model.embedding_store)
        return output

    def fit(self) -> None:
        """Build the standard coreset after moving the full pool back to the device."""
        self.model.memory_bank = build_coreset_from_cpu_embeddings(
            embedding_store=self.model.embedding_store,
            sampling_ratio=self.coreset_sampling_ratio,
            target_device=self.device,
        )
