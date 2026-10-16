# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Finetune API."""

from typing import Annotated

from fastapi import APIRouter, Depends

from server.core.dependencies import auth_admin
from server.services.finetune_service import FinetuneAdapterListOut, FinetuneService, get_finetune_service_instance

router = APIRouter(prefix="/admin/services/{service_id}/adapters", tags=["Finetune"])


@router.get("", summary="List fine-tuned adapters from the running deepfellow-finetune container.")
async def list_adapters(
    _: Annotated[str, Depends(auth_admin)],
    finetune: Annotated[tuple[FinetuneService, str], Depends(get_finetune_service_instance)],
) -> FinetuneAdapterListOut:
    """List fine-tuned adapters."""
    service, instance = finetune
    return await service.list_adapters(instance)
