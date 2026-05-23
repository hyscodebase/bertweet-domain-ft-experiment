from __future__ import annotations

from typing import Any


def apply_full_finetuning(model: Any) -> Any:
    for parameter in model.parameters():
        parameter.requires_grad = True
    print("Full fine-tuning enabled: all model parameters are trainable.")
    return model
