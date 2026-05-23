from __future__ import annotations

from typing import Any


def detect_lora_target_modules(
    model: Any,
    include_attention_modules: bool = True,
    include_mlp_modules: bool = False,
) -> list[str]:
    suffixes: set[str] = set()
    module_names = [name for name, _module in model.named_modules()]
    if include_attention_modules:
        for suffix in ("query", "key", "value"):
            if any(name == suffix or name.endswith(f".{suffix}") for name in module_names):
                suffixes.add(suffix)
    if include_mlp_modules:
        for suffix in ("dense", "intermediate.dense", "output.dense"):
            if any(name == suffix or name.endswith(f".{suffix}") for name in module_names):
                suffixes.add(suffix)
    if not suffixes:
        suffixes.update({"query", "value"})
    return sorted(suffixes)


def _mark_classifier_head_trainable(model: Any) -> None:
    for name, parameter in model.named_parameters():
        if any(token in name for token in ("classifier", "score", "classification_head")):
            parameter.requires_grad = True


def apply_lora(
    model: Any,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    include_attention_modules: bool = True,
    include_mlp_modules: bool = False,
    train_classifier_head: bool = True,
) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA fine-tuning requires the 'peft' package.") from exc

    for parameter in model.parameters():
        parameter.requires_grad = False
    target_modules = detect_lora_target_modules(model, include_attention_modules, include_mlp_modules)
    print(f"LoRA target modules: {target_modules}")
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, config)
    if train_classifier_head:
        _mark_classifier_head_trainable(model)
    model.print_trainable_parameters()
    return model
