from __future__ import annotations

from typing import Any


def _mark_classifier_head_trainable(model: Any) -> None:
    for name, parameter in model.named_parameters():
        if any(token in name for token in ("classifier", "score", "classification_head")):
            parameter.requires_grad = True


def _find_transformer_layers(model: Any) -> list[Any]:
    candidates = [
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("deberta", "encoder", "layer"),
    ]
    for path in candidates:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return list(current)
    base_model = getattr(model, "base_model", None)
    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    return list(layers) if layers is not None else []


def apply_bottleneck_adapter(
    model: Any,
    bottleneck_size: int = 64,
    nonlinearity: str = "gelu",
    train_classifier_head: bool = True,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Adapter fine-tuning requires the 'torch' package.") from exc

    class BottleneckAdapter(torch.nn.Module):
        def __init__(self, hidden_size: int, bottleneck: int, activation_name: str) -> None:
            super().__init__()
            self.down = torch.nn.Linear(hidden_size, bottleneck)
            self.activation = torch.nn.GELU() if activation_name == "gelu" else torch.nn.ReLU()
            self.up = torch.nn.Linear(bottleneck, hidden_size)

        def forward(self, hidden_states):
            return hidden_states + self.up(self.activation(self.down(hidden_states)))

    for parameter in model.parameters():
        parameter.requires_grad = False

    hidden_size = int(model.config.hidden_size)
    layers = _find_transformer_layers(model)
    if not layers:
        raise ValueError("Could not find transformer layers for adapter insertion.")

    for index, layer in enumerate(layers):
        adapter = BottleneckAdapter(hidden_size, bottleneck_size, nonlinearity)
        layer.add_module(f"b0_delta_adapter_{index}", adapter)

        def hook(module, _inputs, output, adapter_name=f"b0_delta_adapter_{index}"):
            layer_adapter = getattr(module, adapter_name)
            if isinstance(output, tuple):
                return (layer_adapter(output[0]),) + output[1:]
            return layer_adapter(output)

        layer.register_forward_hook(hook)

    if train_classifier_head:
        _mark_classifier_head_trainable(model)

    adapter_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "b0_delta_adapter" in name and parameter.requires_grad
    )
    print(f"Inserted {len(layers)} bottleneck adapters with {adapter_params} trainable adapter parameters.")
    return model


def apply_adapter(model: Any, bottleneck_size: int = 64, nonlinearity: str = "gelu", train_classifier_head: bool = True) -> Any:
    return apply_bottleneck_adapter(model, bottleneck_size, nonlinearity, train_classifier_head)
