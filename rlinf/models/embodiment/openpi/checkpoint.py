# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint helpers for OpenPI."""

from __future__ import annotations

import pathlib
from typing import Any

from rlinf.utils.logging import get_logger

logger = get_logger()

FULL_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)

_FSDP_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
)

_BARE_PI0_PREFIXES = (
    "llm.",
    "img.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
    "pointnet.",
    "gift.",
)

_OLD_OPENPI_PREFIX = "paligemma_with_expert."
_OLD_WRAPPER_MODEL_PREFIX = "model."


def _missing_pi0_keys(missing_keys) -> list[str]:
    """Return missing keys that belong to the Pi0 backbone, not extra heads."""
    return [key for key in missing_keys if key.startswith(_BARE_PI0_PREFIXES)]


def resolve_model_safetensors(model_path: Any) -> pathlib.Path | None:
    """Resolve a base ``model.safetensors`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    weights_path = path / "model.safetensors"
    return weights_path if weights_path.exists() else None


def resolve_full_weights(model_path: Any) -> pathlib.Path | None:
    """Resolve an RLinf FSDP ``full_weights.pt`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".pt"):
        return path
    for rel_path in FULL_WEIGHTS_CANDIDATES:
        candidate = path / rel_path
        if candidate.exists():
            return candidate
    return None


def _normalize_key(key: str) -> str:
    while True:
        for prefix in _FSDP_WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def _normalize_state_dict(state_dict):
    """Map checkpoint keys onto the inherited-Pi0 layout.

    New checkpoints store bare Pi0 keys (``llm.*``). Older wrapper checkpoints
    stored the same tensors under ``model.llm.*``; strip that prefix. RLT and
    extra heads (``rlt_module.*``, ``value_head.*``, …) stay as-is.
    """
    normalized = {}
    for key, tensor in state_dict.items():
        key = _normalize_key(key)
        if key.startswith(_OLD_WRAPPER_MODEL_PREFIX):
            rest = key[len(_OLD_WRAPPER_MODEL_PREFIX) :]
            # Only strip when the remainder is a Pi0 module (or nested FSDP
            # leftover). Algorithm heads never lived under ``model.``.
            if (
                rest.startswith(_BARE_PI0_PREFIXES)
                or rest.startswith(_FSDP_WRAPPER_PREFIXES)
                or rest
                in {
                    "goal_role_embedding",
                    "goal_conditioning_spec",
                    "gift_conditioning_spec",
                }
            ):
                key = rest
        if key in normalized:
            raise ValueError(
                f"Duplicate checkpoint key after prefix normalization: {key!r}."
            )
        normalized[key] = tensor
    return normalized


def _convert_official_openpi_keys(state_dict, source) -> dict:
    """Rewrite ``paligemma_with_expert.*`` keys onto the inherited-Pi0 layout.

    Extra heads (``value_head.*``, ``rlt_module.*``, …) are kept as-is.
    """
    if not any(key.startswith(_OLD_OPENPI_PREFIX) for key in state_dict):
        return state_dict

    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import (
        old_to_new_state_dict,
    )

    converted = old_to_new_state_dict(state_dict)
    for key, tensor in state_dict.items():
        if key.startswith(_OLD_OPENPI_PREFIX) or key in converted:
            continue
        converted[key] = tensor
    logger.info(
        "openpi: converted OpenPI PyTorch checkpoint keys from %s in memory",
        source,
    )
    return converted


def _prepare_goal_role_weights(
    model: Any, state_dict: dict[str, Any], *, allow_missing: bool
) -> dict[str, Any]:
    """Initialize a new training role or omit its weights when disabled."""
    import torch

    key = "goal_role_embedding"
    role = getattr(model, key, None)
    if role is None:
        if key in state_dict:
            state_dict = {
                name: value for name, value in state_dict.items() if name != key
            }
            logger.info("openpi: goal role embedding disabled; omitting its weights")
    elif key not in state_dict:
        if not allow_missing:
            raise RuntimeError(
                "Goal role embedding is enabled, but the checkpoint has no "
                "goal_role_embedding weights. Evaluate a role-trained checkpoint "
                "or set openpi.use_goal_role_embedding=False for the original mode."
            )
        with torch.no_grad():
            role.zero_()
        logger.info("openpi: initialized missing goal role embedding to zero for SFT")
    return state_dict


def load_full_weights(
    model,
    weights_path,
    *,
    expect_rlt: bool,
    allow_missing_goal_role: bool = True,
    allow_initialize_gift: bool = False,
) -> None:
    """Load an RLinf ``full_weights.pt`` checkpoint into a Pi0 (or subclass)."""
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = _convert_official_openpi_keys(
        _normalize_state_dict(as_state_dict(loaded)),
        weights_path,
    )
    state_dict = _prepare_goal_attention_weights(
        model,
        state_dict,
        allow_initialize=allow_initialize_gift
        and getattr(model, "gift", None) is not None,
    )
    state_dict = _prepare_goal_role_weights(
        model, state_dict, allow_missing=allow_missing_goal_role
    )
    if expect_rlt and not any(key.startswith("rlt_module.") for key in state_dict):
        raise ValueError(
            "openpi RLT checkpoint has no rlt_module.* weights. "
            "Stage2 must consume a Stage1 checkpoint trained with openpi.use_rlt=True."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    matched = len(state_dict) - len(unexpected)
    if matched <= 0:
        raise RuntimeError(
            f"No tensors from {weights_path} matched the openpi model. "
            "This usually means the checkpoint is still in the legacy official "
            "OpenPI PyTorch key layout."
        )
    missing_pi0 = _missing_pi0_keys(missing)
    if missing_pi0:
        raise RuntimeError(
            f"Pi0 tensors missing from {weights_path}: {missing_pi0[:8]}"
        )
    if expect_rlt and any(key.startswith("rlt_module.") for key in missing):
        raise RuntimeError(
            f"RLT checkpoint {weights_path} did not load all rlt_module weights; "
            f"missing={missing[:8]}"
        )

    if missing or unexpected:
        logger.warning(
            "openpi: loaded checkpoint %s with strict=False "
            "(matched=%d missing=%d unexpected=%d)",
            weights_path,
            matched,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info("openpi: loaded full checkpoint from %s", weights_path)


def load_base_safetensors(
    model,
    safetensors_path,
    *,
    allow_missing_goal_role: bool = True,
    allow_missing_goal_attention: bool = False,
) -> None:
    """Load a base checkpoint, accepting new and legacy OpenPI layouts."""
    import safetensors.torch

    state_dict = _convert_official_openpi_keys(
        safetensors.torch.load_file(str(safetensors_path), device="cpu"),
        safetensors_path,
    )
    state_dict = _prepare_goal_attention_weights(
        model, state_dict, allow_initialize=allow_missing_goal_attention
    )
    state_dict = _prepare_goal_role_weights(
        model, state_dict, allow_missing=allow_missing_goal_role
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys loading {safetensors_path}: {unexpected[:8]}"
        )
    missing_pi0 = _missing_pi0_keys(incompatible.missing_keys)
    if missing_pi0:
        raise RuntimeError(
            f"Pi0 tensors missing from {safetensors_path}: {missing_pi0[:8]}"
        )
    if incompatible.missing_keys:
        logger.info(
            "openpi: loaded base safetensors %s; leaving %d extra module "
            "tensors at their initial values (%s...)",
            safetensors_path,
            len(incompatible.missing_keys),
            incompatible.missing_keys[:4],
        )


def _prepare_goal_attention_weights(
    model: Any, state_dict: dict[str, Any], *, allow_initialize: bool
) -> dict[str, Any]:
    """Allow new branches from base weights only; reject partial or wrong routes."""
    import torch

    expected = {
        name: value
        for name, value in model.state_dict().items()
        if ".goal_attention." in name
        or name.startswith("gift.")
        or name in {"goal_conditioning_spec", "gift_conditioning_spec"}
    }
    incoming = {
        name: value
        for name, value in state_dict.items()
        if ".goal_attention." in name
        or name.startswith("gift.")
        or name in {"goal_conditioning_spec", "gift_conditioning_spec"}
    }
    if not incoming and not expected:
        return state_dict
    if not incoming and expected and allow_initialize:
        if getattr(model, "gift", None) is not None:
            # The constructor sees a fresh SigLIP head, often all zeros. Copy
            # checkpoint weights here, not constructor weights, for adaptation.
            for key in model.gift.goal_projector.state_dict():
                source_key = "img.head." + key
                if source_key not in state_dict:
                    raise RuntimeError(f"Cannot initialize GIFT: missing {source_key}")
                expected["gift.goal_projector." + key] = state_dict[source_key].clone()
        logger.info(
            "openpi: initializing %d missing goal-conditioning tensors from base",
            len(expected),
        )
        return {**state_dict, **expected}
    if expected.keys() != incoming.keys():
        raise RuntimeError(
            "Goal attention checkpoint/config mismatch: "
            f"missing={sorted(expected.keys() - incoming.keys())}, "
            f"unexpected={sorted(incoming.keys() - expected.keys())}. "
            "Use matching goal YAML for evaluation/resume; initialize new branches from base safetensors for SFT."
        )
    if not torch.equal(
        incoming["goal_conditioning_spec"].cpu(),
        expected["goal_conditioning_spec"].cpu(),
    ):
        raise RuntimeError(
            "Goal conditioning configuration differs from checkpoint; match goal_prefix_mode, goal_in_prefix, role, layers and head dimensions in YAML."
        )
    if "gift_conditioning_spec" in expected and not torch.equal(
        incoming["gift_conditioning_spec"].cpu(),
        expected["gift_conditioning_spec"].cpu(),
    ):
        raise RuntimeError(
            "GIFT checkpoint/config mismatch: match gift_use_difference."
        )
    return state_dict
