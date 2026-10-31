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

"""Pi0 config for PyTorch, aligned with JAX models/pi0_config.py."""

from __future__ import annotations

import dataclasses

import torch

from rlinf.models.embodiment.openpi.modules import gemma, model, pointnet


@dataclasses.dataclass
class Pi0Config:
    dtype: str = "bfloat16"
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    pointnet_variant: pointnet.Variant = "pcd"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    pi05: bool = False
    discrete_state_input: bool | None = None
    pcd: bool = False
    # Disable training image augmentation to preserve the full 224x224 view.
    image_augmentation: bool = True
    # Identify LIBERO goal-image tokens in the third image slot.
    use_goal_role_embedding: bool = False
    # Keep the third image as a physical slot, but mask/zero it for pure AE goals.
    goal_in_prefix: bool = True
    # Joint prefix, or block-causal image/text/image ordering inspired by pi0.7.
    goal_prefix_mode: str = "joint"
    # Empty disables the independent Action Expert goal attention.
    goal_cross_attention_layers: tuple[int, ...] = ()
    goal_cross_attention_num_heads: int = 8
    goal_cross_attention_num_kv_heads: int = 1
    goal_cross_attention_head_dim: int = 64
    goal_cross_attention_seed: int = 0
    goal_cross_attention_expected_depth: int = 18
    # GIFT changes the main-view tokens; it does not append goal tokens to Gemma.
    goal_injection_mode: str = "none"
    gift_use_difference: bool = True
    # Full SFT, or train the adapter with the AE / only its output projection.
    gift_train_mode: str = "full"

    def __post_init__(self):
        if self.goal_injection_mode not in {"none", "gift"}:
            raise ValueError(
                f"Unknown goal_injection_mode: {self.goal_injection_mode!r}"
            )
        if self.gift_train_mode not in {"full", "adapter_expert", "adapter_head"}:
            raise ValueError(f"Unknown gift_train_mode: {self.gift_train_mode!r}")
        if self.goal_injection_mode == "none" and self.gift_train_mode != "full":
            raise ValueError("gift_train_mode requires goal_injection_mode='gift'")
        if self.goal_injection_mode == "gift" and (
            not self.pi05
            or self.pcd
            or self.image_augmentation
            or self.goal_in_prefix
            or self.goal_prefix_mode != "joint"
            or self.use_goal_role_embedding
            or self.goal_cross_attention_layers
        ):
            raise ValueError(
                "GIFT requires pi05=True, image_augmentation=False, "
                "goal_in_prefix=False, goal_prefix_mode='joint', no goal role/B attention, "
                "and pcd=False."
            )
        if self.goal_prefix_mode not in {
            "joint",
            "pi07_style",
            "pi07_style_reverse",
        }:
            raise ValueError(f"Unknown goal_prefix_mode: {self.goal_prefix_mode!r}")
        if self.goal_prefix_mode != "joint" and (
            not self.goal_in_prefix or not self.pi05 or self.pcd
        ):
            raise ValueError(
                "pi07-style prefixes require goal_in_prefix=True, pi05=True and pcd=False"
            )
        self.goal_cross_attention_layers = tuple(self.goal_cross_attention_layers)
        layers = self.goal_cross_attention_layers
        if len(set(layers)) != len(layers) or any(
            type(i) is not int or i < 0 for i in layers
        ):
            raise ValueError(
                "goal_cross_attention_layers must contain distinct nonnegative integers"
            )
        if layers:
            depths = [
                gemma.get_config(v).depth
                for v in (self.paligemma_variant, self.action_expert_variant)
            ]
            if depths != [self.goal_cross_attention_expected_depth] * 2:
                raise ValueError(
                    f"Goal attention expects {self.goal_cross_attention_expected_depth} layers in both experts, got {depths}"
                )
            if max(layers) >= depths[0]:
                raise ValueError(
                    f"Goal attention layer index must be below {depths[0]}"
                )
            if not self.pi05:
                raise ValueError(
                    "Independent goal attention currently requires pi05=True"
                )
        if (
            min(
                self.goal_cross_attention_num_heads,
                self.goal_cross_attention_num_kv_heads,
                self.goal_cross_attention_head_dim,
            )
            < 1
            or self.goal_cross_attention_num_heads
            % self.goal_cross_attention_num_kv_heads
        ):
            raise ValueError("Invalid goal attention head dimensions")
        if self.pi05 and self.max_token_len == 48:
            object.__setattr__(self, "max_token_len", 200)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    def fake_obs(self, batch_size: int = 1) -> model.Observation:
        image = torch.ones(batch_size, *model.IMAGE_RESOLUTION, 3)
        image_mask = torch.ones(batch_size, dtype=torch.bool)
        return model.Observation(
            images={
                "base_0_rgb": image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": image,
            },
            image_masks={
                "base_0_rgb": image_mask,
                "left_wrist_0_rgb": image_mask,
                "right_wrist_0_rgb": image_mask,
            },
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.ones(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
            pcd_xyz=torch.ones(batch_size, 16, 2025, 3) if self.pcd else None,
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        return torch.ones(batch_size, self.action_horizon, self.action_dim)
