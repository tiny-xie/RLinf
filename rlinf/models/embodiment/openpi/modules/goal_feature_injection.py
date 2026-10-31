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

"""GIFT residual conditioning of observation tokens before Gemma."""

from __future__ import annotations

import copy

import torch
from torch import nn


class GoalFeatureInjection(nn.Module):
    """Project visual goal differences into a zero-initialized token residual.

    The pointwise convolution mixes channels independently at each patch. The
    caller supplies features from the same encoder and aligned camera views.
    Checkpoint loading copies the loaded observation projector into a new branch.
    """

    def __init__(
        self, observation_projector: nn.Linear, *, use_difference: bool = True
    ):
        super().__init__()
        self.use_difference = use_difference
        self.goal_projector = copy.deepcopy(observation_projector)
        width = observation_projector.out_features
        # Adding a zero branch must not change later random initialization.
        with torch.random.fork_rng(devices=[]):
            self.zero_conv = nn.Conv1d(width, width, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(
        self,
        observation_tokens: torch.Tensor,
        observation_features: torch.Tensor,
        goal_features: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Add a goal residual only for samples with valid observation and goal."""
        if (
            observation_features.shape != goal_features.shape
            or observation_features.shape[:2] != observation_tokens.shape[:2]
        ):
            raise ValueError("GIFT requires matching observation/goal patch grids.")
        valid = valid[:, None, None]
        # Mask before projection as well as after it: invalid features may be NaN,
        # and learned biases must not inject a goal for a masked sample.
        goal = torch.where(valid, goal_features, 0).to(self.goal_projector.weight.dtype)
        if self.use_difference:
            observed = torch.where(valid, observation_features, 0).to(goal.dtype)
            goal = goal - observed
        projected = self.goal_projector(goal)
        residual = self.zero_conv(projected.transpose(1, 2)).transpose(1, 2)
        residual = torch.where(valid, residual, 0)
        return (observation_tokens.to(residual.dtype) + residual).to(
            observation_tokens.dtype
        )
