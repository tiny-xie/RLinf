# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime configuration for the RLinf-native dual YAM environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import NUM_ARM_JOINTS


def _default_joint_limit_min() -> list[list[float]]:
    return [[-float(np.pi)] * NUM_ARM_JOINTS for _ in range(2)]


def _default_joint_limit_max() -> list[list[float]]:
    return [[float(np.pi)] * NUM_ARM_JOINTS for _ in range(2)]


@dataclass
class DualYamJointEnvConfig:
    """Task-level settings; physical device calibration lives in hardware config."""

    is_dummy: bool = False
    task_description: str = "dual-arm YAM manipulation task"
    step_frequency: float = 30.0
    max_num_steps: int = 1000
    max_joint_delta: float = 0.08
    enforce_runtime_joint_limits: bool = True
    joint_limit_min: list[list[float]] = field(default_factory=_default_joint_limit_min)
    joint_limit_max: list[list[float]] = field(default_factory=_default_joint_limit_max)
    feedback_timeout_s: float = 0.25
    engage_duration_s: float = 2.0
    camera_warmup_timeout_s: float = 10.0
    camera_frame_timeout_s: float = 1.0
    camera_stale_timeout_s: float = 2.0
    image_height: int = 128
    image_width: int = 128
    dummy_camera_names: list[str] = field(
        default_factory=lambda: ["top_rgb", "left_rgb", "right_rgb"]
    )
    manual_episode_control_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.is_dummy, bool):
            raise TypeError("is_dummy must be a bool")
        if not isinstance(self.manual_episode_control_only, bool):
            raise TypeError("manual_episode_control_only must be a bool")
        if not isinstance(self.enforce_runtime_joint_limits, bool):
            raise TypeError("enforce_runtime_joint_limits must be a bool")
        self.step_frequency = float(self.step_frequency)
        self.max_joint_delta = float(self.max_joint_delta)
        self.feedback_timeout_s = float(self.feedback_timeout_s)
        self.engage_duration_s = float(self.engage_duration_s)
        self.camera_warmup_timeout_s = float(self.camera_warmup_timeout_s)
        self.camera_frame_timeout_s = float(self.camera_frame_timeout_s)
        self.camera_stale_timeout_s = float(self.camera_stale_timeout_s)
        self.image_height = int(self.image_height)
        self.image_width = int(self.image_width)
        self.max_num_steps = int(self.max_num_steps)
        self.joint_limit_min = np.asarray(
            self.joint_limit_min, dtype=np.float64
        ).reshape(2, NUM_ARM_JOINTS)
        self.joint_limit_max = np.asarray(
            self.joint_limit_max, dtype=np.float64
        ).reshape(2, NUM_ARM_JOINTS)
        scalar_values = {
            "step_frequency": self.step_frequency,
            "max_joint_delta": self.max_joint_delta,
            "feedback_timeout_s": self.feedback_timeout_s,
            "engage_duration_s": self.engage_duration_s,
            "camera_warmup_timeout_s": self.camera_warmup_timeout_s,
            "camera_frame_timeout_s": self.camera_frame_timeout_s,
            "camera_stale_timeout_s": self.camera_stale_timeout_s,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.step_frequency <= 0:
            raise ValueError("step_frequency must be positive")
        if self.max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive")
        if self.feedback_timeout_s <= 0:
            raise ValueError("feedback_timeout_s must be positive")
        if self.engage_duration_s < 0:
            raise ValueError("engage_duration_s must be non-negative")
        if self.camera_warmup_timeout_s <= 0:
            raise ValueError("camera_warmup_timeout_s must be positive")
        if self.camera_frame_timeout_s <= 0:
            raise ValueError("camera_frame_timeout_s must be positive")
        if self.camera_stale_timeout_s <= 0:
            raise ValueError("camera_stale_timeout_s must be positive")
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps must be positive")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if not np.all(self.joint_limit_min < self.joint_limit_max):
            raise ValueError(
                "every YAM joint lower limit must be below its upper limit"
            )
        if not np.all(np.isfinite(self.joint_limit_min)) or not np.all(
            np.isfinite(self.joint_limit_max)
        ):
            raise ValueError("YAM joint limits must be finite")
        self.task_description = str(self.task_description)
        names = [str(name) for name in self.dummy_camera_names]
        if not names or len(set(names)) != len(names):
            raise ValueError("dummy_camera_names must be non-empty and unique")
        self.dummy_camera_names = names


@dataclass
class YamLeaderInterventionConfig:
    """Teaching-handle button and episode-control behavior."""

    wait_for_record_button: bool = True
    sync_on_reset: bool = False
    preserve_sync_between_episodes: bool = False
    poll_frequency: float = 30.0
    button_debounce_s: float = 0.2
    unsynced_action_source: str = "hold"

    def __post_init__(self) -> None:
        if not isinstance(self.wait_for_record_button, bool):
            raise TypeError("wait_for_record_button must be a bool")
        if not isinstance(self.sync_on_reset, bool):
            raise TypeError("sync_on_reset must be a bool")
        if not isinstance(self.preserve_sync_between_episodes, bool):
            raise TypeError("preserve_sync_between_episodes must be a bool")
        self.poll_frequency = float(self.poll_frequency)
        self.button_debounce_s = float(self.button_debounce_s)
        if not np.isfinite(self.poll_frequency) or self.poll_frequency <= 0:
            raise ValueError("poll_frequency must be positive")
        if not np.isfinite(self.button_debounce_s) or self.button_debounce_s < 0:
            raise ValueError("button_debounce_s must be finite and non-negative")
        self.unsynced_action_source = str(self.unsynced_action_source).lower()
        if self.unsynced_action_source not in {"hold", "policy"}:
            raise ValueError("unsynced_action_source must be 'hold' or 'policy'")
