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

"""Gymnasium environment for dual YAM joint-space control."""

from __future__ import annotations

import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.utils.logging import get_logger

from .config import DualYamJointEnvConfig
from .control_runtime import YamControlRuntime
from .mock_backend import MockYamBackendFactory
from .types import DUAL_ARM_ACTION_DIM, YamBackendFactory


class DualYamJointEnv(gym.Env):
    """A 14-D absolute-joint environment for a pair of YAM follower arms.

    The fixed action and state order is
    ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``. Grippers use
    the canonical convention ``0=closed, 1=open``.

    Construction is side-effect free. Cameras and followers are opened lazily
    on the first :meth:`reset`, with cameras warmed before any CAN device.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        override_cfg: dict[str, Any] | None,
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
        *,
        backend_factory: YamBackendFactory | None = None,
        camera_factory: Any = create_camera,
        runtime: YamControlRuntime | None = None,
    ) -> None:
        super().__init__()
        override_values = dict(override_cfg or {})
        self.config = DualYamJointEnvConfig(**override_values)
        self.worker_info = worker_info
        self.hardware_info = hardware_info
        self.env_idx = int(env_idx)
        self._logger = get_logger()
        self._camera_factory = camera_factory
        self._cameras: list[BaseCamera] = []
        self._last_camera_frame: dict[str, np.ndarray] = {}
        self._last_camera_success_s: dict[str, float] = {}
        self._started = False
        self._start_failed = False
        self._closed = False
        self._num_steps = 0
        self._last_tick_s: float | None = None

        hardware_config = self._resolve_hardware_config(hardware_info, runtime)
        self._hardware_config = hardware_config
        if (
            not self.config.is_dummy
            and runtime is None
            and not {"joint_limit_min", "joint_limit_max"}.issubset(override_values)
        ):
            raise ValueError(
                "real YAM environments require explicit joint_limit_min and "
                "joint_limit_max values within the installed i2rt limits"
            )
        worker_node_rank = getattr(worker_info, "cluster_node_rank", None)
        hardware_node_rank = getattr(hardware_config, "node_rank", None)
        if (
            worker_node_rank is not None
            and hardware_node_rank is not None
            and worker_node_rank != hardware_node_rank
        ):
            raise ValueError(
                f"YAM hardware belongs to node {hardware_node_rank}, but the env "
                f"worker is on node {worker_node_rank}"
            )
        if runtime is None:
            if backend_factory is None:
                if self.config.is_dummy:
                    backend_factory = MockYamBackendFactory()
                else:
                    # This module performs no i2rt import. The real SDK is
                    # loaded by the backend only when followers connect.
                    from .i2rt_backend import I2RTYamBackendFactory

                    backend_factory = I2RTYamBackendFactory()
            runtime = YamControlRuntime(
                self.config,
                hardware_config,
                backend_factory,
            )
        self._runtime = runtime

        self._camera_specs = self._resolve_camera_specs(hardware_config)
        self._camera_names = (
            list(self.config.dummy_camera_names)
            if self.config.is_dummy
            else [spec.name for spec in self._camera_specs]
        )
        if not self._camera_names:
            raise ValueError("DualYamJointEnv requires at least one camera")

        arm_low = np.concatenate([self.config.joint_limit_min[0], np.array([0.0])])
        arm_high = np.concatenate([self.config.joint_limit_max[0], np.array([1.0])])
        right_low = np.concatenate([self.config.joint_limit_min[1], np.array([0.0])])
        right_high = np.concatenate([self.config.joint_limit_max[1], np.array([1.0])])
        self.action_space = gym.spaces.Box(
            low=np.concatenate([arm_low, right_low]).astype(np.float32),
            high=np.concatenate([arm_high, right_high]).astype(np.float32),
            dtype=np.float32,
        )
        frame_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(
                            low=self.action_space.low,
                            high=self.action_space.high,
                            shape=(DUAL_ARM_ACTION_DIM,),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    dict.fromkeys(self._camera_names, frame_space)
                ),
            }
        )

    @property
    def task_description(self) -> str:
        """Natural-language task prompt exposed to ``RealWorldEnv``."""
        return self.config.task_description

    @property
    def runtime(self) -> YamControlRuntime:
        """The single owner of follower and optional leader transports."""
        return self._runtime

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start hardware if needed and return measured state without moving."""
        super().reset(seed=seed)
        del options
        self._ensure_started()
        self._num_steps = 0
        self._last_tick_s = None
        return self._get_observation(), {"episode_phase": "pre"}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one bounded absolute-joint target and read the next state."""
        self._ensure_started()
        self._pace()
        result = self._runtime.command(action)
        try:
            observation = self._get_observation()
        except Exception:
            self._runtime.emergency_hold()
            raise
        self._num_steps += 1
        truncated = bool(
            not self.config.manual_episode_control_only
            and self._num_steps >= self.config.max_num_steps
        )
        info = {
            "accepted_action": result.accepted.astype(np.float32),
            "action_clipped": bool(result.clipped),
            "action_rejected": result.rejection_reason,
        }
        return observation, 0.0, False, truncated, info

    def teleop_tick(self, action: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply a leader target without consuming an episode step."""
        self._ensure_started()
        self._pace()
        result = self._runtime.command(action)
        try:
            observation = self._get_observation()
        except Exception:
            self._runtime.emergency_hold()
            raise
        return observation, {
            "accepted_action": result.accepted.astype(np.float32),
            "action_clipped": bool(result.clipped),
            "action_rejected": result.rejection_reason,
        }

    def get_hold_action(self, fallback_action: Any = None) -> np.ndarray:
        """Return current measured positions, never an all-zero fallback."""
        del fallback_action
        self._ensure_started()
        return self._runtime.read_state().as_vector().astype(np.float32)

    def observe(self) -> dict[str, Any]:
        """Return a fresh observation without dispatching a motion command."""
        self._ensure_started()
        return self._get_observation()

    def close(self) -> None:
        """Release all runtime and camera resources idempotently."""
        if self._closed:
            return
        self._start_failed = True
        errors: list[Exception] = []
        try:
            self._runtime.close()
        except Exception as error:  # pragma: no cover - hardware cleanup path
            errors.append(error)
        errors.extend(self._close_cameras())
        if errors:
            details = "; ".join(str(error) for error in errors)
            raise RuntimeError(
                f"failed to fully close DualYamJointEnv: {details}"
            ) from errors[0]
        self._started = False
        self._closed = True

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("cannot use a closed DualYamJointEnv")
        if self._start_failed:
            raise RuntimeError(
                "DualYamJointEnv startup previously failed; close and rebuild it"
            )
        if self._started:
            return
        try:
            if not self.config.is_dummy:
                self._open_and_warm_cameras()
            self._runtime.connect_followers()
            self._runtime.hold()
            self._started = True
        except Exception as start_error:
            self._start_failed = True
            try:
                self._runtime.close()
            except Exception as cleanup_error:  # pragma: no cover - hardware failure
                self._logger.error(
                    "YAM startup failed (%s), and cleanup also failed: %s",
                    start_error,
                    cleanup_error,
                )
            camera_errors = self._close_cameras()
            if camera_errors:  # pragma: no cover - hardware failure path
                self._logger.error(
                    "YAM camera cleanup also failed: %s",
                    "; ".join(str(error) for error in camera_errors),
                )
            raise

    def _get_observation(self) -> dict[str, Any]:
        state = self._runtime.read_state().as_vector().astype(np.float32)
        if self.config.is_dummy:
            frames = {
                name: np.zeros(
                    (self.config.image_height, self.config.image_width, 3),
                    dtype=np.uint8,
                )
                for name in self._camera_names
            }
        else:
            frames = self._read_camera_frames()
        return {"state": {"joint_position": state}, "frames": frames}

    def _open_and_warm_cameras(self) -> None:
        for camera_info in self._camera_specs:
            camera = self._camera_factory(camera_info)
            self._cameras.append(camera)
            camera.open()
        for camera in self._cameras:
            frame = camera.get_frame(timeout=self.config.camera_warmup_timeout_s)
            self._last_camera_frame[camera.name] = self._process_frame(frame)
            self._last_camera_success_s[camera.name] = time.monotonic()

    def _read_camera_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera in self._cameras:
            try:
                raw = camera.get_frame(timeout=self.config.camera_frame_timeout_s)
                frame = self._process_frame(raw)
                self._last_camera_frame[camera.name] = frame
                self._last_camera_success_s[camera.name] = time.monotonic()
            except Exception:
                if camera.name not in self._last_camera_frame:
                    raise
                stale_age_s = (
                    time.monotonic() - self._last_camera_success_s[camera.name]
                )
                if stale_age_s > self.config.camera_stale_timeout_s:
                    raise RuntimeError(
                        f"YAM camera {camera.name!r} has been stale for "
                        f"{stale_age_s:.3f}s"
                    )
                self._logger.warning(
                    "YAM camera %s missed a frame; reusing the last frame",
                    camera.name,
                )
                frame = self._last_camera_frame[camera.name]
            frames[camera.name] = frame.copy()
        return frames

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError(
                f"YAM RGB frame must have shape (H, W, >=3), got {array.shape}"
            )
        # Camera backends expose BGR; policy and dataset observations are RGB.
        rgb = np.ascontiguousarray(array[..., :3][..., ::-1], dtype=np.uint8)
        target_shape = (self.config.image_height, self.config.image_width)
        if rgb.shape[:2] != target_shape:
            import cv2

            rgb = cv2.resize(
                rgb,
                (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.asarray(rgb, dtype=np.uint8)

    def _pace(self) -> None:
        period_s = 1.0 / self.config.step_frequency
        now = time.perf_counter()
        if self._last_tick_s is not None:
            sleep_s = self._last_tick_s + period_s - now
            if sleep_s > 0:
                time.sleep(sleep_s)
        self._last_tick_s = time.perf_counter()

    def _close_cameras(self) -> list[Exception]:
        errors: list[Exception] = []
        failed: list[BaseCamera] = []
        for camera in reversed(self._cameras):
            try:
                camera.close()
            except Exception as error:  # pragma: no cover - hardware cleanup path
                errors.append(error)
                failed.append(camera)
                self._logger.exception("Failed to close YAM camera %s", camera.name)
        self._cameras = list(reversed(failed))
        failed_names = {camera.name for camera in failed}
        self._last_camera_frame = {
            name: frame
            for name, frame in self._last_camera_frame.items()
            if name in failed_names
        }
        self._last_camera_success_s = {
            name: timestamp
            for name, timestamp in self._last_camera_success_s.items()
            if name in failed_names
        }
        return errors

    def _resolve_hardware_config(self, hardware_info: Any, runtime: Any) -> Any:
        if hardware_info is not None:
            config = getattr(hardware_info, "config", hardware_info)
            for field_name in (
                "left_follower",
                "right_follower",
                "left_leader",
                "right_leader",
            ):
                if not hasattr(config, field_name):
                    raise TypeError(
                        f"DualYam hardware config is missing {field_name!r}"
                    )
            return config
        if not self.config.is_dummy and runtime is None:
            raise ValueError(
                "hardware_info is required unless is_dummy=true or a runtime is injected"
            )
        device = SimpleNamespace(channel="mock")
        return SimpleNamespace(
            left_follower=device,
            right_follower=device,
            left_leader=device,
            right_leader=device,
            cameras=[],
        )

    @staticmethod
    def _resolve_camera_specs(hardware_config: Any) -> list[CameraInfo]:
        specs: list[CameraInfo] = []
        for camera in getattr(hardware_config, "cameras", []) or []:
            if isinstance(camera, Mapping):
                values = camera
            else:
                values = vars(camera)
            resolution = tuple(values.get("resolution", (640, 480)))
            if bool(values.get("enable_depth", False)):
                raise ValueError(
                    "DualYamJointEnv currently exposes RGB only; depth must be disabled"
                )
            specs.append(
                CameraInfo(
                    name=str(values["name"]),
                    serial_number=str(values["serial"]),
                    camera_type=str(values.get("camera_type", "realsense")),
                    resolution=(int(resolution[0]), int(resolution[1])),
                    fps=int(values.get("fps", 30)),
                    enable_depth=bool(values.get("enable_depth", False)),
                )
            )
        return specs
