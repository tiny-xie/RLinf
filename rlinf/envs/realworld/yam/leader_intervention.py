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

"""Teaching-handle teleoperation and episode control for dual YAM."""

from __future__ import annotations

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np

from .config import YamLeaderInterventionConfig
from .dual_yam_joint_env import DualYamJointEnv


class DualYamLeaderIntervention(gym.Wrapper):
    """Drive followers from leaders and map handle buttons to collection state.

    Either leader's top button toggles follower synchronization. Either second
    button starts or successfully ends an episode. Button handling is rising-edge
    based, so holding a button cannot repeatedly toggle state.
    """

    def __init__(
        self,
        env: DualYamJointEnv,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.config = YamLeaderInterventionConfig(**dict(config or {}))
        self._recording = False
        self._sync_enabled = False
        self._preserve_sync_on_next_reset = False
        self._previous_buttons = (False, False)
        self._last_button_edge_s = [-math.inf, -math.inf]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset episode state, then optionally wait for the record button."""
        try:
            preserve_sync = bool(
                self._sync_enabled and self._preserve_sync_on_next_reset
            )
            self._preserve_sync_on_next_reset = False
            # A caller may reset early, before a normal done transition. Never
            # carry bilateral feedback across it unless the preceding manual
            # record boundary explicitly requested legacy continuous teleop.
            if self._sync_enabled and not preserve_sync:
                self._disable_sync()
            observation, info = self.env.reset(seed=seed, options=options)
            runtime = self._base_env.runtime
            runtime.connect_leaders()
            if not preserve_sync:
                runtime.release_leader_feedback()
            _, self._previous_buttons = runtime.read_leader_action()
            self._last_button_edge_s = [-math.inf, -math.inf]
            self._recording = not self.config.wait_for_record_button
            self._sync_enabled = preserve_sync

            if self.config.sync_on_reset and not self._sync_enabled:
                runtime.engage()
                self._sync_enabled = True

            skip_wait = bool((options or {}).get("skip_wait_for_start", False))
            if self.config.wait_for_record_button and not skip_wait:
                observation = self._wait_for_record_start()
                self._recording = True
                info = self._decorate_info(info, event="start", record_reset=True)
            else:
                info = self._decorate_info(info, event=None, record_reset=False)
            return observation, info
        except Exception as reset_error:
            try:
                self.env.close()
            except Exception as cleanup_error:  # pragma: no cover - hardware failure
                self._base_env._logger.error(
                    "YAM leader reset failed (%s), and cleanup also failed: %s",
                    reset_error,
                    cleanup_error,
                )
            raise

    def step(self, action: Any):
        """Apply leader, hold, or policy action according to sync configuration."""
        runtime = self._base_env.runtime
        try:
            leader_action, buttons = runtime.read_leader_action()
            top_edge, record_edge = self._button_edges(buttons)
            handoff_hold = False

            if top_edge:
                enabled_now = self._toggle_sync()
                if enabled_now:
                    leader_action, _ = runtime.read_leader_action()
                else:
                    handoff_hold = True

            if self._sync_enabled:
                effective_action = leader_action
                action_replaced = True
            elif self.config.unsynced_action_source == "hold" or handoff_hold:
                effective_action = self._base_env.get_hold_action()
                action_replaced = True
            else:
                effective_action = action
                action_replaced = False

            observation, reward, terminated, truncated, info = self.env.step(
                effective_action
            )
            if self._sync_enabled:
                runtime.apply_leader_feedback()
            event: str | None = None
            record_reset = False
            manual_done = False
            if record_edge:
                if self._recording:
                    event = "end_success"
                    reward = 1.0
                    terminated = True
                    manual_done = True
                else:
                    event = "start"
                    self._recording = True
                    record_reset = True

            preserve_manual_sync = bool(
                manual_done
                and not truncated
                and self._sync_enabled
                and self.config.preserve_sync_between_episodes
            )
            self._preserve_sync_on_next_reset = preserve_manual_sync
            # Automatic environment endings still hand ownership back
            # immediately. A configured manual recording boundary preserves
            # synchronization across the collector's following reset.
            if (
                (terminated or truncated)
                and self._sync_enabled
                and not preserve_manual_sync
            ):
                self._disable_sync()
        except Exception:
            self._sync_enabled = False
            self._preserve_sync_on_next_reset = False
            try:
                runtime.emergency_hold()
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to hold YAM followers after a wrapper error"
                )
            try:
                runtime.release_leader_feedback()
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to release YAM leaders after a step error"
                )
            raise

        accepted = np.asarray(
            info.get("accepted_action", effective_action), dtype=np.float32
        )
        if action_replaced:
            info["intervene_action"] = accepted
        else:
            info.pop("intervene_action", None)
        info["intervened"] = bool(action_replaced)
        info["manual_done"] = manual_done
        info = self._decorate_info(
            info,
            event=event,
            record_reset=record_reset,
        )
        return observation, reward, terminated, truncated, info

    def get_hold_action(self, fallback_action: Any = None) -> np.ndarray:
        """Delegate measured-pose hold generation to the base YAM env."""
        return self._base_env.get_hold_action(fallback_action)

    def _wait_for_record_start(self) -> dict[str, Any]:
        period_s = 1.0 / self.config.poll_frequency
        runtime = self._base_env.runtime
        while True:
            started_s = time.perf_counter()
            action, buttons = runtime.read_leader_action()
            top_edge, record_edge = self._button_edges(buttons)
            if top_edge:
                enabled_now = self._toggle_sync()
                if enabled_now:
                    action, _ = runtime.read_leader_action()
            if self._sync_enabled:
                observation, _ = self._base_env.teleop_tick(action)
                runtime.apply_leader_feedback()
            else:
                observation = self._base_env.observe()
            if record_edge:
                return observation
            remaining_s = period_s - (time.perf_counter() - started_s)
            if remaining_s > 0:
                time.sleep(remaining_s)

    def _toggle_sync(self) -> bool:
        runtime = self._base_env.runtime
        if self._sync_enabled:
            self._disable_sync()
        else:
            runtime.engage()
            self._sync_enabled = True
        return self._sync_enabled

    def _disable_sync(self) -> None:
        """Disable software ownership and release feedback even if hold fails."""
        runtime = self._base_env.runtime
        self._sync_enabled = False
        hold_error: Exception | None = None
        try:
            runtime.hold()
        except Exception as error:  # pragma: no cover - hardware failure path
            hold_error = error
        try:
            runtime.release_leader_feedback()
        except Exception as release_error:
            if hold_error is not None:
                raise RuntimeError(
                    "failed to hold followers and release YAM leader feedback"
                ) from release_error
            raise
        if hold_error is not None:
            raise hold_error

    def _button_edges(self, buttons: tuple[bool, bool]) -> tuple[bool, bool]:
        now_s = time.monotonic()
        edges_list = []
        for index in range(2):
            rising = bool(buttons[index] and not self._previous_buttons[index])
            accepted = bool(
                rising
                and now_s - self._last_button_edge_s[index]
                >= self.config.button_debounce_s
            )
            if accepted:
                self._last_button_edge_s[index] = now_s
            edges_list.append(accepted)
        self._previous_buttons = buttons
        return edges_list[0], edges_list[1]

    def _decorate_info(
        self,
        info: dict[str, Any],
        *,
        event: str | None,
        record_reset: bool,
    ) -> dict[str, Any]:
        phase = "rec" if self._recording else "pre"
        info.update(
            {
                "pre_record": not self._recording,
                "record_reset": bool(record_reset),
                "keyboard_phase": phase,
                "keyboard_event": event,
                "episode_control_phase": phase,
                "episode_control_event": event,
                "segment_advance": False,
                "yam_sync_enabled": bool(self._sync_enabled),
            }
        )
        return info

    @property
    def _base_env(self) -> DualYamJointEnv:
        return self.env.unwrapped
