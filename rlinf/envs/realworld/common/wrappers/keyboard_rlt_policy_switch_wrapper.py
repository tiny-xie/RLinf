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

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardRLTPolicySwitchWrapper(gym.Wrapper):
    """Pedal control for real-world RLT rollouts.

    ``a`` starts an episode under the frozen VLA. While it is running, another
    ``a`` marks failure, each ``b`` press toggles control between the VLA and
    the Stage 2 MLP actor, and ``c`` marks success. The outer real-world timeout
    remains a fallback failure condition.
    """

    IDLE_POLL_S = 0.05
    PEDAL_DEBOUNCE_S = 0.2
    WAIT_HEARTBEAT_S = 10.0

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._running = False
        self._actor_active = False
        self._awaiting_reset = False
        self._last_obs: Any = None
        self._last_press_ts: dict[str, float] = {}

    @property
    def rlt_switch_flags(self) -> bool:
        return self._actor_active

    def reset(self, *, seed=None, options=None):
        self._running = False
        self._actor_active = False
        self._awaiting_reset = False
        self._last_press_ts.clear()
        self.listener.pop_pressed_keys()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs

        self._log_info(
            "RLT rollout is idle. Arrange the scene, then press pedal 'a' "
            "to start under the frozen VLA. Press pedal 'b' to toggle between "
            "the VLA and Stage 2 MLP actor. During rollout, press pedal 'a' "
            "again to mark failure or pedal 'c' to mark success."
        )
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self._log_info("Still waiting for pedal 'a' to start RLT rollout...")
            for key in self.listener.pop_pressed_keys():
                if not self._accept_keypress(key):
                    continue
                if key == "a":
                    self._start_rollout()
                    return obs, self._with_rlt_info(info, event="start")

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Hold the robot at the controller's last target. After an outcome,
            # ignore pedals until outer auto-reset calls reset(), which blocks
            # waiting for the next 'a'.
            time.sleep(self.IDLE_POLL_S)
            if self._awaiting_reset:
                self.listener.pop_pressed_keys()
                return self._idle_response(event="awaiting_reset")

            for key in self.listener.pop_pressed_keys():
                if not self._accept_keypress(key):
                    continue
                if key == "a":
                    self._start_rollout()
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        # RLT success/failure is manually gated here: 'c' produces success and
        # 'a' produces failure. The outer timeout is the fallback failure path.
        reward = 0.0
        terminated = False
        truncated = False

        event: str | None = None
        result: str | None = None
        accepted_keys = [
            key
            for key in self.listener.pop_pressed_keys()
            if self._accept_keypress(key)
        ]

        if "c" in accepted_keys:
            event = "success"
            result = "success"
            reward = 1.0
            terminated = True
            self._running = False
            self._actor_active = False
            self._awaiting_reset = True
            self._log_info("Pedal 'c' pressed; marking RLT rollout success.")
        elif "a" in accepted_keys:
            event = "failure"
            result = "failure"
            terminated = True
            self._running = False
            self._actor_active = False
            self._awaiting_reset = True
            self._log_info(
                "Pedal 'a' pressed during rollout; marking RLT rollout failure."
            )
        elif "b" in accepted_keys:
            self._actor_active = not self._actor_active
            if self._actor_active:
                event = "enter_actor"
                self._log_info(
                    "Pedal 'b' pressed; switching RLT rollout to Stage 2 MLP actor."
                )
            else:
                event = "enter_reference"
                self._log_info(
                    "Pedal 'b' pressed; switching RLT rollout back to frozen VLA."
                )
        info = self._with_rlt_info(info, event=event, result=result)
        return obs, reward, terminated, truncated, info

    def _start_rollout(self) -> None:
        self._running = True
        self._actor_active = False
        self._log_info("Pedal 'a' pressed; starting RLT episode under the frozen VLA.")

    def _accept_keypress(self, key: str) -> bool:
        now = time.monotonic()
        if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
            return False
        self._last_press_ts[key] = now
        return True

    def _with_rlt_info(
        self,
        info: dict[str, Any],
        *,
        event: str | None,
        result: str | None = None,
    ) -> dict[str, Any]:
        info["rlt_switch_flags"] = self._actor_active
        info["rlt_policy_switch_event"] = event
        info["rlt_phase"] = self._phase()
        info["rlt_result"] = result
        return info

    def _idle_response(self, event: str | None):
        info = self._with_rlt_info({}, event=event)
        return self._last_obs, 0.0, False, False, info

    def _phase(self) -> str:
        if self._awaiting_reset:
            return "done"
        if not self._running:
            return "pre"
        if self._actor_active:
            return "actor"
        return "ref"

    def _log_info(self, message: str) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
