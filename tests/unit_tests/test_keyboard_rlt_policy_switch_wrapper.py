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

import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import ModuleType


class _Wrapper:
    def __init__(self, env):
        self.env = env


gym_module = ModuleType("gymnasium")
gym_module.Env = object
gym_module.Wrapper = _Wrapper
gym_core_module = ModuleType("gymnasium.core")
gym_core_module.ActType = object
gym_core_module.ObsType = object
keyboard_listener_module = ModuleType(
    "rlinf.envs.realworld.common.keyboard.keyboard_listener"
)
keyboard_listener_module.KeyboardListener = object
stub_modules = {
    "gymnasium": gym_module,
    "gymnasium.core": gym_core_module,
    "rlinf.envs.realworld.common.keyboard.keyboard_listener": (
        keyboard_listener_module
    ),
}
missing_module = object()
original_modules = {
    name: sys.modules.get(name, missing_module) for name in stub_modules
}
sys.modules.update(stub_modules)

wrapper_path = (
    Path(__file__).parents[2]
    / "rlinf/envs/realworld/common/wrappers/keyboard_rlt_policy_switch_wrapper.py"
)
spec = importlib.util.spec_from_file_location("policy_switch_module", wrapper_path)
assert spec is not None and spec.loader is not None
policy_switch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_switch_module)
for module_name, original_module in original_modules.items():
    if original_module is missing_module:
        del sys.modules[module_name]
    else:
        sys.modules[module_name] = original_module


class _StubEnv:
    def __init__(self):
        self.step_count = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None, options=None):
        return {"step": self.step_count}, {"source": "stub"}

    def step(self, action):
        self.step_count += 1
        return {"step": self.step_count}, 0.25, False, False, {}


class _FakeListener:
    def __init__(self):
        self._batches = deque([[], ["a"]])

    def push(self, *keys: str) -> None:
        self._batches.append(list(keys))

    def pop_pressed_keys(self) -> list[str]:
        if not self._batches:
            return []
        return self._batches.popleft()


class _Clock:
    value = 1.0

    def __call__(self) -> float:
        return self.value


def _make_started_wrapper(monkeypatch):
    listener = _FakeListener()
    clock = _Clock()
    monkeypatch.setattr(policy_switch_module, "KeyboardListener", lambda: listener)
    monkeypatch.setattr(policy_switch_module.time, "monotonic", clock)
    monkeypatch.setattr(policy_switch_module.time, "sleep", lambda _: None)

    env = _StubEnv()
    wrapper = policy_switch_module.KeyboardRLTPolicySwitchWrapper(env)
    obs, info = wrapper.reset()
    return wrapper, env, listener, clock, obs, info


def test_reset_waits_for_a_and_starts_under_vla(monkeypatch):
    wrapper, env, _, _, obs, info = _make_started_wrapper(monkeypatch)

    assert obs == {"step": 0}
    assert env.step_count == 0
    assert wrapper.rlt_switch_flags is False
    assert info["rlt_switch_flags"] is False
    assert info["rlt_policy_switch_event"] == "start"
    assert info["rlt_phase"] == "ref"


def test_b_toggles_between_actor_and_vla(monkeypatch):
    wrapper, env, listener, clock, _, _ = _make_started_wrapper(monkeypatch)

    clock.value = 2.0
    listener.push("b")
    _, _, terminated, truncated, info = wrapper.step(None)
    assert not terminated
    assert not truncated
    assert wrapper.rlt_switch_flags is True
    assert info["rlt_switch_flags"] is True
    assert info["rlt_policy_switch_event"] == "enter_actor"
    assert info["rlt_phase"] == "actor"

    clock.value = 3.0
    listener.push("b")
    _, _, terminated, truncated, info = wrapper.step(None)
    assert not terminated
    assert not truncated
    assert wrapper.rlt_switch_flags is False
    assert info["rlt_switch_flags"] is False
    assert info["rlt_policy_switch_event"] == "enter_reference"
    assert info["rlt_phase"] == "ref"
    assert env.step_count == 2


def test_b_toggle_is_debounced(monkeypatch):
    wrapper, _, listener, clock, _, _ = _make_started_wrapper(monkeypatch)

    clock.value = 2.0
    listener.push("b")
    wrapper.step(None)
    assert wrapper.rlt_switch_flags is True

    clock.value = 2.1
    listener.push("b")
    _, _, _, _, info = wrapper.step(None)
    assert wrapper.rlt_switch_flags is True
    assert info["rlt_policy_switch_event"] is None

    clock.value = 2.3
    listener.push("b")
    wrapper.step(None)
    assert wrapper.rlt_switch_flags is False


def test_c_marks_success_and_disables_actor(monkeypatch):
    wrapper, env, listener, clock, _, _ = _make_started_wrapper(monkeypatch)

    clock.value = 2.0
    listener.push("b")
    wrapper.step(None)

    clock.value = 3.0
    listener.push("c")
    _, reward, terminated, truncated, info = wrapper.step(None)
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert wrapper.rlt_switch_flags is False
    assert info["rlt_policy_switch_event"] == "success"
    assert info["rlt_phase"] == "done"
    assert info["rlt_result"] == "success"

    listener.push("b")
    _, reward, terminated, truncated, info = wrapper.step(None)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["rlt_policy_switch_event"] == "awaiting_reset"
    assert env.step_count == 2


def test_second_a_marks_failure(monkeypatch):
    wrapper, env, listener, clock, _, _ = _make_started_wrapper(monkeypatch)

    clock.value = 2.0
    listener.push("a")
    _, reward, terminated, truncated, info = wrapper.step(None)
    assert reward == 0.0
    assert terminated
    assert not truncated
    assert wrapper.rlt_switch_flags is False
    assert info["rlt_policy_switch_event"] == "failure"
    assert info["rlt_phase"] == "done"
    assert info["rlt_result"] == "failure"
    assert env.step_count == 1


def test_a_failure_disables_actor(monkeypatch):
    wrapper, env, listener, clock, _, _ = _make_started_wrapper(monkeypatch)

    clock.value = 2.0
    listener.push("b")
    wrapper.step(None)

    clock.value = 3.0
    listener.push("a")
    _, reward, terminated, truncated, info = wrapper.step(None)
    assert reward == 0.0
    assert terminated
    assert not truncated
    assert wrapper.rlt_switch_flags is False
    assert info["rlt_policy_switch_event"] == "failure"
    assert info["rlt_phase"] == "done"
    assert info["rlt_result"] == "failure"

    listener.push("b")
    _, reward, terminated, truncated, info = wrapper.step(None)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["rlt_policy_switch_event"] == "awaiting_reset"
    assert env.step_count == 2
