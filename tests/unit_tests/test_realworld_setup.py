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

import logging

import psutil

from rlinf.envs import realworld_setup


class _FakeProcess:
    def __init__(self, name: str, pid: int, error: Exception | None = None):
        self.info = {"name": name}
        self.pid = pid
        self.error = error
        self.killed = False

    def kill(self) -> None:
        if self.error is not None:
            raise self.error
        self.killed = True


def test_ros_cleanup_ignores_processes_that_cannot_be_signalled(monkeypatch, caplog):
    killed = _FakeProcess("roscore", 101)
    denied = _FakeProcess("rosmaster", 102, psutil.AccessDenied(pid=102))
    vanished = _FakeProcess("rosout", 103, psutil.NoSuchProcess(pid=103))
    unrelated = _FakeProcess("python", 104)
    processes = [killed, denied, vanished, unrelated]
    sleep_calls = []

    monkeypatch.setattr(
        realworld_setup.psutil,
        "process_iter",
        lambda attrs: processes,
    )
    monkeypatch.setattr(realworld_setup.time, "sleep", sleep_calls.append)

    with caplog.at_level(logging.WARNING):
        realworld_setup.terminate_existing_ros_processes()

    assert killed.killed
    assert not denied.killed
    assert not vanished.killed
    assert not unrelated.killed
    assert sleep_calls == [0.5]
    assert "pid=102" in caplog.text
    assert "permission denied" in caplog.text
