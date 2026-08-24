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

"""Regression tests for YAM's generic real-world data collector."""

from __future__ import annotations

from types import SimpleNamespace

import psutil
import torch


def test_yam_collector_accepts_recorded_task_descriptions(monkeypatch):
    """The YAM recipe keeps string task metadata alongside tensor observations."""
    # Importing RealWorldEnv has an existing process-cleanup side effect. Keep
    # this unit test isolated from host ROS processes.
    monkeypatch.setattr(psutil, "process_iter", lambda: ())

    from examples.embodiment.collect_real_data import DataCollector

    collector = object.__new__(DataCollector)
    collector.cfg = SimpleNamespace(
        runner=SimpleNamespace(record_task_description=True)
    )
    states = torch.arange(14, dtype=torch.float32).reshape(1, 14)
    descriptions = ["pick_block"]

    processed = collector._process_obs(
        {"states": states, "task_descriptions": descriptions}
    )

    assert processed["task_descriptions"] == descriptions
    assert torch.equal(processed["states"], states)
