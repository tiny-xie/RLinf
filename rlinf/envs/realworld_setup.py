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
import os
import pathlib
import time

import psutil
from filelock import FileLock

logger = logging.getLogger(__name__)

_ROS_PROCESS_NAMES = frozenset({"roscore", "rosmaster", "rosout"})


def terminate_existing_ros_processes() -> None:
    """Terminate stale ROS processes that the current worker may signal.

    Ray workers can see host processes that they do not own, especially when
    they run inside a container. Such processes must not make importing a
    RealWorld actor fail.
    """
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] not in _ROS_PROCESS_NAMES:
                continue
            proc.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # The process exited between process_iter() and kill().
            continue
        except psutil.AccessDenied:
            logger.warning(
                "Cannot terminate ROS process %s (pid=%s): permission denied. "
                "Continuing setup; stop it as its owner if it conflicts with this run.",
                proc.info.get("name", "unknown"),
                getattr(proc, "pid", "unknown"),
            )
            continue
        time.sleep(0.5)


def setup_realworld() -> None:
    """Perform concurrency-safe, node-level RealWorld environment setup."""
    node_lock_file = "/tmp/.realworld.lock"
    if not os.path.exists(os.path.dirname(node_lock_file)):
        node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")

    with FileLock(node_lock_file):
        terminate_existing_ros_processes()
