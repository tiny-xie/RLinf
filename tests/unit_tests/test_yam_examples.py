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

"""Static contract tests for the native RLinf YAM example configs."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_CONFIG = (
    _REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "env"
    / "realworld_dual_yam_joint.yaml"
)
_COLLECT_CONFIG = (
    _REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "realworld_dual_yam_collect_data.yaml"
)
_INSTALL_SCRIPT = _REPO_ROOT / "requirements" / "install.sh"
_YAM_REQUIREMENTS = (
    _REPO_ROOT / "requirements" / "embodied" / "envs" / "yam.txt"
)
_YAM_BUILD_CONSTRAINTS = (
    _REPO_ROOT
    / "requirements"
    / "embodied"
    / "envs"
    / "yam-build-constraints.txt"
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_dual_yam_env_example_exposes_the_canonical_contract():
    config = _load(_ENV_CONFIG)

    assert config["env_type"] == "realworld"
    assert config["total_num_envs"] == 1
    assert config["main_image_key"] == "top_rgb"
    assert config["init_params"]["id"] == "DualYamJointEnv-v1"
    override = config["override_cfg"]
    assert len(override["joint_limit_min"]) == 2
    assert all(len(limits) == 6 for limits in override["joint_limit_min"])
    assert len(override["joint_limit_max"]) == 2
    assert all(len(limits) == 6 for limits in override["joint_limit_max"])
    assert override["leader_intervention"]["enabled"] is False
    assert override["leader_intervention"]["unsynced_action_source"] == "policy"


def test_dual_yam_collection_example_declares_one_complete_station():
    config = _load(_COLLECT_CONFIG)
    hardware = config["cluster"]["node_groups"][0]["hardware"]

    assert hardware["type"] == "DualYam"
    assert len(hardware["configs"]) == 1
    station = hardware["configs"][0]
    devices = [
        station["left_follower"],
        station["right_follower"],
        station["left_leader"],
        station["right_leader"],
    ]
    assert len({device["channel"] for device in devices}) == 4
    assert all(len(device["gripper_limits"]) == 2 for device in devices)
    assert [camera["name"] for camera in station["cameras"]] == [
        "top_rgb",
        "left_rgb",
        "right_rgb",
    ]

    eval_config = config["env"]["eval"]
    intervention = eval_config["override_cfg"]["leader_intervention"]
    collection = eval_config["data_collection"]
    assert eval_config["max_episode_steps"] == 10000
    assert eval_config["override_cfg"]["manual_episode_control_only"] is True
    assert intervention["enabled"] is True
    assert intervention["unsynced_action_source"] == "hold"
    assert collection["export_format"] == "lerobot"
    assert collection["robot_type"] == "dual_yam"
    assert collection["fps"] == 30
    assert collection["finalize_interval"] == 0


def test_dual_yam_examples_have_no_external_application_repo_dependency():
    example_text = _ENV_CONFIG.read_text(encoding="utf-8") + _COLLECT_CONFIG.read_text(
        encoding="utf-8"
    )

    assert "yam-abc-reproduce" not in example_text
    assert "yam_abc_reproduce" not in example_text


def test_yam_install_target_bundles_the_pinned_i2rt_sdk():
    install_text = _INSTALL_SCRIPT.read_text(encoding="utf-8")
    requirements_text = _YAM_REQUIREMENTS.read_text(encoding="utf-8")
    build_constraints_text = _YAM_BUILD_CONSTRAINTS.read_text(encoding="utf-8")

    assert 'yam)' in install_text
    assert 'install_yam_env' in install_text
    assert 'embodied/envs/yam.txt' in install_text
    assert 'embodied/envs/yam-build-constraints.txt' in install_text
    assert "yam-abc-reproduce" not in install_text.lower()
    assert "i2rt @ git+https://github.com/i2rt-robotics/i2rt.git@" in requirements_text
    assert "47fee5e7dec4e30ca054f798bda1c8894b465ed2" in requirements_text
    assert "yam-abc-reproduce" not in requirements_text.lower()
    assert "scikit-build-core<0.10" in build_constraints_text
