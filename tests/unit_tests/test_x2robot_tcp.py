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

import base64
import json
from collections import deque

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from rlinf.envs import SupportedEnvType, get_env_cls  # noqa: E402
from rlinf.envs.realworld.x2robot.lerobot_recorder import (  # noqa: E402
    IMAGE_NAMES,
    X2RobotLeRobotRecorder,
)
from rlinf.envs.realworld.x2robot.realworld_env import (  # noqa: E402
    X2RobotTCPRealWorldEnv,
)
from rlinf.envs.realworld.x2robot.tcp_env import (
    RUNNING_MODE_RLT,
    RUNNING_MODE_VLA,
    X2RobotTCPConfig,
    X2RobotTCPEnv,
    _blend_chunk_transition,
    _decode_embedded_frames,
    _parse_running_mode,
    _running_mode_info,
)
from rlinf.envs.realworld.x2robot.upload_server import UploadServer  # noqa: E402


def test_x2robot_env_type_is_registered():
    assert SupportedEnvType.X2ROBOT_TCP.value == "x2robot_tcp"
    assert get_env_cls("x2robot_tcp") is X2RobotTCPRealWorldEnv


def test_x2robot_config_derives_sequence_geometry():
    config = X2RobotTCPConfig(
        state_history_size=3,
        state_future_size=2,
        latency_step=None,
    )

    assert config.state_seq_len == 6
    assert config.latency_step == 2
    assert config.latency_len == 6

    with pytest.raises(ValueError, match="unsupported policy_mode"):
        X2RobotTCPConfig(policy_mode="invalid")

    with pytest.raises(ValueError, match="lerobot_data_path is required"):
        X2RobotTCPConfig(lerobot_record_enabled=True)


def test_x2robot_running_mode_selects_vla_or_rlt():
    assert _parse_running_mode({}) == RUNNING_MODE_VLA
    assert _parse_running_mode({"running_mode": RUNNING_MODE_RLT}) == RUNNING_MODE_RLT
    assert _running_mode_info(RUNNING_MODE_VLA)["rlt_switch_flags"] is False
    assert _running_mode_info(RUNNING_MODE_RLT)["rlt_switch_flags"] is True

    with pytest.raises(ConnectionError, match="only accepts policy modes"):
        _parse_running_mode({"running_mode": 2})


def test_x2robot_decodes_robo_avatar_embedded_images():
    images = {}
    for index, name in enumerate(("left", "front", "right")):
        frame = np.full((4, 5, 3), index * 20, dtype=np.uint8)
        ok, payload = cv2.imencode(".jpg", frame)
        assert ok
        images[name] = base64.b64encode(payload).decode("ascii")

    frames = _decode_embedded_frames({"images": images})

    assert frames is not None
    assert set(frames) == {"left_wrist_view", "face_view", "right_wrist_view"}
    assert all(frame.shape == (4, 5, 3) for frame in frames.values())

    request = {
        "follow1_pos": np.zeros((4, 7)).tolist(),
        "follow2_pos": np.ones((4, 7)).tolist(),
        "running_mode": RUNNING_MODE_RLT,
        "inference_session_id": 7,
        "images": images,
    }
    payload = json.dumps(request).encode("utf-8")

    class FakeConnection:
        def __init__(self, wire_data):
            self.data = bytearray(wire_data)

        def recv(self, size):
            result = bytes(self.data[:size])
            del self.data[:size]
            return result

    connection = FakeConnection(len(payload).to_bytes(4, "little") + payload)
    env = object.__new__(X2RobotTCPEnv)
    env.config = X2RobotTCPConfig(image_height=4, image_width=5)

    _, decoded_frames, running_mode, session_id = env._read_frame(connection)

    assert running_mode == RUNNING_MODE_RLT
    assert session_id == 7
    assert set(decoded_frames) == set(frames)
    assert not connection.data


def test_x2robot_assembles_state_and_serializes_action_chunk():
    env = object.__new__(X2RobotTCPEnv)
    env.config = X2RobotTCPConfig()
    master_queue = deque(maxlen=100)
    left = np.arange(28, dtype=np.float32).reshape(4, 7)
    right = left + 100

    state = env._assemble_state((left, right), master_queue)

    assert state.shape == (6, 32)
    np.testing.assert_array_equal(state[3, :7], left[-1])
    np.testing.assert_array_equal(state[3, 7:14], right[-1])
    expected_master = np.repeat(state[-1:, :14], state.shape[0], axis=0)
    np.testing.assert_array_equal(state[:, 14:28], expected_master)

    chunk = np.arange(20 * 28, dtype=np.float32).reshape(20, 28)
    payload = env._postprocess(chunk, master_queue)

    command = json.loads(payload)
    assert len(command["follow1_pos"]) == env.config.move_steps + 1
    assert len(command["follow2_pos"]) == env.config.move_steps + 1
    np.testing.assert_array_equal(
        command["follow1_pos"][1], chunk[env.config.latency_step, 14:21]
    )


def test_x2robot_blending_preserves_gripper_dimensions():
    action_chunk = np.array([[0.0, 5.0], [10.0, 6.0], [20.0, 7.0]], dtype=np.float64)
    history = deque([np.array([-1.0, 4.0]), np.array([0.0, 5.0])])

    blended = _blend_chunk_transition(
        action_chunk, history, blend_steps=2, skip_dims=(1,)
    )

    assert blended[1, 0] != action_chunk[1, 0]
    np.testing.assert_array_equal(blended[:, 1], action_chunk[:, 1])


def test_upload_record_keeps_executed_action_without_decoding_images():
    server = UploadServer(decode_images=False)
    header = {
        "seq": 7,
        "t": 1.25,
        "mode": 2,
        "is_takeover": True,
        "follow1_pos": list(range(7)),
        "follow2_pos": list(range(7, 14)),
        "master1_pos": list(range(14, 21)),
        "master2_pos": list(range(21, 28)),
    }

    record = server._build_record(header, [b"", b"", b""])

    assert record["is_takeover"] is True
    assert record["action_28"].shape == (28,)
    assert "frames" not in record

    # Mode is metadata only; the uploader's per-frame flag owns the label.
    header["mode"] = 2
    header["is_takeover"] = False
    record = server._build_record(header, [b"", b"", b""])
    assert record["is_takeover"] is False

    header["mode"] = 3
    header["is_takeover"] = True
    record = server._build_record(header, [b"", b"", b""])
    assert record["is_takeover"] is True

    # Accept the newer field spelling for metadata without changing labeling.
    header["running_mode"] = 2
    record = server._build_record(header, [b"", b"", b""])
    assert record["mode"] == 2
    assert record["is_takeover"] is True


def test_x2robot_lerobot_recorder_writes_per_frame_takeover(tmp_path):
    data_path = tmp_path / "x2robot_lerobot"
    recorder = X2RobotLeRobotRecorder(
        str(data_path),
        task_description="Clean the table.",
        image_height=8,
        image_width=8,
        fps=20,
    )
    recorder.start()
    header = {
        "seq": 7,
        "t": 1.25,
        "mode": 2,
        "is_takeover": True,
        "follow1_pos": list(range(7)),
        "follow2_pos": list(range(7, 14)),
        "master1_pos": list(range(14, 21)),
        "master2_pos": list(range(21, 28)),
    }
    blobs = []
    for value in (20, 40, 60):
        ok, blob = cv2.imencode(".jpg", np.full((8, 8, 3), value, dtype=np.uint8))
        assert ok
        blobs.append(blob.tobytes())

    header["is_takeover"] = False
    recorder.append_step(header, blobs)
    header["seq"] = 8
    header["t"] = 1.30
    header["is_takeover"] = True
    recorder.append_step(header, blobs)
    recorder.finish_episode({"type": "episode_end", "success": True, "n": 2})
    recorder.close()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=str(data_path))
    assert len(dataset) == 2
    assert [bool(value) for value in dataset.hf_dataset["intervene_flag"]] == [
        False,
        True,
    ]
    assert [int(value) for value in dataset.hf_dataset["mode"]] == [2, 2]
    assert dataset.hf_dataset["task_index"] == [0, 0]
    assert dataset.hf_dataset["state"][0][-4:].tolist() == [0.0] * 4
    assert len(dataset.hf_dataset["state"][0]) == 32
    assert len(dataset.hf_dataset["actions"][0]) == 28
    assert all(name in dataset.features for name in IMAGE_NAMES)


def test_x2robot_takeover_release_waits_for_fresh_policy_observation():
    env = object.__new__(X2RobotTCPRealWorldEnv)
    env.num_envs = 1
    env.expand_chunk_obs = True
    env.auto_reset = True
    env.ignore_terminations = True
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env._last_obs = {"marker": "before_takeover"}
    env._pending_success = None

    class FakeVectorEnv:
        @staticmethod
        def call(name, *args):
            del args
            if name == "chunk_round_trip":
                return [(None, 0.0, False, True, {"disconnected": True})]
            if name == "take_episode_end":
                return [None]
            raise AssertionError(f"unexpected env call: {name}")

    env.env = FakeVectorEnv()
    collect_calls = 0

    def collect_intervene(chunk_size, action_dim, policy_chunk):
        nonlocal collect_calls
        del policy_chunk
        collect_calls += 1
        flags = torch.full((1, chunk_size), collect_calls == 2, dtype=torch.bool)
        actions = torch.zeros((1, chunk_size * action_dim), dtype=torch.float32)
        return actions, flags, chunk_size if collect_calls == 2 else 0

    env._collect_intervene = collect_intervene
    env._record_metrics = lambda *args: {**dict(args[-1]), "episode": {}}

    fresh_obs = {"marker": "after_takeover"}

    def handle_auto_reset(dones, final_obs, final_info):
        assert dones.tolist() == [True]
        assert final_obs == env._last_obs
        return fresh_obs, {
            "rlt_switch_flags": torch.tensor([True]),
            "final_info": final_info,
        }

    env._handle_auto_reset = handle_auto_reset
    actions = np.zeros((1, 4, 28), dtype=np.float32)

    obs_list, _, _, truncations, infos_list = env.chunk_step(actions)

    assert obs_list[-1] is fresh_obs
    assert truncations[0, -1]
    assert infos_list[-1]["rlt_switch_flags"].item() is True
    assert infos_list[-1]["final_info"]["intervene_flag"].all()
    assert collect_calls == 2
