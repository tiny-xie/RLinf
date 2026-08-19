# YAM 双臂接入 RLinf 方案

本文记录 2026-08-19 对 YAM NUC 上现有采集代码和 RLinf `main` 的静态审查结果，
目标是给出一条可验证、可回退的接入路线。本文不表示真机程序已经通过验收；任何真机测试
都应在清空工作区、急停可用且有人值守的条件下单独进行。

## 1. 结论

不要直接把当前 `collect_yam_data.py` 塞进 RLinf 的 rollout worker。推荐分三步接入：

1. **保留现有独立采集器**，先把 YAM 原始 episode 稳定转换为标准 LeRobot 数据集，
   用 RLinf 跑通数据检查、归一化统计和 SFT。
2. **实现 RLinf 原生 `DualYam` 环境**，让 NUC 上唯一的 env worker 独占 CAN 和相机，
   GPU 节点只运行 actor/rollout。先支持 14D 双臂关节绝对动作。
3. **最后接在线 DAgger/PICO**。只有在纯策略评测、急停、超时保持、单手/双手接管语义
   都验收后，才让在线训练连接真机。

当前最短闭环是：

```text
YAM leader/follower + 3 cameras
  -> collect_yam_data.py
  -> yam-abc EpisodeRecorder（schema v1）
  -> LeRobot（14D state/action + 3 路 RGB + task）
  -> RLinf OpenPI dataconfig
  -> norm stats
  -> SFT
  -> 离线 action sanity check
  -> DualYamEnv 纯策略评测
  -> DAgger（可选）
```

## 2. 当前采集 pipeline

当前入口为：

```bash
python examples/embodiment/collect_yam_data.py \
  --station examples/embodiment/config/yam_station.yaml \
  --task pick_block \
  --episodes 50 \
  --convert
```

它是 `yam-abc-reproduce` 的薄编排层，刻意不 import `rlinf`：

1. 读取原生 `StationConfig`，不是 Hydra 配置。
2. 先启动并预热 3 个 RealSense worker，再打开 4 条 CAN 链。这个顺序必须保留，
   否则相机初始化阻塞可能触发电机通信 watchdog。
3. `ControlLoop` 以 30 Hz 读取两只 motorized leader，将关节目标镜像到左右 follower。
4. 顶部按钮切换同步；第二个按钮开始/停止一个 episode。
5. `EpisodeRecorder` 先写 canonical episode；`--convert` 在采集结束后批量转换为 LeRobot。

当前 canonical episode 已确认包含：

- 左右臂各 6 个关节状态和 1 个归一化夹爪状态；
- 左右臂各 6 个关节命令和 1 个归一化夹爪命令；
- `top`、`left`、`right` 三路 640×480 RGB 视频及各自时间戳；
- task、30 Hz、帧数、schema version 等 metadata。

LeRobot converter 当前输出契约为：

| 字段 | shape | 顺序/含义 |
|---|---:|---|
| `observation.state` | `(14,)` | `[L_q0..q5, L_grip, R_q0..q5, R_grip]` |
| `action` | `(14,)` | `[L_target_q0..q5, L_grip, R_target_q0..q5, R_grip]` |
| `observation.images.top_rgb` | video | 主视角 |
| `observation.images.left_rgb` | video | 左视角 |
| `observation.images.right_rgb` | video | 右视角 |
| `task` | string | 语言任务，如 `pick_block` |

### 当前未闭环项

- NUC 的 `RLinf_yam` 和嵌套 `yam-abc-reproduce` 都有大量未提交改动。接入前应先在各自
  仓库形成可追踪提交，不能依赖一个 dirty checkout。
- 静态检查时，`data/` 下没有找到转换完成的 LeRobot `meta/info.json`。因此现阶段只能确认
  canonical episode 已写出，不能确认 `--convert` 的最终产物可被 RLinf 加载。
- YAM exporter 与 RLinf 使用的 LeRobot 版本必须锁定并做兼容测试，尤其要避免 writer 写出
  新布局而 RLinf reader 只支持旧布局。
- 当前 schema 没有逐帧 CAN 延迟、相机同步误差、实际下发后的关节目标或 reject reason。
  做在线训练前必须补齐关键 telemetry。

## 3. 如何调“主臂重量/手感”

主臂手感由三个不同概念决定，不能只改一个 `weight`：

### 3.1 `bilateral_kp`：力反馈强度，不是重力补偿

当前 `yam_station.yaml` 为：

```yaml
robot:
  bilateral_kp: 0
```

- `0`：主臂 PD 增益清零，只做重力补偿，手感最自由；当前就是这个模式。
- `> 0`：使用 `YAM` 原生关节 `kp` 的比例值，把主臂拉向从臂实测姿态，产生双边力反馈。
  值越大，遇到从臂跟踪误差时阻力越明显。
- 它不会把下坠的主臂“托轻”。如果目标只是让主臂更轻，不应先增大它。

建议保持 `0` 完成第一版采集。如果确实需要力反馈，只在空场低速下从很小的值逐级试验，
每次只改一个量，并记录每个关节的峰值扭矩、跟踪误差和急停行为。

### 3.2 `ee_mass` 和 `gravity_comp_factor`：真正影响托举感

`ee_mass` 会替换末端组件质量，`gravity_comp_factor` 会逐关节缩放逆动力学重力矩。
当前 YAM 默认因子来自 i2rt 的 `robots/config/yam.yml`：

```yaml
gravity_comp_factor: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0]
```

判断方式：

- 松手后持续下坠：补偿偏小，优先检查 teaching handle 的实际质量/模型，再小幅提高主要承重
  关节的补偿因子。
- 松手后自己上抬：补偿偏大，应降低对应因子或修正末端质量。
- 某一姿态轻、另一姿态重：通常是质量、质心或模型不准，不能用全局比例掩盖。

当前 `RobotConfig.ee_mass` 同时传给 follower 和 leader，但两者末端分别是夹爪与 teaching
handle，质量通常不同。正式接入前应把它拆成 per-device 参数，例如：

```yaml
robots:
  - type: yam_left
    ee_mass: <measured-follower-mass>
controllers:
  - type: yam_lead_left
    ee_mass: <measured-leader-handle-mass>
```

在完成拆分前，不建议用全局 `ee_mass` 调主臂手感，因为它也会改变从臂补偿。

### 3.3 阻尼与摩擦

- `grav_comp_kd` 增大会让主臂更稳、更黏，但不会补足重力；当前 YAM 默认全为 `0`。
- `coulomb_friction` 只有在 `use_coulomb_friction=True` 时生效；当前适配器没有启用它。
- 如果主臂“能悬住但拖动发涩”，应检查阻尼、摩擦、线缆拖拽和关节机械摩擦，不要继续加
  重力补偿。

### 3.4 安全调参顺序

1. 保持 `bilateral_kp=0`，测量 leader teaching handle 的质量和质心。
2. 只连接一只 leader，在多个静态姿态观察下坠/上抬趋势及电机扭矩。
3. 先修正 leader 专属 `ee_mass`，再按关节小幅修正 `gravity_comp_factor`。
4. 确认能自然悬停后，再决定是否需要 `grav_comp_kd`。
5. 最后才增加 `bilateral_kp` 做力反馈，并验证 follower 阻塞、通信超时和急停。

不要在双臂、从臂和相机全部在线时首次试验新的重力或力反馈参数。

## 4. RLinf 原生接入架构

### 4.1 进程和硬件所有权

YAM NUC 上只允许一个 env worker 打开 CAN 与相机：

```text
GPU node                         YAM NUC
actor / rollout  <-- Ray -->     EnvWorker
                                 └─ RealWorldEnv
                                    └─ DualYamJointEnv
                                       ├─ yam-abc follower runtime
                                       ├─ optional leaders
                                       └─ camera workers
```

- 训练节点不直接 import 或打开 i2rt/CAN。
- env worker 启动顺序仍是 camera warm-up → CAN → control loop。
- 纯策略评测只打开两只 follower；采集/接管模式才打开两只 leader。
- close/offload 必须释放控制循环、CAN、相机线程和 socket，并可重复调用。

### 4.2 第一版动作契约：14D joint absolute

第一版应与已有数据完全一致，采用 14D 绝对关节动作，不要同时切换到 TCP：

```text
[L_q(6), L_gripper(1), R_q(6), R_gripper(1)]
```

约束：

- arm joint 单位为 rad；gripper 固定为 `0=closed, 1=open`。
- action 是通过限位与 slew-rate 检查后实际接受的目标。
- observation 使用实测关节位置，不使用上一条 command 冒充状态。
- 每一步记录 policy action、accepted action、actual state、控制周期和错误状态。
- `action_space` 必须包含真实关节上下限；gripper 为 `[0, 1]`。

等 14D 的采集、SFT、评测全部闭环后，再另建版本化的 20D TCP rot6d 数据契约。不要把 14D
数据静默转换成 20D，也不要在一个 dataset repo 内混用两种 representation。

### 4.3 建议新增文件

硬件调度：

```text
rlinf/scheduler/hardware/robots/dual_yam.py
rlinf/scheduler/hardware/robots/__init__.py
rlinf/scheduler/hardware/__init__.py
rlinf/scheduler/__init__.py
```

`DualYamHWConfig` 至少包含：左右 follower/leader CAN channel、arm/gripper type、各设备
`ee_mass`、gripper limits、相机 serial、`node_rank`。配置中不要写死 NUC IP或训练机路径。

环境：

```text
rlinf/envs/realworld/yam/dual_yam_env.py
rlinf/envs/realworld/yam/yam_robot_state.py
rlinf/envs/realworld/yam/tasks/__init__.py
rlinf/envs/realworld/yam/tasks/dual_yam_joint_env.py
rlinf/envs/realworld/yam/__init__.py
```

以 `DOSW1Env` 的 leader/follower 状态机和 `DualFrankaEnv` 的双臂观测契约为参考，注册
`DualYamJointEnv-v1`，再由现有 `RealWorldEnv` 完成 vector-env 包装和统一 observation：

```text
raw state dict + camera dict
  -> RealWorldEnv._wrap_obs
  -> states / main_images / extra_view_images / task_descriptions
```

配置与入口：

```text
examples/embodiment/config/env/realworld_dual_yam_joint.yaml
examples/embodiment/config/realworld_dual_yam_collect_data.yaml
examples/embodiment/config/realworld_dual_yam_eval_openpi.yaml
```

依赖安装应增加 `yam` env target，但必须固定 `yam-abc-reproduce` commit；不能 clone 浮动的
`main`，也不能依赖 NUC 上未提交的嵌套仓库。

## 5. 数据与 OpenPI 接入

### 5.1 先验收 exporter

在接 RLinf loader 前，至少用 3 个短 episode 验证：

1. dataset 根目录存在 `meta/info.json`，LeRobot 能列出 episode 和 task；
2. state/action shape 恒为 14，无 NaN/Inf；
3. 三路视频帧数、时间戳和控制 step 对齐，报告最大与 p95 偏差；
4. action 的左右臂顺序、关节符号和夹爪开合方向与真机一致；
5. episode 末尾没有重复旧相机帧或缺失 action；
6. 用 action 回放到 mock/sim 时轨迹连续且不越限。

### 5.2 新建 YAM dataconfig

不要直接使用当前 `pi0_realworld`：它的 transform 明确假设 19D state 和 7D action，和 YAM
14D/14D 不兼容。应新增 YAM 专用配置，例如：

```text
rlinf/models/embodiment/openpi/dataconfig/yam_dataconfig.py
rlinf/models/embodiment/openpi/policies/yam_policy.py
```

映射建议固定为：

```text
top_rgb   -> base_0_rgb
left_rgb  -> left_wrist_0_rgb
right_rgb -> right_wrist_0_rgb
state     -> 14D
actions   -> [horizon, 14]
```

随后在 OpenPI config registry 注册 `pi0_yam`/`pi05_yam`，增加一份 SFT YAML，并确保：

- `actor.model.action_dim: 14`；
- `num_action_chunks` 与 OpenPI `action_horizon` 完全一致；
- norm stats 从当前 YAM dataset 重新计算，不能复用 Franka/Aloha stats；
- task string 与部署时 prompt 完全一致；
- dataset metadata 写入 action representation 和 schema version。

## 6. 从 SFT 到在线 DAgger

按以下门禁推进，前一项未通过时不要进入下一项：

### M0：冻结依赖与数据契约

- 将 NUC 上 RLinf 改动和 `yam-abc-reproduce` 改动分别提交；
- 固定 yam-abc、i2rt、LeRobot commit/version；
- 冻结 14D joint absolute 顺序、单位、夹爪方向和 3 个 camera key；
- 将 leader/follower 的 `ee_mass` 分开配置。

### M1：离线数据闭环

- mock 采集和转换测试；
- 真机 3 条短 episode 转换；
- RLinf loader smoke test；
- norm stats；
- 小数据 overfit，确认模型输出维度、范围和连续性。

### M2：原生 Env 的 dummy/sim 闭环

- `DualYamHWInfo` 注册和 config 解析测试；
- `DualYamJointEnv-v1` reset/step/close 测试；
- joint limits、slew limit、NaN、timeout、重复 close 测试；
- 相机缺帧和 CAN 异常必须返回结构化错误并进入 hold/estop。

### M3：真机纯策略评测

- 单臂、低速、无物体评测；
- 两臂分区评测；
- policy server/rollout 超时后保持当前目标，绝不发送零位；
- 验证 action chunk 只按受控频率消费，禁止一次性把整个 chunk 瞬间下发；
- 通过后才进行任务评测。

### M4：leader intervention / DAgger

- 复用或泛化 `LeaderFollowerKeyboardIntervention`；
- 明确 MODEL/PAUSE/TELEOP/ESTOP 状态机；
- 接管时先将 follower 平滑对齐 leader，禁止跳变；
- `info["intervene_action"]` 写入实际接受的完整 14D action；
- 额外保留 left/right intervention mask，单手接管不能错误覆盖另一只手；
- 先离线检查 replay buffer，再允许 actor 更新。

## 7. 最低测试矩阵

| 层级 | 测试 | 通过标准 |
|---|---|---|
| config | YAM hardware YAML 解析 | channel、serial、质量、limits 无静默默认错误 |
| robot adapter | state/action round-trip | 14D 顺序、单位、夹爪极性完全一致 |
| safety | NaN、越限、大步长、超时 | reject/hold，不下发危险 command |
| lifecycle | init/reset/close×2 | 无残留 CAN、相机或控制线程 |
| dataset | 3 episode convert/load | shape、task、视频、时间戳完整 |
| SFT | 小数据 overfit | loss 下降，输出范围合理，无维度 padding 错位 |
| dummy env | RLinf collector/eval | 无硬件也能跑通接口和 episode lifecycle |
| hardware | 单臂低速 | 无跳变、无 watchdog、急停有效 |
| dual arm | 分区动作 | 左右 action slice 不串扰、不碰撞 |
| DAgger | 单手/双手接管 | mask、expert action、policy action 语义正确 |

## 8. 第一批实现提交建议

建议按可独立审查的提交拆分：

1. `docs: document YAM integration contract`
2. `build: add pinned YAM environment dependencies`
3. `feat(data): add YAM LeRobot dataconfig`
4. `test(data): validate YAM 14D dataset contract`
5. `feat(realworld): register DualYam hardware`
6. `feat(realworld): add dummy DualYam joint environment`
7. `feat(realworld): connect DualYam hardware runtime`
8. `feat(embodiment): add YAM SFT and eval configs`
9. `feat(realworld): add YAM leader intervention`

第一批代码只应做到 M0 + M1，不要把依赖安装、dataset schema、真机 env 和在线 DAgger
塞进同一个提交。这样出现问题时可以明确区分数据、模型、调度、CAN 或遥操状态机。
