采集双臂 YAM 示教数据
======================

使用 RLinf 原生的 YAM 环境，通过两台电机主臂遥操作两台从臂，并将
14 维关节空间示教直接写成 LeRobot 数据。运行时只依赖 RLinf 和一个固定版本、
经过兼容性验证的 ``i2rt`` 安装包，不需要再下载或导入另一份 YAM 应用仓库。

.. warning::

   本页的数据采集命令会在第一次环境 ``reset()`` 时打开 3 台相机和 4 个
   SocketCAN 设备，并可能驱动两台从臂。运行前请清空工作空间、确保物理急停
   随手可及、松开两个示教手柄的按钮，并让经过培训的操作员留在设备旁。
   只阅读文档或运行下文列出的单元测试不会打开硬件。

概览
----

当前示例覆盖 RLinf 原生环境接入和示教数据采集。YAM 专用的策略数据变换、
SFT 和部署配置属于后续集成阶段。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      采集阶段不需要策略模型

   .. grid-item-card:: 算法
      :text-align: center

      主从臂遥操作 · 可扩展为 DAgger 干预

   .. grid-item-card:: 任务
      :text-align: center

      双臂关节空间示教

   .. grid-item-card:: 硬件
      :text-align: center

      2 台从臂 · 2 台电机主臂 · 3 台 RealSense

| **本页流程：** 安装运行环境 → 配置一套 YAM 工作站 → 导出标定值 → 采集成功轨迹 → 检查 RLinf 与 LeRobot 两类输出。
| **前置条件：** :doc:`安装 RLinf <../../start/installation>` · Linux SocketCAN · 3 台 Intel RealSense · 经过验证的 ``i2rt`` build · 物理急停。

RLinf 如何组织机器人设备
-------------------------

RLinf 把“资源调度”和“设备控制”明确分开：

.. code-block:: text

   Hydra YAML
     -> NodeHardwareConfig(type="DualYam")
     -> DualYamConfig
     -> DualYamRobot.enumerate()             # 只处理配置
     -> 每套完整工作站对应一个 DualYamHWInfo
     -> component_placement 选择 hardware rank 0
     -> WorkerInfo.hardware_infos
     -> RealWorldEnv
     -> create_dual_yam_joint_env()
     -> 可选 DualYamLeaderIntervention
     -> DualYamJointEnv
     -> YamControlRuntime                    # 从臂命令的唯一写入者
     -> 延迟加载 i2rt backend

一份 ``DualYamConfig`` 代表一套完整工作站，其中同时包含左右从臂、左右主臂和
所有相机。调度器把它枚举成一个 hardware rank。因此 ``placement: 0`` 表示
“第一套完整 YAM 工作站”，并不是 node rank 0，也不是 ``can0`` 接口。node rank
与 hardware rank 的区别请参见 :doc:`资源放置 <../../concepts/placement>`。

调度阶段只校验配置和资源独占关系，不导入 ``i2rt``、不探测 CAN、也不打开
相机。环境 worker 被分配到资源后，才通过 ``WorkerInfo`` 收到
``DualYamHWInfo``。真正的硬件连接推迟到第一次 ``reset()``：

.. code-block:: text

   打开并预热全部相机 -> 连接左从臂 -> 原位保持
                    -> 连接右从臂 -> 原位保持 -> 校验反馈
                    -> 仅在启用主臂干预时连接两台主臂

纯策略推理时关闭主臂干预，因此不会打开两个主臂 CAN。采集配置则会启用主臂
wrapper。当前实现要求 4 个 CAN 接口和所有相机都能被同一个环境 worker
所在节点直接访问。

配置边界
~~~~~~~~

三类配置应分别放在对应的 YAML 区域：

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - 配置类别
     - 所在位置
     - 负责内容
   * - 物理工作站
     - ``cluster.node_groups[].hardware``
     - 节点归属、CAN 名称、机械臂/夹爪型号、各设备质量和补偿参数、夹爪原始端点、相机序列号。
   * - 任务与运行时
     - ``env.eval.override_cfg``
     - 任务文本、控制频率、RLinf 关节/步进限制、超时、图像尺寸和主臂回合控制行为。
   * - 进程放置
     - ``cluster.component_placement``
     - 哪个 worker 独占接收一整套工作站资源。

安装
----

通过 RLinf 既有安装入口创建 YAM embodied 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

该入口会安装 RLinf embodied 依赖、RealSense/OpenCV、LeRobot，以及
``requirements/embodied/envs/yam.txt`` 中固定 commit 的官方 ``i2rt`` SDK；不要求也不会
clone YAM 应用仓库，同时没有额外的 wheel 环境变量或独立 SDK 安装路径。模块仍然延迟
导入 ``i2rt``，因此仅调度和 dummy 环境的 import 不会触碰硬件。

兼容 build 必须支持 ``get_yam_robot()`` 使用的公开参数，并能可靠停止自身的
CAN/控制线程。如果 SDK 构造函数不支持逐设备数值型阻尼或摩擦力覆盖，适配层会
明确拒绝该配置，不会修改 SDK 私有数组。

启动 RLinf 前，请通过系统常规配置为 4 个 USB-CAN 适配器设置持久 SocketCAN
名称并拉起接口。示例默认值如下：

.. list-table::
   :header-rows: 1
   :widths: 28 26 46

   * - 角色
     - 默认接口
     - 可选环境变量
   * - 左从臂
     - ``can_left``
     - ``YAM_LEFT_FOLLOWER_CAN``
   * - 右从臂
     - ``can_right``
     - ``YAM_RIGHT_FOLLOWER_CAN``
   * - 左主臂
     - ``can_lead_l``
     - ``YAM_LEFT_LEADER_CAN``
   * - 右主臂
     - ``can_lead_r``
     - ``YAM_RIGHT_LEADER_CAN``

以下命令只读取接口状态，不会向机器人发控制命令：

.. code-block:: bash

   ip -details link show can_left
   ip -details link show can_right
   ip -details link show can_lead_l
   ip -details link show can_lead_r

配置工作站
----------

可复用的环境默认值位于
``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``，完整的单节点
采集示例位于
``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``。

采集示例将两个 follower 的 ``gripper_limits`` 设为 ``null``，使用官方 i2rt
自动标定流程。每次启动时，两只夹爪都会分别向两个方向运动，以检测当前编码圈中的
``[闭合, 张开]`` 电机范围。标定完成前必须保证两只夹爪完全无遮挡。

如果某个工作站明确选择跳过启动标定，应将 ``null`` 替换为本工作站的实测值，
顺序固定为 ``[闭合, 张开]``：

.. code-block:: bash

   # 从臂夹爪原始电机位置，顺序必须是 [闭合, 张开]。
   # 数值可以递增，也可以递减；不要排序。
   export YAM_LEFT_GRIPPER_CLOSED_RAD=<左夹爪闭合实测值>
   export YAM_LEFT_GRIPPER_OPEN_RAD=<左夹爪张开实测值>
   export YAM_RIGHT_GRIPPER_CLOSED_RAD=<右夹爪闭合实测值>
   export YAM_RIGHT_GRIPPER_OPEN_RAD=<右夹爪张开实测值>

   # RealSense 序列号。
   export YAM_TOP_CAMERA_SERIAL=<顶部相机序列号>
   export YAM_LEFT_CAMERA_SERIAL=<左侧相机序列号>
   export YAM_RIGHT_CAMERA_SERIAL=<右侧相机序列号>

不要估算固定夹爪端点。这些值是电机弧度原始端点，不是动作中的归一化 ``[0, 1]``。
顺序本身包含电机方向信息，因此合法的 ``[闭合, 张开]`` 数对也可能递减。固定的多圈
限位还必须匹配本次启动的编码圈；除非所安装的 i2rt build 会将持久化限位对齐到当前
编码圈，否则应优先使用自动标定。

如果 CAN 名称不同，再导出上表中的 4 个可选变量。最终解析出的 4 个名称必须
互不相同，相机名称和序列号也必须分别唯一。

各设备调参项与设备本身放在同一段 hardware 配置中：

.. code-block:: yaml

   left_leader:
     ee_mass: null                 # 使用固定 i2rt 模型中的值
     gravity_comp_factor: null     # 使用所选机械臂模型的默认值
     grav_comp_kd: null
     coulomb_friction: null
     use_coulomb_friction: false
     bilateral_kp: 0.0
     gripper_invert: false

``ee_mass`` 和 ``gravity_comp_factor`` 影响重力支撑；``grav_comp_kd`` 是重力补偿
阻尼；``coulomb_friction`` 与 ``use_coulomb_friction`` 控制库仑摩擦补偿。
``bilateral_kp`` 则控制主臂向从臂实测位置反馈的强度，不是重力补偿参数，初次
采集应保持 ``0.0``。可选值为 ``null`` 时保留固定 SDK 模型的配置。数值型
``grav_comp_kd`` 和 ``coulomb_friction`` 只有在 i2rt 构造函数明确支持时才能使用。

基础配置中的关节上下限是 YAM v1 名义范围。若台面、任务或安装空间更小，应替换
成经过验证的更窄工作区。切换其他 YAM 型号时，必须同时修改 ``arm_type`` 和
RLinf 关节限制；如果 RLinf 配置超出 SDK 限制，启动检查会拒绝继续。

无硬件验证
----------

正式上机前运行 YAM 单元测试。它们全部使用 mock，不会打开 CAN 或相机：

.. code-block:: bash

   pytest -q \
     tests/unit_tests/test_yam_hardware.py \
     tests/unit_tests/test_yam_runtime.py \
     tests/unit_tests/test_yam_env.py \
     tests/unit_tests/test_yam_intervention.py \
     tests/unit_tests/test_yam_imports.py \
     tests/unit_tests/test_yam_examples.py

这些测试覆盖注册与配置转换、14 维顺序、关节/步进限制、陈旧或非有限反馈处理、
干预所有权、清理以及 ``i2rt`` 延迟导入，但不能替代低速真机验收。

采集示教
--------

.. danger::

   下一条命令是真机启动点。第一次 ``reset()`` 会打开相机、连接两台从臂，随后
   连接两台主臂。不要把它当成只检查配置的命令运行。

在 RLinf 仓库根目录启动默认 50 回合采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
     realworld_dual_yam_collect_data

示例配置默认采集 50 个 ``pick_block`` 回合。为了保持 RLinf 既有的“配置名启动”
风格，如需创建其他任务配方，请复制或修改
``realworld_dual_yam_collect_data.yaml`` 中的以下字段：

.. code-block:: yaml

   runner:
     num_data_episodes: 50
   env:
     eval:
       override_cfg:
         task_description: pick_block

该流程会：

1. 调用 RLinf 通用的 ``collect_real_data.py`` 入口；
2. 让调度器分配一份完整 ``DualYam`` 资源；
3. 构建 ``RealWorldEnv`` 和已注册的 ``DualYamJointEnv-v1``；
4. 启用电机主臂干预与按钮回合控制；
5. 将成功轨迹同时写入 RLinf replay 和 LeRobot 数据。

整个过程没有 ``--convert`` 阶段，也不会在运行时 clone 或 import YAM 应用仓库。

示教手柄按钮
~~~~~~~~~~~~

连接主臂时先松开两个按钮。首个采样会建立每个手柄的空闲电平；任一手柄上的
按钮都会控制整套双臂工作站。

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - 控件
     - 当前状态
     - 结果
   * - 顶部/第一个按钮
     - 未同步
     - 从从臂实测位置平滑过渡到当前主臂位置，随后两台主臂共同控制两台从臂。
   * - 顶部/第一个按钮
     - 已同步
     - 两台从臂原位保持，两台主臂回到重力补偿空闲状态。
   * - 录制/第二个按钮
     - 回合开始前等待
     - 从当前位置开始一个新录制回合。
   * - 录制/第二个按钮
     - 正在录制
     - 以 reward ``1`` 和 success 结束回合，并保持遥操同步以继续下一个回合。
   * - 夹爪扳机
     - 默认映射
     - 松开为夹爪 ``1``（张开），按下为 ``0``（闭合）；在对应主臂设置 ``gripper_invert: true`` 可反转。

示例使用 ``sync_on_reset: false``，操作者准备好后只需按一次顶部按钮接管；
``preserve_sync_between_episodes: true`` 使录制按钮只负责切分回合，不释放遥操，
只有顶部按钮切换同步状态。同时使用 ``unsynced_action_source: hold``，所以
collector 的占位零动作不会把从臂送向零位。按钮事件采用上升沿触发并做消抖。

观测与动作契约
--------------

所有状态和动作都使用同一绝对 14 维布局：

.. code-block:: text

   [left_q0, ..., left_q5, left_gripper,
    right_q0, ..., right_q5, right_gripper]

机械臂关节单位为弧度；夹爪归一化为 ``0=闭合, 1=张开``。观测中的关节值来自
从臂实测位置。所有命令都会校验精确 shape 和有限性，并把夹爪裁剪到有效范围。
默认情况下，机械臂命令还会裁剪到配置限位，并根据实测位置应用
``max_joint_delta`` 步进限制。采集示例设置
``enforce_runtime_joint_limits: false`` 来对齐旧版 yam-abc 遥操路径：平滑接合完成后，
主臂关节目标直接交给 i2rt，由 i2rt 应用硬件限位。wrapper 会把最终目标写入
``intervene_action``。

相机与数据键会在三层边界发生有意的重命名：

.. list-table::
   :header-rows: 1
   :widths: 23 37 40

   * - 边界
     - 键名
     - 含义
   * - YAM Gym 观测
     - ``frames.top_rgb``、``frames.left_rgb``、``frames.right_rgb``
     - 具名 RGB 帧；``state.joint_position`` 是 14 维实测状态。
   * - ``RealWorldEnv``
     - ``main_images``、``extra_view_images``
     - ``top_rgb`` 成为主视角；其他名称排序后，当前配置中 index 0 是 ``left_rgb``，index 1 是 ``right_rgb``。
   * - RLinf LeRobot writer
     - ``image``、``extra_view_image-0``、``extra_view_image-1``
     - 最终数据集中顶部、左侧、右侧相机的字面 feature 名；状态和动作分别是 ``state``、``actions``。

如果下游 transform 依赖该顺序，请保留示例中的相机名称。通用 collector 会保留
视角顺序，但不会在最终 LeRobot 列中保留 ``left_rgb``/``right_rgb`` 语义名。
后续 YAM 策略 dataconfig 需要显式映射这些字面键名。

输出目录
--------

``collect_data.sh`` 会创建新的 ``logs/<timestamp>/``。同一个成功回合会写到两个
位置：

.. code-block:: text

   logs/<timestamp>/
   |-- demos/                         # RLinf TrajectoryReplayBuffer (.pt)
   `-- collected_data/
       `-- rank_0/
           `-- id_0/                 # 本次运行的 LeRobot shard
               |-- meta/info.json
               |-- meta/episodes.jsonl
               |-- meta/tasks.jsonl
               |-- meta/stats.json
               |-- data/...
               `-- videos/...        # 具体布局取决于 LeRobot 版本

示例设置 ``finalize_interval: 0``，因此所有回合留在同一个 shard，并在正常退出时
统一完成 metadata。只有复用显式 ``save_dir`` 时才应设置 ``resume: true``；恢复
运行会写新的 ``id_N`` shard，不覆盖已 finalize 的 shard。

LeRobot 帧包含 ``state``、``actions``、``image``、``extra_view_image-0``、
``extra_view_image-1``、``done``、``is_success``、``intervene_flag`` 和
``segment_id``。任务文本通过 LeRobot task metadata 保存。通用 writer 行为请参见
:doc:`数据采集 <../../guides/data_collection>`。

YAM 文件职责总览
----------------

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - 文件
     - 含义与职责
   * - ``rlinf/scheduler/hardware/robots/dual_yam.py``
     - 定义一套完整工作站资源，转换嵌套 Hydra 配置，校验 CAN/相机唯一性，并在不访问硬件的前提下完成枚举。
   * - ``rlinf/envs/realworld/yam/types.py``
     - 定义统一 14 维状态/动作契约、类型化状态、命令结果和 backend 协议。
   * - ``rlinf/envs/realworld/yam/config.py``
     - 校验任务层控制频率、关节限制、相机超时和主臂干预行为。
   * - ``rlinf/envs/realworld/yam/i2rt_backend.py``
     - 作为唯一 ``i2rt`` 边界进行延迟导入，并适配从臂、主臂、按钮、健康状态与清理 API。
   * - ``rlinf/envs/realworld/yam/mock_backend.py``
     - 为 dummy 模式和测试提供完全不访问硬件的主从臂实现。
   * - ``rlinf/envs/realworld/yam/control_runtime.py``
     - 统一拥有所有 transport、串行化从臂写入、校验每条命令、平滑接管、故障保持并按安全顺序关闭。
   * - ``rlinf/envs/realworld/yam/dual_yam_joint_env.py``
     - 实现 Gym action/observation space、延迟启动、相机处理、step 节拍和资源关闭。
   * - ``rlinf/envs/realworld/yam/leader_intervention.py``
     - 实现双主臂同步、按钮回合控制、policy/hold/leader 命令所有权和 ``intervene_action`` 上报。
   * - ``rlinf/envs/realworld/yam/tasks/__init__.py``
     - 注册 ``DualYamJointEnv-v1``、校验 ``main_image_key``，并按配置装配主臂干预 wrapper。
   * - ``rlinf/envs/realworld/yam/__init__.py``
     - 汇总公开 YAM API，并触发任务注册。
   * - ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``
     - 可复用的 Gym/任务默认值及显式 RLinf 安全限制。
   * - ``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``
     - 一套完整工作站的调度、遥操作和直接 LeRobot 采集配置。
   * - ``requirements/install.sh`` 中的 ``--env yam``
     - 构建包含相机、LeRobot 和固定 i2rt SDK 的完整 YAM 环境，不要求 YAM 应用仓库。
   * - ``requirements/embodied/envs/yam.txt``
     - 固定官方 i2rt commit，并声明原生 runtime 所需的相机和配置依赖。
   * - ``requirements/embodied/envs/yam-build-constraints.txt``
     - 将 i2rt 的 ruckig 源码构建约束限制在 YAM 环境内部。
   * - ``examples/embodiment/collect_data.sh``
     - YAM 配置复用的、保持原状的通用采集入口；它按配置名启动并创建带时间戳的日志目录。
   * - ``rlinf/envs/realworld/__init__.py``
     - 从 RLinf 真机入口导入 YAM task 包，使 Gym 注册生效。
   * - ``rlinf/scheduler/__init__.py``、``rlinf/scheduler/hardware/__init__.py``、``rlinf/scheduler/hardware/robots/__init__.py``
     - 导出 YAM 调度类型，并把 ``DualYam`` 装入 hardware-policy registry。
   * - ``tests/unit_tests/test_yam_hardware.py``
     - 测试 registry 转换、工作站枚举、资源冲突以及保持方向的夹爪标定。
   * - ``tests/unit_tests/test_yam_runtime.py``
     - 测试延迟连接、命令安全、反馈超时、角色相关 i2rt 模式和清理。
   * - ``tests/unit_tests/test_yam_env.py``
     - 测试 dummy Gym 契约、14 维顺序与幂等关闭。
   * - ``tests/unit_tests/test_yam_intervention.py``
     - 测试策略/主臂所有权、同步失败清理和回合结束行为。
   * - ``tests/unit_tests/test_yam_imports.py``
     - 守护无硬件导入路径，确保 ``i2rt`` 保持延迟加载。
   * - ``tests/unit_tests/test_yam_examples.py``
     - 守护公开 YAML 契约、直接 LeRobot 设置和“环境内固定 SDK、不依赖应用仓库”的安装规则。

已知边界
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 范围
     - 当前边界
   * - 控制表示
     - 目前只有绝对 14 维关节空间；TCP/笛卡尔动作需要另建带版本的环境和数据契约。
   * - 相机
     - 只支持 RealSense RGB，``enable_depth`` 必须为 ``false``。
   * - 放置
     - 从臂、电机主臂和相机必须与同一个环境 worker 共节点；尚未实现按机械臂拆分的远程 controller。
   * - 干预粒度
     - 同步和干预同时作用于双臂，尚无左右独立 intervention mask。
   * - 模型
     - 当前示例只采数据，尚未提供 YAM 专用模型 dataconfig、归一化统计、SFT recipe 和策略部署示例。
   * - SDK 调参
     - 重力系数和摩擦补偿开关使用 SDK 公开参数；逐设备数值型阻尼/摩擦覆盖需要兼容的 ``i2rt`` 构造函数，否则会被拒绝。
   * - 故障响应
     - 软件仅提供尽力而为的实测位置保持和资源清理，不能代替物理急停。
   * - 验证范围
     - 单元测试使用 mock；正式双臂采集前仍需固定 SDK 版本，并逐臂、低速完成真机验收。
