#!/usr/bin/env python
"""
Terrain Viewer - 纯地形可视化脚本（不加载机器人）

用于在地形训练前预览和调试 BarrierTrack 地形。
所有14种障碍物类型都可以单独或组合查看。

用法:
    # 默认：展示全部四种 Parkour 障碍物
    python legged_gym/legged_gym/scripts/terrain_viewer.py

    # 查看特定障碍物
    python legged_gym/legged_gym/scripts/terrain_viewer.py --options hurdle,leap,stairsup,tilt

    # 查看所有14种障碍物
    python legged_gym/legged_gym/scripts/terrain_viewer.py --options all

    # 多行多列展示
    python legged_gym/legged_gym/scripts/terrain_viewer.py --num-rows 3 --num-cols 8

    # 展示特定难度级别
    python legged_gym/legged_gym/scripts/terrain_viewer.py --difficulty 0.5

键盘控制:
    W/S/A/D/Q/E - 移动相机
    鼠标右键拖动 - 旋转视角
    滚轮 - 缩放
    [/] - 切换难度级别
    ;/'  - 切换列（障碍物类型）
    P   - 切换 Perlin 噪声
    R   - 重置视角
    ESC - 退出
"""

import numpy as np
import sys
import os
import argparse

# 确保 legged_gym 在 Python path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if LEGGED_GYM_ROOT not in sys.path:
    sys.path.insert(0, LEGGED_GYM_ROOT)

import isaacgym
from isaacgym import gymapi, gymutil
import legged_gym.envs  # must import envs before utils to break circular import
from legged_gym.utils.terrain.barrier_track import BarrierTrack


# ============================================================================
# 所有支持的障碍物类型及其默认参数
# ============================================================================
ALL_OBSTACLE_TYPES = [
    "jump", "hurdle", "leap", "tilt", "crawl", "down",
    "stairsup", "stairsdown", "slope", "slopeup", "slopedown",
    "tilted_ramp", "discrete_rect", "wave",
    "bridge_a", "bridge_b", "t_stairs",
]

# Parkour 任务默认障碍物（不含 tilt，已被 bridge 替代）
PARKOUR_OBSTACLES = ["hurdle", "stairsup", "bridge_a", "bridge_b", "t_stairs"]

# 障碍物默认参数（与 distill config 的目标参数对齐）
DEFAULT_OBSTACLE_PARAMS = {
    "jump":    dict(height=(0.25, 0.35), depth=(0.05, 0.15), fake_offset=0.0),
    "hurdle":  dict(height=(0.15, 0.35), depth=(0.05, 0.08)),   # 30cm木板, 3-5cm厚
    "tilt":    dict(width=(0.15, 0.35), depth=(0.8, 1.5), opening_angle=0.0, wall_height=0.5),
    "crawl":   dict(height=(0.25, 0.35), depth=(0.04, 0.08), wall_height=0.8),
    "down":    dict(height=(0.1, 0.25), depth=(0.1, 0.3)),
    "stairsup":   dict(height=(0.08, 0.15), depth=(0.1, 0.3), n_stairs=2),
    "stairsdown": dict(height=(0.08, 0.15), depth=(0.1, 0.3), n_stairs=3),
    "slope":   dict(slope_angle=[0.2, 0.5], face_angle=[-3.14, 3.14], length=[1.2, 2.0]),
    "slopeup": dict(slope_angle=[0.2, 0.5], face_angle=[-0.3, 0.3], length=[1.2, 2.0]),
    "slopedown": dict(slope_angle=[0.2, 0.5], face_angle=[-0.3, 0.3], length=[1.2, 2.0]),
    "tilted_ramp": dict(tilt_angle=0.3, switch_spacing=0.5, overlap_size=0.1, depth=0.2, length=0.0),
    "discrete_rect": dict(max_height=0.2, max_size=0.8, min_size=0.2, num_rects=16),
    "wave":    dict(amplitude=(0.05, 0.15), frequency=(2.0, 5.0)),
    "bridge_a":  dict(n_beams=5, beam_width=0.40, gap_width=0.15, bridge_height=0.15, bridge_mesh_source="procedural", rotate_90=True, ramp_length=0.40),
    "bridge_b":  dict(n_beams=3, beam_width=0.20, gap_width=0.10, bridge_height=0.20, bridge_mesh_source="procedural", rotate_90=False, ramp_length=0.40),
    "t_stairs": dict(step_height=0.10, step_depth=0.18, n_steps=4, stair_width=2.00, platform_width=0.8, rotate_90=False),
}


def create_terrain_config(args):
    """构建 BarrierTrack 需要的 terrain config 对象"""
    cfg = type("Config", (), {})()
    cfg.mesh_type = None
    cfg.num_rows = args.num_rows
    cfg.num_cols = args.num_cols
    cfg.horizontal_scale = 0.025
    cfg.vertical_scale = 0.005
    cfg.border_size = args.border_size
    cfg.slope_treshold = 20.0
    cfg.curriculum = args.curriculum
    cfg.static_friction = 1.0
    cfg.dynamic_friction = 1.0
    cfg.restitution = 0.0
    cfg.max_init_terrain_level = 0
    cfg.track_block_length = args.track_block_length
    cfg.track_width = args.track_width
    cfg.measure_heights = False
    cfg.pad_unavailable_info = False

    # 障碍物选择
    if args.options == "all":
        options = list(ALL_OBSTACLE_TYPES)
    else:
        options = [o.strip() for o in args.options.split(",")]

    # 构建 BarrierTrack_kwargs
    track_kwargs = dict(
        options=options,
        n_obstacles_per_track=args.n_obstacles_per_track,
        randomize_obstacle_order=args.randomize_order,
        track_width=args.track_width,
        track_block_length=args.track_block_length,
        wall_thickness=(args.wall_thickness_min, args.wall_thickness_max),
        wall_height=args.wall_height,
        add_perlin_noise=args.perlin_noise,
        border_perlin_noise=args.perlin_noise,
        border_height=args.border_height,
        virtual_terrain=False,
        draw_virtual_terrain=True,
        engaging_next_threshold=1.2,
        engaging_finish_threshold=0.0,
        curriculum_perlin=False,
        no_perlin_threshold=0.04,
        walk_in_skill_gap=True,
    )

    # 添加各障碍物参数
    for obs_name in options:
        if obs_name in DEFAULT_OBSTACLE_PARAMS:
            track_kwargs[obs_name] = dict(DEFAULT_OBSTACLE_PARAMS[obs_name])
            # 桥的 mesh 构建方式可通过 CLI 覆盖（仅当显式指定时）
            if obs_name in ("bridge_a", "bridge_b") and args.bridge_mesh_source is not None:
                track_kwargs[obs_name]["bridge_mesh_source"] = args.bridge_mesh_source

    cfg.BarrierTrack_kwargs = track_kwargs

    cfg.TerrainPerlin_kwargs = dict(
        zScale=[args.perlin_zscale_min, args.perlin_zscale_max],
        frequency=10,
    )

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="BarrierTrack Terrain Viewer - 纯地形预览工具"
    )
    # 地形布局参数
    parser.add_argument("--num-rows", type=int, default=1,
                        help="难度级别行数 (default: 1)")
    parser.add_argument("--num-cols", type=int, default=4,
                        help="地形列数 (default: 4)")
    parser.add_argument("--border-size", type=float, default=3.0,
                        help="地形边界大小 (m) (default: 3.0)")

    # 障碍物选择
    parser.add_argument("--options", type=str, default="hurdle,stairsup,bridge_a,bridge_b,t_stairs",
                        help="障碍物类型，逗号分隔，或 'all' 查看全部 (default: hurdle,stairsup,bridge_a,bridge_b,t_stairs)")
    parser.add_argument("--n-obstacles-per-track", type=int, default=4,
                        help="每条轨道障碍物数量 (default: 4)")
    parser.add_argument("--randomize-order", action="store_true", default=False,
                        help="随机排列障碍物顺序 (default: False，即按 options 顺序)")
    parser.add_argument("--bridge-mesh-source", type=str, default=None,
                        choices=["procedural", "platform", "stl"],
                        help="覆盖所有桥的 mesh 构建方式，默认不覆盖（使用各障碍物自身配置）")

    # 轨道参数
    parser.add_argument("--track-width", type=float, default=2.4,
                        help="轨道宽度 (m) (default: 2.4)")
    parser.add_argument("--track-block-length", type=float, default=2.4,
                        help="每块长度 (m) (default: 2.4)")
    parser.add_argument("--wall-thickness-min", type=float, default=0.04,
                        help="侧壁最小厚度 (m) (default: 0.04)")
    parser.add_argument("--wall-thickness-max", type=float, default=0.2,
                        help="侧壁最大厚度 (m) (default: 0.2)")
    parser.add_argument("--wall-height", type=float, default=0.3,
                        help="侧壁高度 (m)，负值=低于地面 (default: 0.3)")
    parser.add_argument("--border-height", type=float, default=0.0,
                        help="边界高度 (m) (default: 0.0)")

    # 难度
    parser.add_argument("--curriculum", action="store_true", default=False,
                        help="使用课程难度（行的梯度） (default: False)")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="固定难度级别 [0.0, 1.0]，None 则随机 (default: None)")

    # Perlin 噪声
    parser.add_argument("--perlin-noise", action="store_true", default=False,
                        help="启用 Perlin 噪声 (default: False)")
    parser.add_argument("--perlin-zscale-min", type=float, default=0.0,
                        help="Perlin 噪声最小幅度 (m) (default: 0.0)")
    parser.add_argument("--perlin-zscale-max", type=float, default=0.02,
                        help="Perlin 噪声最大幅度 (m) (default: 0.02)")

    # Isaac Gym 参数
    parser.add_argument("--headless", action="store_true", default=False,
                        help="无头模式（无显示）")
    parser.add_argument("--sim-device", type=str, default="cuda:0",
                        help="模拟设备 (default: cuda:0)")
    parser.add_argument("--graphics-device", type=int, default=0,
                        help="图形设备 ID (default: 0)")
    parser.add_argument("--physics-engine", type=str, default="physx",
                        help="物理引擎 (default: physx)")
    parser.add_argument("--compute-device-id", type=int, default=0,
                        help="计算设备 ID (default: 0)")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                        help="使用 GPU (default: True)")
    parser.add_argument("--use-gpu-pipeline", action="store_true", default=True,
                        help="使用 GPU 管线 (default: True)")

    args = parser.parse_args()

    # =========================================================================
    # 1. 初始化 Isaac Gym
    # =========================================================================
    print("=" * 60)
    print("  BarrierTrack Terrain Viewer")
    print("=" * 60)
    print(f"  障碍物: {args.options}")
    print(f"  布局: {args.num_rows} 行 x {args.num_cols} 列")
    print(f"  Perlin 噪声: {'开' if args.perlin_noise else '关'}")
    print(f"  轨道宽度: {args.track_width}m, 块长度: {args.track_block_length}m")
    print("=" * 60)

    gym = gymapi.acquire_gym()

    # 配置 sim params
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.num_subscenes = 4
    sim_params.physx.num_threads = 4
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # 解析设备
    if args.physics_engine == "physx":
        physics_engine = gymapi.SIM_PHYSX
    elif args.physics_engine == "flex":
        physics_engine = gymapi.SIM_FLEX
    else:
        physics_engine = gymapi.SIM_PHYSX

    sim_device_id = 0 if args.sim_device == "cpu" else 0
    graphics_device_id = args.graphics_device

    # 创建 sim
    print("创建 simulation...")
    sim = gym.create_sim(sim_device_id, graphics_device_id, physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create simulation!")

    # =========================================================================
    # 2. 创建地形
    # =========================================================================
    print("创建地形...")
    terrain_cfg = create_terrain_config(args)

    # 构建 num_envs (rows * cols)
    num_envs = args.num_rows * args.num_cols

    # 创建 BarrierTrack 实例
    barrier_track = BarrierTrack(terrain_cfg, num_envs)

    # 如果需要固定难度，通过 monkey-patch get_difficulty
    if args.difficulty is not None:
        original_get_difficulty = barrier_track.get_difficulty
        def fixed_get_difficulty(env_row_idx, env_col_idx):
            return (args.difficulty, False)
        barrier_track.get_difficulty = fixed_get_difficulty

    # 添加地形到 sim
    device = args.sim_device if args.sim_device != "cpu" else "cpu"
    barrier_track.add_terrain_to_sim(gym, sim, device)

    # =========================================================================
    # 3. 添加基础地面平面（确保地形外有地面）
    # =========================================================================
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # Z-up
    plane_params.distance = 0.0
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)

    # =========================================================================
    # 4. 创建 Viewer
    # =========================================================================
    if not args.headless:
        camera_props = gymapi.CameraProperties()
        viewer = gym.create_viewer(sim, camera_props)
        if viewer is None:
            print("Warning: Failed to create viewer, running headless")
            args.headless = True
            viewer = None
    else:
        viewer = None

    # 设置初始相机位置（俯瞰视角）
    if viewer is not None:
        total_width = terrain_cfg.num_cols * (
            barrier_track.env_width
        ) + 2 * terrain_cfg.border_size
        total_length = terrain_cfg.num_rows * (
            barrier_track.n_blocks_per_track * terrain_cfg.track_block_length
        ) + 2 * terrain_cfg.border_size
        cam_pos = gymapi.Vec3(
            total_length / 2,
            total_width / 2 - 2.0,
            4.0,
        )
        cam_target = gymapi.Vec3(
            total_length / 2,
            total_width / 2,
            0.3,
        )
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # =========================================================================
    # 5. 打印地形信息
    # =========================================================================
    print("\n" + "=" * 60)
    print("  地形信息")
    print("=" * 60)
    print(f"  轨道总尺寸: {barrier_track.env_length:.1f}m (长) x "
          f"{barrier_track.env_width:.1f}m (宽)")
    print(f"  每轨道块数: {barrier_track.n_blocks_per_track} "
          f"(起始块 + {barrier_track.n_blocks_per_track - 1} 个障碍物)")
    print(f"  块尺寸: {terrain_cfg.track_block_length:.1f}m x "
          f"{terrain_cfg.track_width:.1f}m")
    print(f"  障碍物顺序: {barrier_track.track_kwargs['options']}")

    # 打印各障碍物实际参数
    for i, obs_name in enumerate(barrier_track.track_kwargs["options"]):
        if obs_name in barrier_track.track_kwargs:
            params = barrier_track.track_kwargs[obs_name]
            param_str = "  ".join(f"{k}={v}" for k, v in params.items())
            print(f"  障碍物[{i}]: {obs_name} -> {param_str}")

    print("=" * 60)
    print("\n键盘控制:")
    print("  W/S/A/D/Q/E - 移动相机")
    print("  鼠标右键拖动 - 旋转视角")
    print("  滚轮 - 缩放")
    print("  [/] - 切换难度视角（行）")
    print("  ;/'  - 切换障碍物视角（列）")
    print("  P   - 切换 Perlin 噪声（重新生成地形）")
    print("  R   - 重置视角")
    print("  ESC - 退出")
    print()

    # =========================================================================
    # 6. 主渲染循环
    # =========================================================================
    if args.headless:
        print("Headless mode. Exiting.")
        return

    # 相机跟踪状态
    current_row = 0
    current_col = 0
    camera_pos = np.array([total_length / 2, total_width / 2 - 2.0, 4.0])
    camera_target = np.array([total_length / 2, total_width / 2, 0.3])
    move_speed = 0.5  # m/s
    rotate_speed = 0.5  # rad/s

    prev_time = None

    while not gym.query_viewer_has_closed(viewer):
        # 处理键盘事件
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "exit" and evt.value > 0:
                gym.destroy_viewer(viewer)
                print("退出.")
                return

        # 简单的相机移动（通过轮询方式不可用，使用固定的俯瞰视角）
        # Isaac Gym viewer 自带自由飞行相机，用户可以用鼠标自由探索

        # 渲染
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    print("退出.")


if __name__ == "__main__":
    main()
