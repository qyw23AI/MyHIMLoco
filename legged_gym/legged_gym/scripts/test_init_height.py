"""
Test different init_state.pos.z values for mybot_v3
to find the optimal neutral standing height.

Fixed default joint angles:
    thigh = 0.8
    calf  = -1.4
    hip   = 0.0

Test range:
    0.34 ~ 0.42
"""



from isaacgym import gymtorch

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

import torch
import numpy as np


def test_init_heights():

    args = get_args()

    args.task = "mybot_v3"
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(
        name=args.task
    )

    # =========================================================
    # Override config
    # =========================================================

    env_cfg.env.num_envs = 1

    env_cfg.terrain.mesh_type = "plane"

    env_cfg.noise.add_noise = False

    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False

    env_cfg.commands.heading_command = False

    # disable command motion
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    # =========================================================
    # Fixed standing pose
    # =========================================================

    # env_cfg.init_state.default_joint_angles = {

    #     "FL_hip_joint": 0.0,
    #     "FR_hip_joint": 0.0,
    #     "RL_hip_joint": 0.0,
    #     "RR_hip_joint": 0.0,

    #     "FL_thigh_joint": 0.8,
    #     "FR_thigh_joint": 0.8,
    #     "RL_thigh_joint": 0.8,
    #     "RR_thigh_joint": 0.8,

    #     "FL_calf_joint": -1.4,
    #     "FR_calf_joint": -1.4,
    #     "RL_calf_joint": -1.4,
    #     "RR_calf_joint": -1.4,
    # }
    env_cfg.init_state.default_joint_angles = {

            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,

            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,

            "FL_calf_joint": -1.4,
            "RL_calf_joint": -1.4,
            "FR_calf_joint": -1.4,
            "RR_calf_joint": -1.4,
    }
    # =========================================================
    # Create environment
    # =========================================================

    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg
    )

    # =========================================================
    # Get foot body names
    # =========================================================

    body_names = env.gym.get_asset_rigid_body_names(
        env.gym.get_actor_asset(
            env.envs[0],
            env.actor_handles[0]
        )
    )

    foot_names = [
        s for s in body_names
        if env.cfg.asset.foot_name in s
    ]

    print("\nFoot bodies:")
    print(foot_names)

    # =========================================================
    # Test range
    # =========================================================

    z_values = np.arange(0.34, 0.44, 0.01)

    results = []

    # =========================================================
    # Main loop
    # =========================================================

    for z in z_values:

        z = round(float(z), 2)

        print(f"\n{'=' * 70}")
        print(f"Testing init_state.pos.z = {z}")
        print(f"{'=' * 70}")

        # -----------------------------------------------------
        # Reset environment
        # -----------------------------------------------------

        env_ids = torch.arange(
            env.num_envs,
            device=env.device
        )

        env.cfg.init_state.pos[2] = z

        base_init_state_list = (
            env.cfg.init_state.pos
            + env.cfg.init_state.rot
            + env.cfg.init_state.lin_vel
            + env.cfg.init_state.ang_vel
        )

        env.base_init_state = torch.tensor(
            base_init_state_list,
            device=env.device,
            dtype=torch.float
        )

        env.reset_idx(env_ids)

        # force update root z
        env.root_states[:, 2] = z

        env.gym.set_actor_root_state_tensor(
            env.sim,
            gymtorch.unwrap_tensor(
                env.root_states
            )
        )

        # -----------------------------------------------------
        # Let physics settle
        # -----------------------------------------------------

        for _ in range(100):

            actions = torch.zeros(
                env.num_envs,
                env.num_actions,
                device=env.device
            )

            env.step(actions)

        # -----------------------------------------------------
        # Read state
        # -----------------------------------------------------

        base_z = env.root_states[0, 2].item()

        base_lin_vel = torch.norm(
            env.root_states[0, 7:10]
        ).item()

        foot_z = env.feet_pos[
            0,
            :,
            2
        ].cpu().numpy()

        foot_contact_forces = (
            env.contact_forces[
                0,
                env.feet_indices,
                2
            ]
            .cpu()
            .numpy()
        )

        print(f"\nBase z        : {base_z:.4f} m")
        print(f"Base velocity : {base_lin_vel:.4f} m/s")

        print("\nFoot states:")

        for i, name in enumerate(foot_names):

            contact = (
                foot_contact_forces[i] > 1.0
            )

            status = (
                "CONTACT"
                if contact
                else "FLOAT"
            )

            print(
                f"{name:12s} | "
                f"z = {foot_z[i]:+.4f} m | "
                f"Fz = {foot_contact_forces[i]:7.2f} N | "
                f"{status}"
            )

        # -----------------------------------------------------
        # Evaluate
        # -----------------------------------------------------

        min_foot_z = np.min(foot_z)

        all_contact = np.all(
            foot_contact_forces > 1.0
        )

        any_contact = np.any(
            foot_contact_forces > 1.0
        )

        stable = base_lin_vel < 0.15

        if min_foot_z < -0.025:

            result = "TOO LOW (penetration)"

        elif not any_contact:

            result = "TOO HIGH (floating)"

        elif not stable:

            result = "UNSTABLE"

        elif all_contact:

            result = "GOOD"

        else:

            result = "PARTIAL CONTACT"

        print(f"\nResult: {result}")

        results.append({
            "z": z,
            "result": result,
            "base_vel": base_lin_vel,
            "min_foot_z": min_foot_z
        })

    # =========================================================
    # Final summary
    # =========================================================

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    good_results = [
        r for r in results
        if r["result"] == "GOOD"
    ]

    for r in results:

        print(
            f"z={r['z']:.2f} | "
            f"{r['result']:24s} | "
            f"base_vel={r['base_vel']:.4f}"
        )

    if len(good_results) > 0:

        best = min(
            good_results,
            key=lambda x: x["base_vel"]
        )

        print(
            f"\nBest init_state.pos.z = "
            f"{best['z']:.2f}"
        )

    else:

        print("\nNo GOOD standing pose found.")

    # =========================================================
    # Cleanup
    # =========================================================

    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":

    test_init_heights()