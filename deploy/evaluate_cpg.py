"""Headless fixed-command evaluation of the deployed CPG policy.

Example:
    python deploy/evaluate_cpg.py --output-dir logs/cpg_evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

import bsrl_cpg_deploy as deploy


# (duration in seconds, forward velocity in m/s, yaw rate in rad/s)
DEFAULT_SCHEDULE = (
    (2.0, 0.0, 0.0),
    (4.0, 0.5, 0.0),
    (4.0, 1.0, 0.0),
    (2.0, 0.0, 0.0),
    (4.0, 0.8, 0.25),
    (4.0, 0.8, -0.25),
    (3.0, 0.0, 0.0),
)
JOINT_LABELS = ("Left hip", "Left knee", "Right hip", "Right knee")
COLORS = ("#276C9B", "#D98736", "#4D8D73", "#9B5D82")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=deploy.DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=deploy.PROJECT_ROOT / "logs" / "cpg_evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--schedule",
        type=Path,
        help="Optional YAML list of [duration_s, target_vx_mps, target_yaw_rps] segments.",
    )
    return parser.parse_args()


def load_schedule(path: Path | None) -> tuple[tuple[float, float, float], ...]:
    raw = DEFAULT_SCHEDULE if path is None else yaml.safe_load(path.read_text(encoding="utf-8"))
    schedule = tuple(tuple(map(float, segment)) for segment in raw)
    if not schedule or any(
        len(segment) != 3 or not np.isfinite(segment).all() or segment[0] <= 0 or segment[1] < 0
        for segment in schedule
    ):
        raise ValueError("Schedule needs [positive duration, nonnegative vx, finite yaw rate] segments")
    return schedule


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData, foot_ids: tuple[int, int], ground_id: int) -> tuple[int, int]:
    """Return contact flags for the two foot bodies touching the ground plane."""
    flags = [0, 0]
    for contact in data.contact[:data.ncon]:
        if ground_id not in (contact.geom1, contact.geom2):
            continue
        other_geom = contact.geom2 if contact.geom1 == ground_id else contact.geom1
        body_id = int(model.geom_bodyid[other_geom])
        for side, foot_id in enumerate(foot_ids):
            if body_id == foot_id:
                flags[side] = 1
    return flags[0], flags[1]


def record_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: deploy.RuntimeState,
    cpg: deploy.LowLimbCPG,
    binding: deploy.JointBinding,
    joint_ids: np.ndarray,
    foot_ids: tuple[int, int],
    ground_id: int,
    segment_index: int,
) -> dict[str, float | int]:
    body_velocity, body_rate = deploy.root_velocity_body(data)
    gravity = deploy.projected_gravity(data.qpos[3:7])
    reference = deploy.cpg_reference_in_policy_order(cpg)
    actual = data.qpos[binding.qpos_addr[joint_ids]]
    left_contact, right_contact = foot_contacts(model, data, foot_ids, ground_id)
    row: dict[str, float | int] = {
        "time_s": float(data.time),
        "segment": segment_index,
        "target_vx_mps": float(state.command[0]),
        "actual_vx_mps": float(body_velocity[0]),
        "actual_vy_mps": float(body_velocity[1]),
        "target_yaw_rps": float(state.command[2]),
        "actual_yaw_rps": float(body_rate[2]),
        "base_height_m": float(data.qpos[2]),
        "base_roll_rad": float(np.arctan2(-gravity[1], -gravity[2])),
        "base_pitch_rad": float(np.arctan2(gravity[0], -gravity[2])),
        "left_foot_contact": left_contact,
        "right_foot_contact": right_contact,
        "cpg_rhythm_x": float(cpg.rhythm.x),
        "cpg_rhythm_y": float(cpg.rhythm.y),
        "cpg_frequency_hz": float(cpg.rhythm.frequency),
        "torque_rms_nm": float(np.sqrt(np.mean(np.square(state.joint_effort)))),
    }
    for index, name in enumerate(("left_hip", "left_knee", "right_hip", "right_knee")):
        row[f"{name}_reference_rad"] = float(reference[index])
        row[f"{name}_actual_rad"] = float(actual[index])
    return row


def simulate(config: dict, schedule: tuple[tuple[float, float, float], ...], seed: int) -> list[dict[str, float | int]]:
    np.random.seed(seed)
    simulation_dt = float(config["simulation_dt"])
    decimation = int(config["control_decimation"])
    if simulation_dt <= 0 or decimation <= 0:
        raise ValueError("simulation_dt and control_decimation must be positive")
    control_dt = simulation_dt * decimation
    staged, xml_path = deploy.stage_assets_to_ascii_path(deploy.resolve_path(config["xml_path"]))
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.timestep = simulation_dt
        data = mujoco.MjData(model)
        joint_names = list(config["policy_joint_names"])
        deploy.require_model_layout(model, list(config["mujoco_joint_names"]), joint_names)
        binding = deploy.bind_joints(model, joint_names)
        model.dof_armature[binding.qvel_addr] = float(config["joint_armature"])
        joint_ids = np.asarray([joint_names.index(name) for name in deploy.CPG_JOINT_NAMES], dtype=np.int32)
        foot_ids = tuple(deploy.model_body_id(model, name) for name in ("link_left_ankle_roll", "link_right_ankle_roll"))
        ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        if ground_id < 0:
            raise ValueError("Ground geom 'ground' not found")

        defaults = np.asarray(config["default_angles"], dtype=np.float64)
        kps = np.asarray(config["kps"], dtype=np.float64)
        kds = np.asarray(config["kds"], dtype=np.float64)
        powers = np.asarray(config["peak_powers"], dtype=np.float64) if "peak_powers" in config else None
        limits = np.asarray(config["velocity_limits"], dtype=np.float64) if "velocity_limits" in config else None
        state = deploy.create_runtime_state(config, defaults, kds)
        cpg = deploy.LowLimbCPG()
        session, input_name, output_name = deploy.load_policy(deploy.resolve_path(config["policy_path"]))
        deploy.validate_policy_shape(session, int(config["num_obs"]), int(config["num_actions"]))
        deploy.reset_runtime(
            model, data, binding, state, cpg, deploy.CommandController(config), defaults,
            float(config["root_height"]), config, session, input_name, output_name,
        )

        rows: list[dict[str, float | int]] = []
        elapsed_steps = 0
        for segment_index, (duration, vx, yaw) in enumerate(schedule):
            end_step = round(sum(item[0] for item in schedule[:segment_index + 1]) / control_dt)
            while elapsed_steps < end_step:
                state.command[:] = (vx, 0.0, yaw)
                cpg.step(vx, control_dt)
                rhythm_xy = np.asarray((cpg.rhythm.x, cpg.rhythm.y), dtype=np.float32)
                obs = deploy.build_observation(data, binding, state, defaults, rhythm_xy, config)
                state.action[:] = deploy.run_policy(session, input_name, output_name, obs, config.get("clip_actions"))
                state.target_q[:] = state.action * float(config["action_scale"]) + defaults
                for _ in range(decimation):
                    deploy.step_control(model, data, binding, state, kps, kds, powers, limits)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError(f"Non-finite MuJoCo state at t={data.time:.3f} s")
                rows.append(record_row(model, data, state, cpg, binding, joint_ids, foot_ids, ground_id, segment_index))
                elapsed_steps += 1
        return rows
    finally:
        staged.cleanup()


def plot_results(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    def series(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=float)

    time = series("time_s")
    figure, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    for axis, name, title, color in zip(
        axes, ("left_hip", "left_knee", "right_hip", "right_knee"), JOINT_LABELS, COLORS
    ):
        axis.plot(time, series(f"{name}_reference_rad"), color=color, linewidth=1.4, label="CPG reference")
        axis.plot(time, series(f"{name}_actual_rad"), color="#252D35", linewidth=1.0, alpha=0.8, label="Actual")
        axis.set_ylabel(f"{title} [rad]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("CPG reference and actual joint trajectories")
    figure.tight_layout()
    figure.savefig(output_dir / "joint_trajectories.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(time, series("target_vx_mps"), label="Target", linewidth=1.5)
    axes[0].plot(time, series("actual_vx_mps"), label="Actual", linewidth=1.0)
    axes[0].set_ylabel("Forward velocity [m/s]")
    axes[1].plot(time, series("target_yaw_rps"), label="Target", linewidth=1.5)
    axes[1].plot(time, series("actual_yaw_rps"), label="Actual", linewidth=1.0)
    axes[1].set_ylabel("Yaw rate [rad/s]")
    axes[2].step(time, series("left_foot_contact"), where="post", label="Left foot")
    axes[2].step(time, series("right_foot_contact") + 1.2, where="post", label="Right foot")
    axes[2].set_yticks((0, 1, 1.2, 2.2), ("L off", "L on", "R off", "R on"))
    axes[2].set_ylabel("Ground contact")
    axes[3].plot(time, series("base_height_m"), label="Base height")
    axes[3].set_ylabel("Base height [m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("Velocity tracking, foot contacts, and base height")
    figure.tight_layout()
    figure.savefig(output_dir / "motion_and_contacts.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(time, np.rad2deg(series("base_roll_rad")), label="Roll")
    axes[0].plot(time, np.rad2deg(series("base_pitch_rad")), label="Pitch")
    axes[0].set_ylabel("Base tilt [deg]")
    axes[1].plot(time, series("actual_vy_mps"), label="Lateral velocity")
    axes[1].set_ylabel("Lateral velocity [m/s]")
    axes[2].plot(time, series("torque_rms_nm"), label="Joint torque RMS")
    axes[2].set_ylabel("Torque [N m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("Balance and control effort")
    figure.tight_layout()
    figure.savefig(output_dir / "balance_and_effort.png", dpi=170)
    plt.close(figure)


def summarize(rows: list[dict[str, float | int]]) -> dict[str, object]:
    def values(name: str, selected: list[dict[str, float | int]] = rows) -> np.ndarray:
        return np.asarray([row[name] for row in selected], dtype=float)

    def rms(error: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(error))))

    summary: dict[str, object] = {
        "samples": len(rows),
        "duration_s": float(rows[-1]["time_s"]),
        "forward_velocity_rmse_mps": rms(values("actual_vx_mps") - values("target_vx_mps")),
        "yaw_rate_rmse_rps": rms(values("actual_yaw_rps") - values("target_yaw_rps")),
        "lateral_velocity_rms_mps": rms(values("actual_vy_mps")),
        "minimum_base_height_m": float(np.min(values("base_height_m"))),
        "base_roll_rms_deg": float(np.rad2deg(rms(values("base_roll_rad")))),
        "base_pitch_rms_deg": float(np.rad2deg(rms(values("base_pitch_rad")))),
        "mean_joint_torque_rms_nm": float(np.mean(values("torque_rms_nm"))),
        "left_contact_fraction": float(np.mean(values("left_foot_contact"))),
        "right_contact_fraction": float(np.mean(values("right_foot_contact"))),
        "both_feet_off_fraction": float(np.mean((values("left_foot_contact") + values("right_foot_contact")) == 0)),
    }
    summary["joint_tracking_rmse_rad"] = {
        name: rms(values(f"{name}_actual_rad") - values(f"{name}_reference_rad"))
        for name in ("left_hip", "left_knee", "right_hip", "right_knee")
    }
    summary["segments"] = []
    for segment in sorted({int(row["segment"]) for row in rows}):
        selected = [row for row in rows if row["segment"] == segment]
        summary["segments"].append({
            "segment": segment,
            "samples": len(selected),
            "target_vx_mps": float(selected[0]["target_vx_mps"]),
            "actual_vx_mean_mps": float(np.mean(values("actual_vx_mps", selected))),
            "target_yaw_rps": float(selected[0]["target_yaw_rps"]),
            "actual_yaw_mean_rps": float(np.mean(values("actual_yaw_rps", selected))),
            "minimum_base_height_m": float(np.min(values("base_height_m", selected))),
        })
    return summary


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    schedule = load_schedule(args.schedule)
    rows = simulate(config, schedule, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trace.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["seed"] = args.seed
    summary["config_file"] = str(args.config.resolve())
    summary["schedule"] = [
        {"duration_s": duration, "target_vx_mps": vx, "target_yaw_rps": yaw}
        for duration, vx, yaw in schedule
    ]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(rows, args.output_dir)
    print(f"Saved {len(rows)} samples and plots to {args.output_dir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
