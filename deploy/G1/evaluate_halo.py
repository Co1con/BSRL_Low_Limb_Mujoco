"""Headless fixed-command evaluation of the deployed G1 HALO policy.

Example:
    python deploy/G1/evaluate_halo.py --output-dir logs/g1_halo_evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort
import yaml

import g1_halo_deploy as deploy


JOINT_KEYS = ("left_hip", "left_knee", "right_hip", "right_knee")
JOINT_LABELS = ("Left hip", "Left knee", "Right hip", "Right knee")
COLORS = ("#276C9B", "#D98736", "#4D8D73", "#9B5D82")

# (duration in seconds, forward velocity in m/s, yaw rate in rad/s)
DEFAULT_SCHEDULE = (
    (2.0, 0.0, 0.0),
    (3.0, 0.8, 0.0),
    (4.0, 0.8, 0.3),
    (4.0, 0.8, -0.3),
    (3.0, -0.8, 0.0),
    (4.0, -0.8, 0.3),
    (4.0, -0.8, -0.3),
    (2.0, 0.0, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=deploy.DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=deploy.PROJECT_ROOT / "logs" / "g1_halo_evaluation")
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
        len(segment) != 3 or not np.isfinite(segment).all() or segment[0] <= 0
        for segment in schedule
    ):
        raise ValueError("Schedule needs [positive duration, finite signed vx, finite yaw rate] segments")
    return schedule


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData, foot_ids: tuple[int, int], ground_id: int) -> tuple[int, int]:
    flags = [0, 0]
    for contact in data.contact[: data.ncon]:
        if ground_id not in (contact.geom1, contact.geom2):
            continue
        other_geom = contact.geom2 if contact.geom1 == ground_id else contact.geom1
        body_id = int(model.geom_bodyid[other_geom])
        for side, foot_id in enumerate(foot_ids):
            if body_id == foot_id:
                flags[side] = 1
    return flags[0], flags[1]


def simulate(
    config: dict, schedule: tuple[tuple[float, float, float], ...], seed: int
) -> list[dict[str, float | int]]:
    np.random.seed(seed)
    simulation_dt = float(config["simulation_dt"])
    decimation = int(config["control_decimation"])
    if simulation_dt <= 0.0 or decimation <= 0:
        raise ValueError("simulation_dt and control_decimation must be positive")
    control_dt = simulation_dt * decimation
    names = list(config["policy_joint_names"])
    default_q = np.asarray([config["default_joint_angles"].get(name, 0.0) for name in names], dtype=np.float64)
    kp, kd, velocity_limits = np.asarray([deploy.joint_gains(name) for name in names], dtype=np.float64).T
    xml_path = Path(config["xml_path"].replace("{PROJECT_ROOT}", str(deploy.PROJECT_ROOT))).resolve()
    policy_path = Path(config["policy_path"].replace("{PROJECT_ROOT}", str(deploy.PROJECT_ROOT))).resolve()

    with tempfile.TemporaryDirectory(prefix="g1_evaluation_") as temp_root:
        staged_assets = Path(temp_root) / "G1"
        shutil.copytree(xml_path.parent, staged_assets)
        model = mujoco.MjModel.from_xml_path(str(staged_assets / xml_path.name))
        model.opt.timestep = simulation_dt
        data = mujoco.MjData(model)
        qpos_ids, qvel_ids, motor_ids = deploy.joint_binding(model, names)
        model.dof_armature[qvel_ids] = 0.01
        effort_limits = np.asarray([deploy.joint_effort_limit(name) for name in names], dtype=np.float64)
        model.actuator_ctrlrange[motor_ids] = np.column_stack((-effort_limits, effort_limits))
        joint_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
        model.jnt_actfrcrange[joint_ids] = np.column_stack((-effort_limits, effort_limits))
        halo_ids = np.asarray([names.index(name) for name in deploy.HALO_JOINT_NAMES], dtype=np.int32)
        foot_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        )
        ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if min(*foot_ids, ground_id) < 0:
            raise ValueError("G1 foot bodies or ground geom 'floor' not found")

        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape != [1, 127] or outputs[0].shape != [1, 29]:
            raise ValueError("Expected ONNX [1, 127] -> [1, 29]")
        input_name, output_name = inputs[0].name, outputs[0].name

        data.qpos[:] = 0.0
        data.qpos[2] = float(config["root_height"])
        data.qpos[3] = 1.0
        data.qpos[qpos_ids] = default_q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        halo = deploy.HALO()
        last_action = np.zeros(29, dtype=np.float32)
        effort = np.zeros(29, dtype=np.float64)
        target_q = default_q.copy()
        rows: list[dict[str, float | int]] = []
        command = np.zeros(3, dtype=np.float32)
        elapsed_steps = 0
        for segment_index, (duration, vx, yaw) in enumerate(schedule):
            end_step = round(sum(item[0] for item in schedule[: segment_index + 1]) / control_dt)
            while elapsed_steps < end_step:
                command[:] = (vx, 0.0, yaw)
                halo.step(vx, yaw, control_dt)
                obs = deploy.observation(data, qpos_ids, qvel_ids, default_q, effort, last_action, command, halo, config)
                action = session.run([output_name], {input_name: obs[None, :]})[0].reshape(-1)
                last_action[:] = np.clip(action, -float(config["clip_actions"]), float(config["clip_actions"]))
                target_q[:] = default_q + float(config["action_scale"]) * last_action
                for _ in range(decimation):
                    q, dq = data.qpos[qpos_ids], data.qvel[qvel_ids]
                    torque = np.clip(kp * (target_q - q) - kd * dq, -effort_limits, effort_limits)
                    torque = np.where((dq >= velocity_limits) & (torque > 0.0), 0.0, torque)
                    torque = np.where((dq <= -velocity_limits) & (torque < 0.0), 0.0, torque)
                    data.ctrl[motor_ids] = torque
                    effort[:] = torque
                    mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError(f"Non-finite MuJoCo state at t={data.time:.3f} s")

                linear_velocity, angular_velocity = deploy.root_velocity_body(data)
                roll, pitch = deploy.base_roll_pitch(data.qpos[3:7])
                reference = deploy.halo_reference_in_policy_order(halo)
                actual = data.qpos[qpos_ids[halo_ids]]
                left_contact, right_contact = foot_contacts(model, data, foot_ids, ground_id)
                row: dict[str, float | int] = {
                    "time_s": float(data.time), "segment": segment_index,
                    "target_vx_mps": float(command[0]), "actual_vx_mps": float(linear_velocity[0]),
                    "actual_vy_mps": float(linear_velocity[1]), "target_yaw_rps": float(command[2]),
                    "actual_yaw_rps": float(angular_velocity[2]), "base_height_m": float(data.qpos[2]),
                    "base_roll_rad": roll, "base_pitch_rad": pitch,
                    "left_foot_contact": left_contact, "right_foot_contact": right_contact,
                    "halo_rhythm_x": float(halo.rhythm.x), "halo_rhythm_y": float(halo.rhythm.y),
                    "halo_frequency_hz": float(halo.rhythm.frequency),
                    "torque_rms_nm": float(np.sqrt(np.mean(np.square(effort)))),
                }
                for index, name in enumerate(JOINT_KEYS):
                    row[f"{name}_reference_rad"] = float(reference[index])
                    row[f"{name}_actual_rad"] = float(actual[index])
                rows.append(row)
                elapsed_steps += 1
        return rows


def plot_results(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    def series(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=float)

    time = series("time_s")
    target_vx = series("target_vx_mps")
    target_yaw = series("target_yaw_rps")
    segments = series("segment").astype(int)
    figure = plt.figure(figsize=(21, 18))
    grid = figure.add_gridspec(4, 2, height_ratios=(1.0, 1.0, 1.0, 2.0), hspace=0.42, wspace=0.18)
    ax_vx = figure.add_subplot(grid[0, 0])
    ax_vy = figure.add_subplot(grid[0, 1])
    ax_yaw = figure.add_subplot(grid[1, 0])
    ax_joints = figure.add_subplot(grid[1, 1])
    ax_frequency = figure.add_subplot(grid[2, :])
    ax_contacts = figure.add_subplot(grid[3, :])

    ax_vx.plot(time, target_vx, label="target")
    ax_vx.plot(time, series("actual_vx_mps"), label="actual")
    ax_vx.set_title("Body-frame x velocity")
    ax_vx.set_ylabel("m/s")
    ax_vx.legend(loc="upper right")

    ax_vy.plot(time, np.zeros_like(time), label="target")
    ax_vy.plot(time, series("actual_vy_mps"), label="actual")
    ax_vy.set_title("Body-frame y velocity")
    ax_vy.set_ylabel("m/s")
    ax_vy.legend(loc="upper right")

    ax_yaw.plot(time, series("target_yaw_rps"), label="target")
    ax_yaw.plot(time, series("actual_yaw_rps"), label="actual")
    ax_yaw.set_title("Body yaw rate")
    ax_yaw.set_ylabel("rad/s")
    ax_yaw.legend(loc="upper right")

    for name, label in zip(
        ("left_hip", "left_knee", "right_hip", "right_knee"),
        ("left_hip_pitch", "left_knee_pitch", "right_hip_pitch", "right_knee_pitch"),
    ):
        ax_joints.plot(time, series(f"{name}_actual_rad"), label=label)
    ax_joints.set_title("Hip/knee pitch joint positions")
    ax_joints.set_ylabel("rad")
    ax_joints.legend(loc="upper right")

    ax_frequency.plot(
        time, series("halo_frequency_hz"), color="tab:green", linewidth=1.8, label="filtered step frequency"
    )
    ax_frequency.set_title("Detected step frequency by target x-velocity segment")
    ax_frequency.set_ylabel("Hz")
    ax_frequency.legend(loc="upper right")

    segment_ids = np.unique(segments)
    for order, segment_id in enumerate(segment_ids):
        mask = segments == segment_id
        start, end = time[mask][0], time[mask][-1]
        if order % 2 == 0:
            ax_frequency.axvspan(start, end, color="#DCE5F2", alpha=0.38, zorder=0)
        ax_frequency.axvline(start, color="#AAB3BF", linewidth=0.7, alpha=0.65)
        ax_frequency.text(
            0.5 * (start + end), 0.93,
            f"vx={target_vx[mask][0]:+.1f}, yaw={target_yaw[mask][0]:+.1f}",
            transform=ax_frequency.get_xaxis_transform(), ha="center", va="top", color="#4F5965",
        )
    ax_frequency.axvline(time[-1], color="#AAB3BF", linewidth=0.7, alpha=0.65)

    left_contact = series("left_foot_contact") > 0.5
    right_contact = series("right_foot_contact") > 0.5
    ax_contacts.fill_between(time, 1.15, 1.85, where=left_contact, step="post", color="tab:blue")
    ax_contacts.fill_between(time, 0.15, 0.85, where=right_contact, step="post", color="tab:orange")
    ax_contacts.set_ylim(0.0, 2.0)
    ax_contacts.set_yticks((1.5, 0.5), ("Left foot", "Right foot"))
    ax_contacts.set_title("Foot ground contact")

    for axis in (ax_vx, ax_vy, ax_yaw, ax_joints, ax_frequency, ax_contacts):
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
        axis.set_xlim(time[0], time[-1])

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "gait_evaluation.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def summarize(rows: list[dict[str, float | int]]) -> dict[str, object]:
    def values(name: str, selected: list[dict[str, float | int]] = rows) -> np.ndarray:
        return np.asarray([row[name] for row in selected], dtype=float)

    def rms(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(value))))

    summary: dict[str, object] = {
        "samples": len(rows), "duration_s": float(rows[-1]["time_s"]),
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
        "joint_tracking_rmse_rad": {
            name: rms(values(f"{name}_actual_rad") - values(f"{name}_reference_rad")) for name in JOINT_KEYS
        },
    }
    summary["segments"] = []
    for segment in sorted({int(row["segment"]) for row in rows}):
        selected = [row for row in rows if row["segment"] == segment]
        # Ignore the first 0.5 s after each command change for tracking metrics.
        settled = selected[min(25, len(selected) - 1):]
        target_yaw = float(selected[0]["target_yaw_rps"])
        summary["segments"].append({
            "segment": segment,
            "samples": len(selected),
            "target_vx_mps": float(selected[0]["target_vx_mps"]),
            "actual_vx_mean_mps": float(np.mean(values("actual_vx_mps", settled))),
            "vx_mae_mps": float(np.mean(np.abs(values("actual_vx_mps", settled) - values("target_vx_mps", settled)))),
            "target_yaw_rps": target_yaw,
            "actual_yaw_mean_rps": float(np.mean(values("actual_yaw_rps", settled))),
            "yaw_mae_rps": float(np.mean(np.abs(values("actual_yaw_rps", settled) - target_yaw))),
            "yaw_sign_accuracy": (
                float(np.mean(np.sign(values("actual_yaw_rps", settled)) == np.sign(target_yaw)))
                if abs(target_yaw) > 0.05 else None
            ),
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
        writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
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
