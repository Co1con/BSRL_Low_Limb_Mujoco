"""Run the exported 29-DoF G1 CPG policy in MuJoCo."""

from __future__ import annotations

import argparse
import queue
import shutil
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import glfw
import numpy as np
import onnxruntime as ort
import yaml

from LowLimbCPG import LowLimbCPG

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = Path(__file__).with_suffix(".yaml")

DEFAULT_RENDER_FPS = 120.0
COMMAND_PRINT_INTERVAL_S = 0.5
KEY_CODES = {
    "UP": glfw.KEY_UP, "ARROWUP": glfw.KEY_UP, "DOWN": glfw.KEY_DOWN, "ARROWDOWN": glfw.KEY_DOWN,
    "LEFT": glfw.KEY_LEFT, "ARROWLEFT": glfw.KEY_LEFT, "RIGHT": glfw.KEY_RIGHT, "ARROWRIGHT": glfw.KEY_RIGHT,
}


def normalize_key(key: object) -> str:
    key_text = "" if key is None else str(key).strip()
    return "" if key_text.lower() in {"", "none", "off", "disable", "disabled"} else key_text


def key_code(key: str) -> int | None:
    key = normalize_key(key)
    if not key:
        return None
    code = KEY_CODES.get(key.upper())
    if code is None:
        if len(key) != 1:
            raise ValueError(f"Unsupported key '{key}'")
        code = ord(key.upper())
    return code


def root_velocity_body(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    rot = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot, data.qpos[3:7])
    return rot.reshape(3, 3).T @ data.qvel[0:3], data.qvel[3:6].copy()


def base_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


CPG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
)


def cpg_reference_in_policy_order(cpg: LowLimbCPG) -> np.ndarray:
    """与训练侧 observations._cpg_angles_to_robot_order 保持同一关节极性。"""
    angles = cpg.angles
    return np.array(
        [-angles["left_hip"], angles["left_knee"],
         -angles["right_hip"], angles["right_knee"]],
        dtype=np.float64,
    )


@dataclass
class JointAngleTrace:
    """暂存控制采样时刻的四关节实际角、CPG 参考角和策略目标角。"""

    time: list[float]
    actual: list[np.ndarray]
    reference: list[np.ndarray]
    policy_target: list[np.ndarray]
    command: list[np.ndarray]
    body_velocity: list[np.ndarray]
    yaw_rate: list[float]
    trunk_height: list[float]
    base_roll: list[float]
    base_pitch: list[float]
    lateral_position: list[float]

    @classmethod
    def empty(cls) -> "JointAngleTrace":
        return cls([], [], [], [], [], [], [], [], [], [], [])

    def append(self, sample_time: float, actual: np.ndarray,
               reference: np.ndarray, policy_target: np.ndarray,
               command: np.ndarray, body_velocity: np.ndarray,
               yaw_rate: float, trunk_height: float, base_roll: float,
               base_pitch: float, lateral_position: float) -> None:
        self.time.append(sample_time)
        self.actual.append(actual.copy())
        self.reference.append(reference.copy())
        self.policy_target.append(policy_target.copy())
        self.command.append(command.copy())
        self.body_velocity.append(body_velocity.copy())
        self.yaw_rate.append(yaw_rate)
        self.trunk_height.append(trunk_height)
        self.base_roll.append(base_roll)
        self.base_pitch.append(base_pitch)
        self.lateral_position.append(lateral_position)


def plot_joint_angles(trace: JointAngleTrace, target_height: float,
                      output_dir: Path | None = None, show: bool = True) -> None:
    if not trace.time:
        print("No joint-angle samples to plot.")
        return

    import matplotlib.pyplot as plt

    sample_time = np.asarray(trace.time)
    actual = np.stack(trace.actual)
    reference = np.stack(trace.reference)
    policy_target = np.stack(trace.policy_target)
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
    for joint_index, (axis, name) in enumerate(zip(axes, CPG_JOINT_NAMES)):
        axis.plot(sample_time, actual[:, joint_index], label="Actual")
        axis.plot(sample_time, reference[:, joint_index], label="CPG reference")
        # axis.plot(sample_time, policy_target[:, joint_index], label="Policy target", alpha=0.7)
        axis.set_ylabel(f"{name}\n[rad]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Simulation time [s]")
    figure.tight_layout()

    command = np.stack(trace.command)
    body_velocity = np.stack(trace.body_velocity)
    motion_figure, motion_axes = plt.subplots(2, 2, sharex=True, figsize=(14, 8))
    motion_axes[0, 0].plot(sample_time, command[:, 0], label="Target")
    motion_axes[0, 0].plot(sample_time, body_velocity[:, 0], label="Actual")
    motion_axes[0, 0].set_title("Body-frame forward velocity")
    motion_axes[0, 0].set_ylabel("m/s")

    motion_axes[0, 1].plot(sample_time, command[:, 1], label="Target")
    motion_axes[0, 1].plot(sample_time, body_velocity[:, 1], label="Actual")
    motion_axes[0, 1].set_title("Body-frame lateral velocity")
    motion_axes[0, 1].set_ylabel("m/s")

    motion_axes[1, 0].plot(sample_time, command[:, 2], label="Target")
    motion_axes[1, 0].plot(sample_time, trace.yaw_rate, label="Actual")
    motion_axes[1, 0].set_title("Body yaw rate")
    motion_axes[1, 0].set_ylabel("rad/s")

    motion_axes[1, 1].axhline(target_height, color="tab:blue", label="Initial height")
    motion_axes[1, 1].plot(sample_time, trace.trunk_height, color="tab:orange", label="Actual")
    motion_axes[1, 1].set_title("Trunk height above ground (pelvis)")
    motion_axes[1, 1].set_ylabel("m")

    for axis in motion_axes.flat:
        axis.set_xlabel("Simulation time [s]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    motion_figure.tight_layout()

    balance_figure, balance_axes = plt.subplots(3, 1, sharex=True, figsize=(14, 9))
    balance_axes[0].plot(sample_time, np.degrees(trace.base_roll))
    balance_axes[0].set_ylabel("Base roll [deg]")
    balance_axes[1].plot(sample_time, np.degrees(trace.base_pitch))
    balance_axes[1].set_ylabel("Base pitch [deg]")
    balance_axes[2].plot(sample_time, trace.lateral_position)
    balance_axes[2].set_ylabel("Lateral position [m]")
    balance_axes[2].set_xlabel("Simulation time [s]")
    for axis in balance_axes:
        axis.grid(alpha=0.25)
    balance_figure.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / "joint_angles.png", dpi=150)
        motion_figure.savefig(output_dir / "motion.png", dpi=150)
        balance_figure.savefig(output_dir / "balance.png", dpi=150)
        print(f"Saved plots to {output_dir}")
    if show:
        plt.show()
    else:
        plt.close(figure)
        plt.close(motion_figure)
        plt.close(balance_figure)


class CommandController:
    def __init__(self, config: dict) -> None:
        control_config = config.get("command_control", {})
        self.debounce_s = float(control_config.get("debounce_s", 0.02))
        self.vx_levels = np.asarray(control_config.get("lin_vel_x_levels", [0.0, float(config["walk_vx"])]), dtype=np.float32)
        self.yaw_rate_step = float(control_config.get("yaw_rate_step", 0.1))
        self.yaw_rate_limit = float(control_config.get("yaw_rate_limit", 0.5))
        self.keys = {
            "vx_up": normalize_key(control_config.get("increase_vx_key", "UP")),
            "vx_down": normalize_key(control_config.get("decrease_vx_key", "DOWN")),
            "yaw_left": normalize_key(control_config.get("yaw_left_key", "LEFT")),
            "yaw_right": normalize_key(control_config.get("yaw_right_key", "RIGHT")),
            "stop": normalize_key(control_config.get("stop_key", "U")),
        }
        self.last_trigger_time = {name: -float("inf") for name in self.keys}
        self.vx_index = 0
        self.vx_value = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

    @property
    def vx(self) -> float:
        return self.vx_value

    def reset(self, command: np.ndarray, config: dict) -> None:
        command[:] = np.asarray(config["cmd_init"], dtype=np.float32)
        self.vx_index = int(np.argmin(np.abs(self.vx_levels - command[0])))
        self.vx_value = float(command[0])
        self.vy = float(command[1])
        self.yaw_rate = float(np.clip(command[2], -self.yaw_rate_limit, self.yaw_rate_limit))
        command[2] = self.yaw_rate
        for name in self.last_trigger_time:
            self.last_trigger_time[name] = -float("inf")

    def handle_key(self, pressed_key: int, now: float) -> bool:
        action = next((name for name, key in self.keys.items() if key_code(key) == pressed_key), None)
        if action is None or now - self.last_trigger_time[action] < self.debounce_s:
            return False
        self.last_trigger_time[action] = now
        if action == "stop":
            self.vx_index = 0
            self.vx_value = 0.0
            self.yaw_rate = 0.0
        elif action == "vx_up":
            self.vx_index = min(self.vx_index + 1, self.vx_levels.size - 1)
            self.vx_value = float(self.vx_levels[self.vx_index])
        elif action == "vx_down":
            self.vx_index = max(self.vx_index - 1, 0)
            self.vx_value = float(self.vx_levels[self.vx_index])
        elif action == "yaw_left":
            self.yaw_rate = min(self.yaw_rate + self.yaw_rate_step, self.yaw_rate_limit)
        elif action == "yaw_right":
            self.yaw_rate = max(self.yaw_rate - self.yaw_rate_step, -self.yaw_rate_limit)
        return True

    def update(self, command: np.ndarray) -> None:
        command[0] = self.vx_value
        command[1] = self.vy
        command[2] = self.yaw_rate


def joint_gains(name: str) -> tuple[float, float, float]:
    """(kp, kd, velocity limit) from the exported Isaac Lab actuator settings."""
    if "hip_roll" in name:
        return 100.0, 2.0, 20.0
    if "hip_" in name:
        return 100.0, 2.0, 32.0
    if "knee" in name:
        return 150.0, 4.0, 20.0
    if "ankle" in name:
        return 40.0, 2.0, 37.0
    if name == "waist_yaw_joint":
        return 200.0, 5.0, 32.0
    if name.startswith("waist_"):
        return 40.0, 5.0, 37.0
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return 40.0, 1.0, 22.0
    if "shoulder" in name or "elbow" in name or "wrist_roll" in name:
        return 40.0, 1.0, 37.0
    raise ValueError(f"No PD gains for {name}")


def joint_effort_limit(name: str) -> float:
    """Effort limits from the policy's saved Isaac Lab env.yaml."""
    if "hip_roll" in name or "knee" in name:
        return 139.0
    if "hip_" in name or name == "waist_yaw_joint":
        return 88.0
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return 5.0
    if name.startswith("waist_"):
        return 25.0
    if "ankle" in name or "shoulder" in name or "elbow" in name or "wrist_roll" in name:
        return 25.0
    raise ValueError(f"No effort limit for {name}")


def joint_binding(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(names) != 29 or len(set(names)) != 29 or (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError(f"Expected a 29-DoF free-base G1; got nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
                      for i in range(model.nu)]
    if set(actuator_names) != set(names):
        raise ValueError(f"Policy and MuJoCo joints differ: {set(names) ^ set(actuator_names)}")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qpos_ids = np.asarray([model.jnt_qposadr[i] for i in joint_ids], dtype=np.int32)
    qvel_ids = np.asarray([model.jnt_dofadr[i] for i in joint_ids], dtype=np.int32)
    motor_ids = np.asarray([actuator_names.index(name) for name in names], dtype=np.int32)
    return qpos_ids, qvel_ids, motor_ids


def observation(data: mujoco.MjData, qpos_ids: np.ndarray, qvel_ids: np.ndarray,
                default_q: np.ndarray, effort: np.ndarray, last_action: np.ndarray,
                command: np.ndarray, cpg: LowLimbCPG, cfg: dict) -> np.ndarray:
    rot = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot, data.qpos[3:7])
    gravity = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    obs = np.concatenate((
        data.qvel[3:6] * cfg["ang_vel_scale"], gravity, command,
        data.qpos[qpos_ids] - default_q,
        data.qvel[qvel_ids] * cfg["joint_vel_scale"],
        effort * cfg["joint_effort_scale"], last_action,
        [cpg.rhythm.x, cpg.rhythm.y],
    )).astype(np.float32)
    if obs.shape != (127,) or not np.isfinite(obs).all():
        raise ValueError("Expected 127 finite G1 observations")
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--config", type=Path, help="Configuration YAML (alternative to positional path).")
    parser.add_argument("--stand-only", action="store_true", help="Hold the default pose without policy.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer.")
    parser.add_argument("--check", action="store_true", help="Validate model, policy, and one inference step.")
    parser.add_argument("--duration", type=float, help="Simulation seconds; 0 keeps the viewer open.")
    parser.add_argument("--vx", type=float, help="Initial forward command in m/s.")
    parser.add_argument("--vy", type=float, help="Initial lateral command in m/s.")
    parser.add_argument("--yaw", type=float, help="Initial yaw command in rad/s.")
    parser.add_argument("--seed", type=int, help="CPG random seed.")
    parser.add_argument("--render-fps", type=float, help="Viewer frames per second.")
    parser.add_argument("--no-plot", action="store_true", help="Skip trajectory plots when the run ends.")
    parser.add_argument("--plot-dir", type=Path, help="Also save trajectory plots to this directory.")
    parser.add_argument("--diagnostics", action="store_true", help="Print frame timing with velocity reports.")
    args = parser.parse_args()

    cfg_path = args.config or args.config_file
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    np.random.seed(int(cfg.get("seed", 0) if args.seed is None else args.seed))
    initial_command = np.asarray(cfg["cmd_init"], dtype=np.float32)
    for index, override in enumerate((args.vx, args.vy, args.yaw)):
        if override is not None:
            initial_command[index] = override
    if initial_command.shape != (3,):
        raise ValueError("cmd_init must be [vx, vy, yaw_rate]")
    cfg["cmd_init"] = initial_command.tolist()
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    simulation_duration = float(cfg["simulation_duration"] if args.duration is None else args.duration)
    render_fps = float(cfg.get("render_fps", DEFAULT_RENDER_FPS) if args.render_fps is None else args.render_fps)
    if simulation_dt <= 0.0 or control_decimation <= 0 or render_fps <= 0.0:
        raise ValueError("simulation_dt, control_decimation, and render_fps must be positive")
    if args.headless and simulation_duration <= 0.0 and not args.check:
        raise ValueError("Headless rollout requires positive --duration")
    render_period = 1.0 / render_fps
    print_interval_s = float(cfg.get("print_interval_s", COMMAND_PRINT_INTERVAL_S))
    max_catchup_s = float(cfg.get("max_catchup_s", 0.05))
    if print_interval_s <= 0.0 or max_catchup_s <= 0.0:
        raise ValueError("print_interval_s and max_catchup_s must be positive")
    startup_pause_s = float(cfg.get("startup_pause_s", 0.0))
    policy_enabled = bool(cfg.get("policy_enabled", True)) and not args.stand_only

    names = list(cfg["policy_joint_names"])
    default_q = np.asarray([cfg["default_joint_angles"].get(name, 0.0) for name in names], dtype=np.float64)
    kp, kd, velocity_limits = np.asarray([joint_gains(name) for name in names], dtype=np.float64).T
    xml_path = Path(cfg["xml_path"].replace("{PROJECT_ROOT}", str(PROJECT_ROOT))).resolve()
    policy_path = Path(cfg["policy_path"].replace("{PROJECT_ROOT}", str(PROJECT_ROOT))).resolve()
    trace = JointAngleTrace.empty()

    # Stage only the G1 model so MuJoCo can load meshes from an ASCII Windows path.
    with tempfile.TemporaryDirectory(prefix="g1_mujoco_") as temp_root:
        staged_assets = Path(temp_root) / "G1"
        shutil.copytree(xml_path.parent, staged_assets)
        model = mujoco.MjModel.from_xml_path(str(staged_assets / xml_path.name))
        model.opt.timestep = simulation_dt
        data = mujoco.MjData(model)
        qpos_ids, qvel_ids, motor_ids = joint_binding(model, names)
        model.dof_armature[qvel_ids] = 0.01
        effort_limits = np.asarray([joint_effort_limit(name) for name in names])
        model.actuator_ctrlrange[motor_ids, 0] = -effort_limits
        model.actuator_ctrlrange[motor_ids, 1] = effort_limits
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        model.jnt_actfrcrange[joint_ids, 0] = -effort_limits
        model.jnt_actfrcrange[joint_ids, 1] = effort_limits
        cpg_joint_ids = np.asarray([names.index(name) for name in CPG_JOINT_NAMES], dtype=np.int32)
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id < 0:
            raise ValueError("G1 pelvis body not found")

        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape != [1, 127] or outputs[0].shape != [1, 29]:
            raise ValueError("Expected ONNX [1, 127] -> [1, 29]")
        input_name, output_name = inputs[0].name, outputs[0].name
        cpg = LowLimbCPG()
        command_controller = CommandController(cfg)
        command = np.asarray(cfg["cmd_init"], dtype=np.float32)
        last_action = np.zeros(29, dtype=np.float32)
        effort = np.zeros(29, dtype=np.float64)
        target_q = default_q.copy()

        def update_policy() -> None:
            cpg.step(float(command[0]), simulation_dt * control_decimation)
            obs = observation(data, qpos_ids, qvel_ids, default_q, effort, last_action, command, cpg, cfg)
            action = session.run([output_name], {input_name: obs[None, :]})[0].reshape(-1)
            if action.shape != (29,) or not np.isfinite(action).all():
                raise ValueError("Expected 29 finite policy actions")
            last_action[:] = np.clip(action, -float(cfg["clip_actions"]), float(cfg["clip_actions"]))
            target_q[:] = default_q + float(cfg["action_scale"]) * last_action

        def reset_runtime() -> None:
            data.qpos[:] = 0.0
            data.qpos[2] = float(cfg["root_height"])
            data.qpos[3] = 1.0
            data.qpos[qpos_ids] = default_q
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            cpg.reset()
            last_action.fill(0.0)
            effort.fill(0.0)
            target_q[:] = default_q
            command_controller.reset(command, cfg)
            mujoco.mj_forward(model, data)
            if policy_enabled:
                update_policy()

        def step() -> None:
            q, dq = data.qpos[qpos_ids], data.qvel[qvel_ids]
            torque = kp * (target_q - q) - kd * dq
            torque = np.clip(torque, model.actuator_ctrlrange[motor_ids, 0], model.actuator_ctrlrange[motor_ids, 1])
            torque = np.where((dq >= velocity_limits) & (torque > 0), 0.0, torque)
            torque = np.where((dq <= -velocity_limits) & (torque < 0), 0.0, torque)
            data.ctrl[motor_ids] = torque
            effort[:] = torque
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"Non-finite simulation state at t={data.time:.3f}s")
            if args.headless and policy_enabled and data.qpos[2] < float(cfg["fall_height"]):
                raise RuntimeError(f"G1 fell at t={data.time:.2f}s: pelvis height={data.qpos[2]:.3f}m")

        reset_runtime()
        print(f"G1 ready: nq={model.nq}, nv={model.nv}, nu={model.nu}, ONNX 127 -> 29")
        if args.check:
            return

        counter = 0
        wall_start_time = time.perf_counter()
        real_wall_start_time = wall_start_time
        sim_start_time = data.time
        next_frame_time = wall_start_time
        last_data_time = data.time
        next_command_print_time = data.time
        key_events: queue.SimpleQueue[tuple[int, float]] = queue.SimpleQueue()
        reset_code = key_code(cfg.get("reset_key", "R"))
        start_x = float(data.qpos[0])
        viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(
            model, data, key_callback=lambda key: key_events.put((key, time.perf_counter()))
        )
        with viewer_context as viewer:
            if viewer is not None:
                viewer.opt.geomgroup[3] = 0
                camera_cfg = cfg.get("viewer", {}).get("follow_camera", {})
                if camera_cfg.get("enabled", True):
                    body_name = str(camera_cfg.get("body", "pelvis"))
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    if body_id < 0:
                        raise ValueError(f"Camera body not found: {body_name}")
                    with viewer.lock():
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.trackbodyid = body_id
                        viewer.cam.distance = float(camera_cfg.get("distance", 3.0))
                        viewer.cam.azimuth = float(camera_cfg.get("azimuth", 140.0))
                        viewer.cam.elevation = float(camera_cfg.get("elevation", -20.0))
                viewer.sync()
                wall_start_time = time.perf_counter()
                real_wall_start_time = wall_start_time
                next_frame_time = wall_start_time
            if viewer is not None and startup_pause_s > 0.0:
                time.sleep(startup_pause_s)
                wall_start_time = time.perf_counter()
                real_wall_start_time = wall_start_time
                next_frame_time = wall_start_time

            while (viewer is None or viewer.is_running()) and (simulation_duration <= 0.0 or data.time - sim_start_time < simulation_duration):
                script_reset_requested = False
                while not key_events.empty():
                    pressed_key, pressed_at = key_events.get_nowait()
                    if pressed_key == reset_code:
                        script_reset_requested = True
                    elif command_controller.handle_key(pressed_key, pressed_at):
                        command_controller.update(command)
                        print(f"command: vx={command[0]:.2f} m/s, yaw={command[2]:+.2f} rad/s")
                viewer_reset_requested = viewer is not None and data.time + simulation_dt < last_data_time
                if viewer_reset_requested or script_reset_requested:
                    reset_runtime()
                    counter = 0
                    wall_start_time = time.perf_counter()
                    real_wall_start_time = wall_start_time
                    sim_start_time = data.time
                    next_frame_time = wall_start_time
                    last_data_time = data.time
                    next_command_print_time = data.time
                    start_x = float(data.qpos[0])
                    trace = JointAngleTrace.empty()
                    if viewer is not None:
                        viewer.sync()
                    continue

                if viewer is None:
                    target_sim_time = data.time + 0.05
                else:
                    now = time.perf_counter()
                    sim_elapsed = data.time - sim_start_time
                    if now - wall_start_time - sim_elapsed > max_catchup_s:
                        wall_start_time = now - sim_elapsed - max_catchup_s
                    target_sim_time = sim_start_time + now - wall_start_time
                while data.time < target_sim_time and (simulation_duration <= 0.0 or data.time - sim_start_time < simulation_duration):
                    step()
                    counter += 1
                    if counter % control_decimation == 0:
                        command_controller.update(command)
                        if policy_enabled:
                            update_policy()
                        else:
                            last_action.fill(0.0)
                            target_q[:] = default_q
                        linear_velocity, angular_velocity = root_velocity_body(data)
                        roll, pitch = base_roll_pitch(data.qpos[3:7])
                        trace.append(
                            data.time - sim_start_time,
                            data.qpos[qpos_ids[cpg_joint_ids]],
                            cpg_reference_in_policy_order(cpg),
                            target_q[cpg_joint_ids],
                            command,
                            linear_velocity,
                            float(angular_velocity[2]),
                            float(data.xpos[pelvis_id, 2]),
                            roll,
                            pitch,
                            float(data.qpos[1]),
                        )
                    if data.time >= next_command_print_time:
                        linear_velocity, angular_velocity = root_velocity_body(data)
                        print("velocity target/actual: "
                              f"vx={command[0]: .3f}/{linear_velocity[0]: .3f} m/s, "
                              f"vy={command[1]: .3f}/{linear_velocity[1]: .3f} m/s, "
                              f"yaw={command[2]: .3f}/{angular_velocity[2]: .3f} rad/s")
                        next_command_print_time += print_interval_s

                pre_sync_time = data.time
                sync_start_time = time.perf_counter()
                if viewer is not None:
                    viewer.sync()
                sync_elapsed = time.perf_counter() - sync_start_time
                if viewer is not None and data.time + simulation_dt < pre_sync_time:
                    reset_runtime()
                    counter = 0
                    wall_start_time = time.perf_counter()
                    real_wall_start_time = wall_start_time
                    sim_start_time = data.time
                    next_frame_time = wall_start_time
                    last_data_time = data.time
                    next_command_print_time = data.time
                    start_x = float(data.qpos[0])
                    trace = JointAngleTrace.empty()
                    viewer.sync()
                    continue
                if viewer is not None:
                    if args.diagnostics and data.time - sim_start_time >= 1.0 and int(data.time - sim_start_time) != int(last_data_time - sim_start_time):
                        print(f"viewer sync={sync_elapsed * 1000:.1f} ms, "
                              f"sim lag={max(0.0, time.perf_counter() - wall_start_time - (data.time - sim_start_time)) * 1000:.1f} ms, "
                              f"sim rate={(data.time - sim_start_time) / max(time.perf_counter() - real_wall_start_time, 1e-6):.2f}x")
                    next_frame_time += render_period
                    sleep_time = next_frame_time - time.perf_counter()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    else:
                        next_frame_time = time.perf_counter()
                last_data_time = data.time

        print(f"Completed {data.time - sim_start_time:.2f}s; base height={data.qpos[2]:.3f}m; "
              f"forward distance={data.qpos[0] - start_x:.3f}m")

    if args.plot_dir is not None or (not args.headless and not args.no_plot):
        plot_joint_angles(trace, float(cfg["root_height"]), args.plot_dir,
                          show=not args.headless and not args.no_plot)


if __name__ == "__main__":
    main()
