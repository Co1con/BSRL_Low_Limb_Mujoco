import argparse
import queue
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import glfw
import numpy as np
import onnxruntime as ort
import yaml

from LowLimbCPG import LowLimbCPG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "deploy" / "BSRL" / "bsrl_cpg_deploy.yaml"
DEFAULT_RENDER_FPS = 30.0
COMMAND_PRINT_INTERVAL_S = 0.5
KEY_CODES = {
    "UP": glfw.KEY_UP,
    "ARROWUP": glfw.KEY_UP,
    "DOWN": glfw.KEY_DOWN,
    "ARROWDOWN": glfw.KEY_DOWN,
    "LEFT": glfw.KEY_LEFT,
    "ARROWLEFT": glfw.KEY_LEFT,
    "RIGHT": glfw.KEY_RIGHT,
    "ARROWRIGHT": glfw.KEY_RIGHT,
}


def resolve_path(path: str) -> str:
    return path.replace("{PROJECT_ROOT}", str(PROJECT_ROOT))


def stage_assets_to_ascii_path(xml_path: str) -> tuple[tempfile.TemporaryDirectory, str]:
    """Copy MuJoCo assets to an ASCII temp path for Windows loaders."""
    source_xml_path = Path(xml_path).resolve()
    if not source_xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {source_xml_path}")
    assets_dir = source_xml_path.parents[1]
    relative_xml_path = source_xml_path.relative_to(assets_dir)
    staged_root = tempfile.TemporaryDirectory(prefix="bsrl_mujoco_")
    staged_assets_dir = Path(staged_root.name) / assets_dir.name
    try:
        shutil.copytree(assets_dir, staged_assets_dir)
    except Exception:
        staged_root.cleanup()
        raise
    return staged_root, str(staged_assets_dir / relative_xml_path)


def key_code(key: str) -> int | None:
    key = normalize_key(key)
    if not key:
        return None

    code = KEY_CODES.get(key.upper())
    if code is None:
        if len(key) != 1:
            raise ValueError(f"Unsupported key '{key}'. Use one character or UP/DOWN/LEFT/RIGHT.")
        code = ord(key.upper())
    return code


def model_body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    return body_id


def normalize_key(key: object) -> str:
    key_text = "" if key is None else str(key).strip()
    if key_text.lower() in {"", "none", "off", "disable", "disabled"}:
        return ""
    return key_text


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(quat_conj(q), vq), q)[1:]


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))


def root_velocity_body(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """返回根部机体系线速度和角速度，仅用于运行日志，不进入策略网络。"""
    quat = data.qpos[3:7]
    lin_vel_b = quat_rotate_inverse(quat, data.qvel[0:3])
    ang_vel_b = data.qvel[3:6].copy()
    return lin_vel_b, ang_vel_b


def base_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def clip_by_motor_limits(
    tau: np.ndarray,
    dq: np.ndarray,
    ctrlrange: np.ndarray,
    peak_powers: np.ndarray | None,
    velocity_limits: np.ndarray | None,
) -> np.ndarray:
    tau = np.clip(tau, ctrlrange[:, 0], ctrlrange[:, 1])

    if peak_powers is not None:
        power_torque_limit = peak_powers / np.maximum(np.abs(dq), 1e-6)
        motor_torque_limit = np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1]))
        torque_limit = np.minimum(motor_torque_limit, power_torque_limit)
        tau = np.clip(tau, -torque_limit, torque_limit)

    if velocity_limits is not None:
        tau = np.where((dq >= velocity_limits) & (tau > 0.0), 0.0, tau)
        tau = np.where((dq <= -velocity_limits) & (tau < 0.0), 0.0, tau)

    return tau


@dataclass(frozen=True)
class JointBinding:
    qpos_addr: np.ndarray
    qvel_addr: np.ndarray
    actuator_ids: np.ndarray
    ctrlrange: np.ndarray


@dataclass
class RuntimeState:
    action: np.ndarray
    target_q: np.ndarray
    target_dq: np.ndarray
    joint_effort: np.ndarray
    command: np.ndarray


CPG_JOINT_NAMES = (
    "joint_left_hip_pitch",
    "joint_left_knee_pitch",
    "joint_right_hip_pitch",
    "joint_right_knee_pitch",
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
        axis.plot(sample_time, policy_target[:, joint_index], label="Policy target", alpha=0.7)
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
    motion_axes[1, 1].set_title("Trunk height above ground (base_link)")
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


def configure_follow_camera(viewer: "mujoco.viewer.Handle", model: mujoco.MjModel, config: dict) -> None:
    viewer_config = config.get("viewer", {})
    camera_config = viewer_config.get("follow_camera", {})
    if not bool(camera_config.get("enabled", True)):
        return

    body_id = model_body_id(model, str(camera_config.get("body", "base_link")))
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = body_id
        viewer.cam.distance = float(camera_config.get("distance", 3.0))
        viewer.cam.azimuth = float(camera_config.get("azimuth", 140.0))
        viewer.cam.elevation = float(camera_config.get("elevation", -20.0))


def require_model_layout(model: mujoco.MjModel, mujoco_joint_names: list[str], policy_joint_names: list[str]) -> None:
    if model.nq != 7 + len(policy_joint_names):
        raise ValueError(f"Expected nq={7 + len(policy_joint_names)}, got nq={model.nq}")
    if model.nv != 6 + len(policy_joint_names):
        raise ValueError(f"Expected nv={6 + len(policy_joint_names)}, got nv={model.nv}")
    if model.nu != len(policy_joint_names):
        raise ValueError(f"Expected nu={len(policy_joint_names)}, got nu={model.nu}")

    actuator_joints = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))

    if actuator_joints != mujoco_joint_names:
        raise ValueError(f"Actuator order mismatch:\nexpected={mujoco_joint_names}\nactual={actuator_joints}")


def bind_joints(model: mujoco.MjModel, policy_joint_names: list[str]) -> JointBinding:
    qpos_addr = []
    qvel_addr = []
    actuator_joint_to_id = {}

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_joint_to_id[joint_name] = actuator_id

    for name in policy_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {name}")
        if name not in actuator_joint_to_id:
            raise ValueError(f"Actuator not found for joint: {name}")
        qpos_addr.append(model.jnt_qposadr[joint_id])
        qvel_addr.append(model.jnt_dofadr[joint_id])

    actuator_ids = np.asarray([actuator_joint_to_id[name] for name in policy_joint_names], dtype=np.int32)
    return JointBinding(
        qpos_addr=np.asarray(qpos_addr, dtype=np.int32),
        qvel_addr=np.asarray(qvel_addr, dtype=np.int32),
        actuator_ids=actuator_ids,
        ctrlrange=model.actuator_ctrlrange[actuator_ids],
    )


def reset_to_default_pose(data: mujoco.MjData, binding: JointBinding, default_angles: np.ndarray, root_height: float) -> None:
    data.qpos[:] = 0.0
    data.qpos[2] = root_height
    data.qpos[3] = 1.0
    data.qpos[binding.qpos_addr] = default_angles
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0


def build_observation(
    data: mujoco.MjData,
    binding: JointBinding,
    state: RuntimeState,
    default_angles: np.ndarray,
    rhythm_xy: np.ndarray,
    config: dict,
) -> np.ndarray:
    quat = data.qpos[3:7]
    # MuJoCo free joint 的 qvel[3:6] 已经是机体系角速度，对齐 IsaacLab 的 base_ang_vel。
    omega = data.qvel[3:6] * float(config["ang_vel_scale"])
    gravity = projected_gravity(quat)
    cmd = state.command * np.asarray(config["cmd_scale"], dtype=np.float32)
    qj = (data.qpos[binding.qpos_addr] - default_angles) * float(config["dof_pos_scale"])
    dqj = data.qvel[binding.qvel_addr] * float(config["dof_vel_scale"])
    effort = state.joint_effort * float(config["joint_effort_scale"])

    obs = np.concatenate([omega, gravity, cmd, qj, dqj, effort, state.action, rhythm_xy]).astype(np.float32)
    if obs.shape != (int(config["num_obs"]),):
        raise ValueError(f"Observation shape mismatch: expected {(int(config['num_obs']),)}, got {obs.shape}")
    return obs


def load_policy(policy_path: str) -> tuple[ort.InferenceSession, str, str]:
    session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name


def validate_policy_shape(session: ort.InferenceSession, num_obs: int, num_actions: int) -> None:
    """Check the exported policy against the observation and action configuration."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Policy must have exactly one input and one output")
    if inputs[0].shape != [1, num_obs] or outputs[0].shape != [1, num_actions]:
        raise ValueError(
            f"Policy shape mismatch: expected [1, {num_obs}] -> [1, {num_actions}], "
            f"got {inputs[0].shape} -> {outputs[0].shape}"
        )


def run_policy(session: ort.InferenceSession, input_name: str, output_name: str, obs: np.ndarray,
               clip_actions: float | None = None) -> np.ndarray:
    if not np.isfinite(obs).all():
        raise ValueError("Policy observation contains non-finite values")
    action = session.run([output_name], {input_name: obs.reshape(1, -1)})[0].reshape(-1).astype(np.float32)
    if not np.isfinite(action).all():
        raise ValueError("Policy action contains non-finite values")
    if clip_actions is not None:
        action = np.clip(action, -clip_actions, clip_actions)
    return action


def create_runtime_state(config: dict, default_angles: np.ndarray, kds: np.ndarray) -> RuntimeState:
    num_actions = int(config["num_actions"])
    return RuntimeState(
        action=np.zeros(num_actions, dtype=np.float32),
        target_q=default_angles.copy(),
        target_dq=np.zeros_like(kds),
        joint_effort=np.zeros(num_actions, dtype=np.float64),
        command=np.asarray(config["cmd_init"], dtype=np.float32),
    )


def step_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    binding: JointBinding,
    state: RuntimeState,
    kps: np.ndarray,
    kds: np.ndarray,
    peak_powers: np.ndarray | None,
    velocity_limits: np.ndarray | None,
) -> None:
    q = data.qpos[binding.qpos_addr]
    dq = data.qvel[binding.qvel_addr]
    tau = pd_control(state.target_q, q, kps, state.target_dq, dq, kds)
    tau = clip_by_motor_limits(tau, dq, binding.ctrlrange, peak_powers, velocity_limits)
    data.ctrl[binding.actuator_ids] = tau
    state.joint_effort[:] = tau
    mujoco.mj_step(model, data)


def reset_runtime(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    binding: JointBinding,
    state: RuntimeState,
    cpg: LowLimbCPG,
    command_controller: CommandController,
    default_angles: np.ndarray,
    root_height: float,
    config: dict,
    session: ort.InferenceSession | None,
    input_name: str,
    output_name: str,
) -> None:
    # reset 必须同时清物理状态和控制器内部状态，否则旧 action 会在下一控制步把姿态拉回去。
    reset_to_default_pose(data, binding, default_angles, root_height)
    cpg.reset()
    state.action.fill(0.0)
    state.target_q[:] = default_angles
    state.target_dq.fill(0.0)
    state.joint_effort.fill(0.0)
    command_controller.reset(state.command, config)
    mujoco.mj_forward(model, data)

    if session is not None:
        rhythm_xy = np.array([cpg.rhythm.x, cpg.rhythm.y], dtype=np.float32)
        obs = build_observation(data, binding, state, default_angles, rhythm_xy, config)
        state.action[:] = run_policy(session, input_name, output_name, obs, config.get("clip_actions"))
        state.target_q[:] = state.action * float(config["action_scale"]) + default_angles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the BSRL CPG policy in MuJoCo.")
    parser.add_argument("config_file", nargs="?", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--config", type=Path, help="Configuration YAML (alternative to positional path).")
    parser.add_argument("--stand-only", action="store_true", help="Keep zero action and only test default standing pose.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or args.config_file
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    np.random.seed(int(config.get("seed", 0) if args.seed is None else args.seed))
    initial_command = np.asarray(config["cmd_init"], dtype=np.float32)
    for index, override in enumerate((args.vx, args.vy, args.yaw)):
        if override is not None:
            initial_command[index] = override
    config["cmd_init"] = initial_command.tolist()

    policy_path = resolve_path(config["policy_path"])
    xml_path = resolve_path(config["xml_path"])
    simulation_duration = float(config["simulation_duration"] if args.duration is None else args.duration)
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    startup_pause_s = float(config.get("startup_pause_s", 0.0))
    render_fps = float(config.get("render_fps", DEFAULT_RENDER_FPS) if args.render_fps is None else args.render_fps)
    if render_fps <= 0.0:
        raise ValueError("render-fps must be positive")
    render_period = 1.0 / render_fps
    print_interval_s = float(config.get("print_interval_s", COMMAND_PRINT_INTERVAL_S))
    max_catchup_s = float(config.get("max_catchup_s", 0.05))
    if print_interval_s <= 0.0 or max_catchup_s <= 0.0:
        raise ValueError("print_interval_s and max_catchup_s must be positive")
    if args.headless and simulation_duration <= 0.0 and not args.check:
        raise ValueError("Headless rollout requires positive --duration")
    root_height = float(config.get("root_height", 0.8665))

    kps = np.asarray(config["kps"], dtype=np.float64)
    kds = np.asarray(config["kds"], dtype=np.float64)
    default_angles = np.asarray(config["default_angles"], dtype=np.float64)
    peak_powers = np.asarray(config["peak_powers"], dtype=np.float64) if "peak_powers" in config else None
    velocity_limits = np.asarray(config["velocity_limits"], dtype=np.float64) if "velocity_limits" in config else None
    policy_joint_names = list(config.get("policy_joint_names", config.get("joint_names", [])))
    mujoco_joint_names = list(config.get("mujoco_joint_names", policy_joint_names))
    policy_enabled = bool(config.get("policy_enabled", True)) and not args.stand_only

    state = create_runtime_state(config, default_angles, kds)
    command_controller = CommandController(config)
    cpg = LowLimbCPG()
    trace = JointAngleTrace.empty()
    staged_assets, staged_xml_path = stage_assets_to_ascii_path(xml_path)

    try:
        model = mujoco.MjModel.from_xml_path(staged_xml_path)
        model.opt.timestep = simulation_dt
        data = mujoco.MjData(model)

        # 策略关节顺序和 MuJoCo 执行器顺序可能不同，所有读写都通过显式地址映射完成。
        require_model_layout(model, mujoco_joint_names, policy_joint_names)
        binding = bind_joints(model, policy_joint_names)
        if "joint_armature" in config:
            model.dof_armature[binding.qvel_addr] = float(config["joint_armature"])
        trunk_body_id = model_body_id(model, "base_link")
        cpg_joint_ids = np.asarray(
            [policy_joint_names.index(name) for name in CPG_JOINT_NAMES], dtype=np.int32
        )
        session = None
        input_name = ""
        output_name = ""
        if policy_enabled:
            session, input_name, output_name = load_policy(policy_path)
            validate_policy_shape(session, int(config["num_obs"]), int(config["num_actions"]))

        reset_runtime(
            model,
            data,
            binding,
            state,
            cpg,
            command_controller,
            default_angles,
            root_height,
            config,
            session,
            input_name,
            output_name,
        )
        print(f"BSRL ready: nq={model.nq}, nv={model.nv}, nu={model.nu}, "
              f"ONNX {config['num_obs']} -> {config['num_actions']}")
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
        reset_code = key_code(config.get("reset_key", "R"))
        start_x = float(data.qpos[0])

        viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(
            model, data, key_callback=lambda key: key_events.put((key, time.perf_counter()))
        )
        with viewer_context as viewer:
            if viewer is not None:
                viewer.opt.geomgroup[3] = 0
                configure_follow_camera(viewer, model, config)
                viewer.sync()
                wall_start_time = time.perf_counter()
                real_wall_start_time = wall_start_time
                next_frame_time = wall_start_time
            if startup_pause_s > 0.0 and viewer is not None:
                time.sleep(startup_pause_s)
                wall_start_time = time.perf_counter()
                real_wall_start_time = wall_start_time
                sim_start_time = data.time
                next_frame_time = wall_start_time
                next_command_print_time = data.time

            while (viewer is None or viewer.is_running()) and (simulation_duration <= 0.0 or data.time - sim_start_time < simulation_duration):
                script_reset_requested = False
                while not key_events.empty():
                    pressed_key, pressed_at = key_events.get_nowait()
                    if pressed_key == reset_code:
                        script_reset_requested = True
                    elif command_controller.handle_key(pressed_key, pressed_at):
                        command_controller.update(state.command)
                        print(f"command: vx={state.command[0]:.2f} m/s, yaw={state.command[2]:+.2f} rad/s")
                viewer_reset_requested = viewer is not None and data.time + simulation_dt < last_data_time

                if viewer_reset_requested or script_reset_requested:
                    reset_runtime(
                        model,
                        data,
                        binding,
                        state,
                        cpg,
                        command_controller,
                        default_angles,
                        root_height,
                        config,
                        session,
                        input_name,
                        output_name,
                    )
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

                # 每帧渲染前批量推进仿真，避免 Windows 1 ms sleep 精度不足造成慢动作。
                while data.time < target_sim_time and (
                    simulation_duration <= 0.0 or data.time - sim_start_time < simulation_duration
                ):
                    step_control(model, data, binding, state, kps, kds, peak_powers, velocity_limits)
                    counter += 1

                    if counter % control_decimation == 0:
                        command_controller.update(state.command)

                        cpg.step(float(state.command[0]), simulation_dt * control_decimation)
                        rhythm_xy = np.array([cpg.rhythm.x, cpg.rhythm.y], dtype=np.float32)

                        if policy_enabled:
                            obs = build_observation(data, binding, state, default_angles, rhythm_xy, config)
                            state.action[:] = run_policy(session, input_name, output_name, obs, config.get("clip_actions"))
                            state.target_q[:] = state.action * float(config["action_scale"]) + default_angles
                        else:
                            state.action.fill(0.0)
                            state.target_q[:] = default_angles

                        actual_lin_vel_b, actual_ang_vel_b = root_velocity_body(data)
                        roll, pitch = base_roll_pitch(data.qpos[3:7])
                        trace.append(
                            data.time - sim_start_time,
                            data.qpos[binding.qpos_addr[cpg_joint_ids]],
                            cpg_reference_in_policy_order(cpg),
                            state.target_q[cpg_joint_ids],
                            state.command,
                            actual_lin_vel_b,
                            float(actual_ang_vel_b[2]),
                            float(data.xpos[trunk_body_id, 2]),
                            roll,
                            pitch,
                            float(data.qpos[1]),
                        )

                    if data.time >= next_command_print_time:
                        actual_lin_vel_b, actual_ang_vel_b = root_velocity_body(data)
                        print(
                            "velocity target/actual: "
                            f"vx={state.command[0]: .3f}/{actual_lin_vel_b[0]: .3f} m/s, "
                            f"vy={state.command[1]: .3f}/{actual_lin_vel_b[1]: .3f} m/s, "
                            f"yaw={state.command[2]: .3f}/{actual_ang_vel_b[2]: .3f} rad/s"
                        )
                        next_command_print_time += print_interval_s

                pre_sync_time = data.time
                sync_start_time = time.perf_counter()
                if viewer is not None:
                    viewer.sync()
                sync_elapsed = time.perf_counter() - sync_start_time
                if viewer is not None and data.time + simulation_dt < pre_sync_time:
                    reset_runtime(
                        model,
                        data,
                        binding,
                        state,
                        cpg,
                        command_controller,
                        default_angles,
                        root_height,
                        config,
                        session,
                        input_name,
                        output_name,
                    )
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
                    if args.diagnostics and data.time - sim_start_time >= 1.0 and int((data.time - sim_start_time) / 1.0) != int((last_data_time - sim_start_time) / 1.0):
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
    finally:
        staged_assets.cleanup()

    if args.plot_dir is not None or (not args.headless and not args.no_plot):
        plot_joint_angles(trace, root_height, args.plot_dir, show=not args.headless and not args.no_plot)


if __name__ == "__main__":
    main()
