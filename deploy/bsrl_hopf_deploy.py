import argparse
import ctypes
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "deploy" / "bsrl_hopf_deploy.yaml"
DEFAULT_RENDER_FPS = 60.0


def resolve_path(path: str) -> str:
    return path.replace("{PROJECT_ROOT}", str(PROJECT_ROOT))


def stage_assets_to_ascii_path(xml_path: str) -> tuple[tempfile.TemporaryDirectory, str]:
    """Copy MuJoCo assets to an ASCII temp path for Windows loaders."""
    source_xml_path = Path(xml_path).resolve()
    assets_dir = source_xml_path.parents[2]
    relative_xml_path = source_xml_path.relative_to(assets_dir)
    staged_root = tempfile.TemporaryDirectory(prefix="bsrl_mujoco_")
    staged_assets_dir = Path(staged_root.name) / assets_dir.name
    shutil.copytree(assets_dir, staged_assets_dir)
    return staged_root, str(staged_assets_dir / relative_xml_path)


def get_key_pressed(key: str) -> bool:
    if len(key) != 1:
        raise ValueError(f"Only single-character keys are supported, got '{key}'")
    return bool(ctypes.windll.user32.GetAsyncKeyState(ord(key.upper())) & 0x8000)


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


class MasterHopf:
    def __init__(self, config: dict) -> None:
        self.mu = float(config["mu"])
        self.gamma = float(config["gamma"])
        self.stop_eps = float(config["stop_eps"])
        self.phase_eps = float(config["phase_eps"])
        self.velocity_freq_quadratic = float(config.get("velocity_freq_quadratic", 0.0))
        self.velocity_freq_linear = float(config.get("velocity_freq_linear", config.get("velocity_freq_slope", 0.0)))
        self.velocity_freq_intercept = float(config["velocity_freq_intercept"])
        self.velocity_freq_min = float(config.get("velocity_freq_min", 0.0))
        self.velocity_freq_max = float(config.get("velocity_freq_max", float("inf")))
        self.return_omega_min = 2.0 * math.pi * float(config["return_freq_min"])
        self.return_omega_max = 2.0 * math.pi * float(config["return_freq_max"])
        self.x = 0.0
        self.y = 0.0
        self.omega = 0.0
        self.last_omega = self.return_omega_max
        self.reset()

    @property
    def phase(self) -> float:
        return math.atan2(self.y, self.x) % (2.0 * math.pi)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def reset(self, phase: float = 0.0) -> None:
        radius = math.sqrt(self.mu)
        self.x = radius * math.cos(phase)
        self.y = radius * math.sin(phase)
        self.omega = 0.0
        self.last_omega = self.return_omega_max

    def velocity_to_frequency(self, vx: float) -> float:
        if abs(vx) < self.stop_eps:
            return 0.0

        abs_vx = abs(float(vx))
        freq = self.velocity_freq_quadratic * abs_vx * abs_vx + self.velocity_freq_linear * abs_vx + self.velocity_freq_intercept
        return min(max(freq, self.velocity_freq_min), self.velocity_freq_max)

    def step_velocity(self, vx: float, dt: float) -> np.ndarray:
        return self.step(2.0 * math.pi * self.velocity_to_frequency(vx), dt)

    def step(self, omega_cmd: float, dt: float) -> np.ndarray:
        phase = self.phase
        walking = abs(omega_cmd) > self.stop_eps
        near_start = phase <= self.phase_eps or phase >= 2.0 * math.pi - self.phase_eps

        if walking:
            self.omega = omega_cmd
            self.last_omega = abs(omega_cmd)
        elif near_start:
            self.reset()
            return self.xy
        else:
            self.omega = min(max(self.last_omega, self.return_omega_min), self.return_omega_max)

        r2 = self.x * self.x + self.y * self.y
        self.x += (self.gamma * (self.mu - r2) * self.x - self.omega * self.y) * dt
        self.y += (self.gamma * (self.mu - r2) * self.y + self.omega * self.x) * dt

        if not walking:
            phase = self.phase
            if phase <= self.phase_eps or phase >= 2.0 * math.pi - self.phase_eps:
                self.reset()

        return self.xy


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
    hopf_xy: np.ndarray,
    config: dict,
) -> np.ndarray:
    # MuJoCo 自由基的角速度在全局系，这里转到机体系以匹配训练观测。
    quat = data.qpos[3:7]
    omega = quat_rotate_inverse(quat, data.qvel[3:6]) * float(config["ang_vel_scale"])
    gravity = projected_gravity(quat)
    cmd = state.command * np.asarray(config["cmd_scale"], dtype=np.float32)
    qj = (data.qpos[binding.qpos_addr] - default_angles) * float(config["dof_pos_scale"])
    dqj = data.qvel[binding.qvel_addr] * float(config["dof_vel_scale"])
    effort = state.joint_effort * float(config["joint_effort_scale"])

    obs = np.concatenate([omega, gravity, cmd, qj, dqj, effort, state.action, hopf_xy]).astype(np.float32)
    if obs.shape != (int(config["num_obs"]),):
        raise ValueError(f"Observation shape mismatch: expected {(int(config['num_obs']),)}, got {obs.shape}")
    return obs


def load_policy(policy_path: str) -> tuple[ort.InferenceSession, str, str]:
    session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name


def run_policy(session: ort.InferenceSession, input_name: str, output_name: str, obs: np.ndarray) -> np.ndarray:
    return session.run([output_name], {input_name: obs.reshape(1, -1)})[0].reshape(-1).astype(np.float32)


def update_command(command: np.ndarray, config: dict) -> None:
    command[:] = np.asarray(config["cmd_init"], dtype=np.float32)
    try:
        walking = get_key_pressed(str(config["walk_key"]))
    except AttributeError:
        walking = False

    if walking:
        command[0] = float(config["walk_vx"])


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the BSRL Hopf policy in MuJoCo.")
    parser.add_argument("config_file", nargs="?", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stand-only", action="store_true", help="Keep zero action and only test default standing pose.")
    parser.add_argument("--render-fps", type=float, default=DEFAULT_RENDER_FPS, help="Viewer sync frequency.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    policy_path = resolve_path(config["policy_path"])
    xml_path = resolve_path(config["xml_path"])
    simulation_duration = float(config["simulation_duration"])
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    startup_pause_s = float(config.get("startup_pause_s", 0.0))
    render_period = 1.0 / max(args.render_fps, 1.0)

    kps = np.asarray(config["kps"], dtype=np.float64)
    kds = np.asarray(config["kds"], dtype=np.float64)
    default_angles = np.asarray(config["default_angles"], dtype=np.float64)
    peak_powers = np.asarray(config["peak_powers"], dtype=np.float64) if "peak_powers" in config else None
    velocity_limits = np.asarray(config["velocity_limits"], dtype=np.float64) if "velocity_limits" in config else None
    policy_joint_names = list(config.get("policy_joint_names", config.get("joint_names", [])))
    mujoco_joint_names = list(config.get("mujoco_joint_names", policy_joint_names))
    policy_enabled = bool(config.get("policy_enabled", True)) and not args.stand_only

    state = create_runtime_state(config, default_angles, kds)
    hopf = MasterHopf(config["hopf"])
    staged_assets, staged_xml_path = stage_assets_to_ascii_path(xml_path)

    try:
        model = mujoco.MjModel.from_xml_path(staged_xml_path)
        model.opt.timestep = simulation_dt
        data = mujoco.MjData(model)

        # 策略关节顺序和 MuJoCo 执行器顺序可能不同，所有读写都通过显式地址映射完成。
        require_model_layout(model, mujoco_joint_names, policy_joint_names)
        binding = bind_joints(model, policy_joint_names)
        reset_to_default_pose(data, binding, default_angles, float(config.get("root_height", 0.8665)))
        mujoco.mj_forward(model, data)

        session = None
        input_name = ""
        output_name = ""
        if policy_enabled:
            session, input_name, output_name = load_policy(policy_path)
            first_obs = build_observation(data, binding, state, default_angles, hopf.xy, config)
            state.action[:] = run_policy(session, input_name, output_name, first_obs)
            state.target_q[:] = state.action * float(config["action_scale"]) + default_angles

        counter = 0
        wall_start_time = time.perf_counter()
        sim_start_time = data.time
        next_render_time = wall_start_time

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.geomgroup[3] = 0
            viewer.sync()
            if startup_pause_s > 0.0:
                time.sleep(startup_pause_s)
                wall_start_time = time.perf_counter()
                sim_start_time = data.time
                next_render_time = wall_start_time

            while viewer.is_running() and (simulation_duration <= 0.0 or data.time - sim_start_time < simulation_duration):
                wall_elapsed = time.perf_counter() - wall_start_time
                sim_elapsed = data.time - sim_start_time

                if sim_elapsed > wall_elapsed + simulation_dt:
                    time.sleep(min(sim_elapsed - wall_elapsed, render_period))
                    continue

                step_control(model, data, binding, state, kps, kds, peak_powers, velocity_limits)
                counter += 1

                if counter % control_decimation == 0:
                    update_command(state.command, config)

                    if policy_enabled:
                        hopf_xy = hopf.step_velocity(float(state.command[0]), simulation_dt * control_decimation)
                        obs = build_observation(data, binding, state, default_angles, hopf_xy, config)
                        state.action[:] = run_policy(session, input_name, output_name, obs)
                        state.target_q[:] = state.action * float(config["action_scale"]) + default_angles
                    else:
                        state.action.fill(0.0)
                        state.target_q[:] = default_angles

                # 仿真按 data.time 对齐真实时间；GUI 无需 1000 Hz 刷新，低频同步即可。
                now = time.perf_counter()
                if now >= next_render_time:
                    viewer.sync()
                    next_render_time = now + render_period
    finally:
        staged_assets.cleanup()


if __name__ == "__main__":
    main()
