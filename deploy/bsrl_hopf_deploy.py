import argparse
import ctypes
import math
import shutil
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "deploy" / "bsrl_hopf_deploy.yaml"


def resolve_path(path: str) -> str:
    return path.replace("{PROJECT_ROOT}", str(PROJECT_ROOT))


def stage_assets_to_ascii_path(xml_path: str) -> tuple[tempfile.TemporaryDirectory, str]:
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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def clip_by_motor_limits(tau: np.ndarray, dq: np.ndarray, ctrlrange: np.ndarray, peak_powers: np.ndarray | None) -> np.ndarray:
    tau = np.clip(tau, ctrlrange[:, 0], ctrlrange[:, 1])
    if peak_powers is None:
        return tau

    power_torque_limit = peak_powers / np.maximum(np.abs(dq), 1e-3)
    torque_limit = np.minimum(np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1])), power_torque_limit)
    return np.clip(tau, -torque_limit, torque_limit)


class MasterHopf:
    def __init__(self, config: dict) -> None:
        self.mu = float(config["mu"])
        self.gamma = float(config["gamma"])
        self.stop_eps = float(config["stop_eps"])
        self.phase_eps = float(config["phase_eps"])
        self.velocity_freq_slope = float(config["velocity_freq_slope"])
        self.velocity_freq_intercept = float(config["velocity_freq_intercept"])
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

    @property
    def is_at_rest(self) -> bool:
        phase = self.phase
        near_start = phase <= self.phase_eps or phase >= 2.0 * math.pi - self.phase_eps
        return abs(self.omega) <= self.stop_eps and near_start

    def reset(self, phase: float = 0.0) -> None:
        radius = math.sqrt(self.mu)
        self.x = radius * math.cos(phase)
        self.y = radius * math.sin(phase)
        self.omega = 0.0
        self.last_omega = self.return_omega_max

    def velocity_to_frequency(self, vx: float) -> float:
        if abs(vx) < self.stop_eps:
            return 0.0
        return self.velocity_freq_slope * abs(vx) + self.velocity_freq_intercept

    def step_velocity(self, vx: float, dt: float) -> np.ndarray:
        omega_cmd = 2.0 * math.pi * self.velocity_to_frequency(vx)
        return self.step(omega_cmd, dt)

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
        dx = self.gamma * (self.mu - r2) * self.x - self.omega * self.y
        dy = self.gamma * (self.mu - r2) * self.y + self.omega * self.x
        self.x += dx * dt
        self.y += dy * dt

        phase = self.phase
        if not walking and (phase <= self.phase_eps or phase >= 2.0 * math.pi - self.phase_eps):
            self.reset()
        return self.xy


def check_model_order(model: mujoco.MjModel, joint_names: list[str]) -> None:
    if model.nq != 7 + len(joint_names):
        raise ValueError(f"Expected nq={7 + len(joint_names)}, got nq={model.nq}")
    if model.nv != 6 + len(joint_names):
        raise ValueError(f"Expected nv={6 + len(joint_names)}, got nv={model.nv}")
    if model.nu != len(joint_names):
        raise ValueError(f"Expected nu={len(joint_names)}, got nu={model.nu}")

    actuator_joints = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_joints.append(joint_name)

    if actuator_joints != joint_names:
        raise ValueError(f"Actuator order mismatch:\nexpected={joint_names}\nactual={actuator_joints}")


def reset_to_default_pose(data: mujoco.MjData, default_angles: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qpos[2] = 0.8665
    data.qpos[3] = 1.0
    data.qpos[7 : 7 + default_angles.size] = default_angles
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0


def build_observation(
    data: mujoco.MjData,
    command: np.ndarray,
    default_angles: np.ndarray,
    action: np.ndarray,
    joint_effort: np.ndarray,
    hopf_xy: np.ndarray,
    config: dict,
) -> np.ndarray:
    quat = data.qpos[3:7].copy()
    omega = quat_rotate_inverse(quat, data.qvel[3:6].copy()) * float(config["ang_vel_scale"])
    gravity = projected_gravity(quat)
    cmd = command * np.asarray(config["cmd_scale"], dtype=np.float32)
    qj = (data.qpos[7:] - default_angles) * float(config["dof_pos_scale"])
    dqj = data.qvel[6:] * float(config["dof_vel_scale"])
    effort = joint_effort * float(config["joint_effort_scale"])

    obs = np.concatenate([omega, gravity, cmd, qj, dqj, effort, action, hopf_xy]).astype(np.float32)
    if obs.shape != (int(config["num_obs"]),):
        raise ValueError(f"Observation shape mismatch: expected {(int(config['num_obs']),)}, got {obs.shape}")
    return obs


def print_joint_table(title: str, joint_names: list[str], target_q: np.ndarray, q: np.ndarray) -> None:
    print(title)
    print("index,name,target_q,actual_q,error")
    for i, name in enumerate(joint_names):
        print(f"{i:02d},{name},{target_q[i]: .6f},{q[i]: .6f},{target_q[i] - q[i]: .6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the BSRL Hopf policy in MuJoCo.")
    parser.add_argument("config_file", nargs="?", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stand-only", action="store_true", help="Keep zero action and only test default standing pose.")
    args = parser.parse_args()

    with args.config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    policy_path = resolve_path(config["policy_path"])
    xml_path = resolve_path(config["xml_path"])
    simulation_duration = float(config["simulation_duration"])
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    startup_pause_s = float(config.get("startup_pause_s", 0.0))

    kps = np.asarray(config["kps"], dtype=np.float64)
    kds = np.asarray(config["kds"], dtype=np.float64)
    peak_powers = np.asarray(config["peak_powers"], dtype=np.float64) if "peak_powers" in config else None
    default_angles = np.asarray(config["default_angles"], dtype=np.float64)
    joint_names = list(config["joint_names"])
    num_actions = int(config["num_actions"])
    policy_enabled = bool(config.get("policy_enabled", True)) and not args.stand_only

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_vel = np.zeros_like(kds)
    joint_effort = np.zeros(num_actions, dtype=np.float64)
    cmd = np.asarray(config["cmd_init"], dtype=np.float32)
    hopf = MasterHopf(config["hopf"])

    staged_assets, staged_xml_path = stage_assets_to_ascii_path(xml_path)
    model = mujoco.MjModel.from_xml_path(staged_xml_path)
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)
    check_model_order(model, joint_names)
    reset_to_default_pose(data, default_angles)
    mujoco.mj_forward(model, data)

    session = None
    input_name = ""
    output_name = ""
    if policy_enabled:
        session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

    print(f"xml_path={xml_path}")
    print(f"staged_xml_path={staged_xml_path}")
    print(f"policy_path={policy_path}")
    print(f"default_angles shape={default_angles.shape} values={default_angles}")
    print_joint_table("initial pose after reset_to_default_pose()", joint_names, target_dof_pos, data.qpos[7:])
    if policy_enabled:
        first_obs = build_observation(data, cmd, default_angles, action, joint_effort, hopf.xy, config)
        first_action = session.run([output_name], {input_name: first_obs.reshape(1, -1)})[0].reshape(-1)
        print(f"policy_input={input_name} shape={first_obs.reshape(1, -1).shape}")
        print(f"policy_output={output_name} shape={first_action.reshape(1, -1).shape}")
        print(f"first_action shape={first_action.shape} values={first_action}")
    else:
        print("policy is disabled; standing with zero action.")

    counter = 0
    start = time.time()
    last_print = start

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[3] = 0
        viewer.sync()
        if startup_pause_s > 0.0:
            time.sleep(startup_pause_s)

        while viewer.is_running() and (simulation_duration <= 0.0 or time.time() - start < simulation_duration):
            step_start = time.time()

            tau = pd_control(target_dof_pos, data.qpos[7:], kps, target_dof_vel, data.qvel[6:], kds)
            tau = clip_by_motor_limits(tau, data.qvel[6:], model.actuator_ctrlrange, peak_powers)
            data.ctrl[:] = tau
            joint_effort = tau.copy()
            mujoco.mj_step(model, data)

            counter += 1
            if counter % control_decimation == 0:
                try:
                    walking = get_key_pressed(str(config["walk_key"]))
                except AttributeError:
                    walking = False
                cmd[:] = np.asarray(config["cmd_init"], dtype=np.float32)
                if walking:
                    cmd[0] = float(config["walk_vx"])

                if policy_enabled:
                    hopf_xy = hopf.step_velocity(float(cmd[0]), simulation_dt * control_decimation)
                    obs = build_observation(data, cmd, default_angles, action, joint_effort, hopf_xy, config)
                    if abs(float(cmd[0])) <= hopf.stop_eps and hopf.is_at_rest:
                        action = np.zeros(num_actions, dtype=np.float32)
                    else:
                        action = session.run([output_name], {input_name: obs.reshape(1, -1)})[0].reshape(-1).astype(np.float32)
                    target_dof_pos = action * float(config["action_scale"]) + default_angles
                    print("actions:", action)
                else:
                    action = np.zeros(num_actions, dtype=np.float32)
                    target_dof_pos = default_angles.copy()

            viewer.sync()
            now = time.time()

            sleep_time = simulation_dt - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    staged_assets.cleanup()
