import csv
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PITCH_JOINT_NAMES = (
    "joint_left_hip_pitch",
    "joint_left_knee_pitch",
    "joint_right_hip_pitch",
    "joint_right_knee_pitch",
)
FOOT_BODY_NAMES = {
    "left_contact": "link_left_ankle_roll",
    "right_contact": "link_right_ankle_roll",
}


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
    quat = data.qpos[3:7]
    lin_vel_b = quat_rotate_inverse(quat, data.qvel[0:3])
    ang_vel_b = data.qvel[3:6].copy()
    return lin_vel_b, ang_vel_b


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
        self.velocity_freq_slope = float(config["velocity_freq_slope"])
        self.velocity_freq_intercept = float(config["velocity_freq_intercept"])
        self.velocity_freq_min = float(config.get("velocity_freq_min", 0.0))
        self.velocity_freq_max = float(config.get("velocity_freq_max", float("inf")))
        self.return_omega_min = 2.0 * np.pi * float(config["return_freq_min"])
        self.return_omega_max = 2.0 * np.pi * float(config["return_freq_max"])
        self.reset()

    @property
    def phase(self) -> float:
        return np.arctan2(self.y, self.x) % (2.0 * np.pi)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def reset(self, phase: float = 0.0) -> None:
        radius = np.sqrt(self.mu)
        self.x = radius * np.cos(phase)
        self.y = radius * np.sin(phase)
        self.omega = 0.0
        self.last_omega = self.return_omega_max

    def velocity_to_frequency(self, vx: float) -> float:
        if abs(vx) < self.stop_eps:
            return 0.0
        freq = self.velocity_freq_slope * abs(float(vx)) + self.velocity_freq_intercept
        return min(max(freq, self.velocity_freq_min), self.velocity_freq_max)

    def step_velocity(self, vx: float, dt: float) -> np.ndarray:
        return self.step(2.0 * np.pi * self.velocity_to_frequency(vx), dt)

    def step(self, omega_cmd: float, dt: float) -> np.ndarray:
        phase = self.phase
        walking = abs(omega_cmd) > self.stop_eps
        near_start = phase <= self.phase_eps or phase >= 2.0 * np.pi - self.phase_eps

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
            if phase <= self.phase_eps or phase >= 2.0 * np.pi - self.phase_eps:
                self.reset()

        return self.xy


class RuntimeState:
    def __init__(self, config: dict, default_angles: np.ndarray, kds: np.ndarray) -> None:
        num_actions = int(config["num_actions"])
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.target_q = default_angles.copy()
        self.target_dq = np.zeros_like(kds)
        self.joint_effort = np.zeros(num_actions, dtype=np.float64)
        self.command = np.asarray(config["cmd_init"], dtype=np.float32)


class JointBinding:
    def __init__(self, qpos_addr: np.ndarray, qvel_addr: np.ndarray, actuator_ids: np.ndarray, ctrlrange: np.ndarray):
        self.qpos_addr = qpos_addr
        self.qvel_addr = qvel_addr
        self.actuator_ids = actuator_ids
        self.ctrlrange = ctrlrange


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
    omega = data.qvel[3:6] * float(config["ang_vel_scale"])
    gravity = projected_gravity(data.qpos[3:7])
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


def parse_command_segments(values: list[str] | None) -> list[tuple[float, float, float, float]]:
    if not values:
        return [
            (0.0, 0.0, 0.0, 5.0),
            (0.6, 0.0, 0.0, 5.0),
            (0.9, 0.0, 0.0, 5.0),
            (1.2, 0.0, 0.0, 5.0),
            (1.5, 0.0, 0.0, 5.0),
            (1.0, 0.0, 0.0, 5.0),
            (0.6, 0.0, 0.0, 5.0),
            (0.0, 0.0, 0.0, 5.0),
        ]

    segments = []
    for value in values:
        parts = [float(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Command segment must be 'vx,vy,yaw_rate,duration', got: {value}")
        if parts[3] <= 0.0:
            raise ValueError(f"Command duration must be positive, got: {value}")
        segments.append((parts[0], parts[1], parts[2], parts[3]))
    return segments


def command_at_time(segments: list[tuple[float, float, float, float]], t: float) -> np.ndarray:
    elapsed = 0.0
    for vx, vy, yaw_rate, duration in segments:
        elapsed += duration
        if t < elapsed:
            return np.asarray([vx, vy, yaw_rate], dtype=np.float32)
    return np.asarray(segments[-1][:3], dtype=np.float32)


def bind_named_qpos(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> dict[str, int]:
    qpos_addr = {}
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {name}")
        qpos_addr[name] = int(model.jnt_qposadr[joint_id])
    return qpos_addr


def body_geom_ids(model: mujoco.MjModel, body_name: str) -> set[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    return {geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id}


def foot_contact_states(model: mujoco.MjModel, data: mujoco.MjData, foot_geom_ids: dict[str, set[int]]) -> dict[str, int]:
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    contacts = {name: 0 for name in foot_geom_ids}

    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 != ground_id and geom2 != ground_id:
            continue
        other_geom = geom2 if geom1 == ground_id else geom1
        for name, geom_ids in foot_geom_ids.items():
            if other_geom in geom_ids:
                contacts[name] = 1
    return contacts


def record_sample(
    samples: list[dict[str, float]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: RuntimeState,
    pitch_qpos_addr: dict[str, int],
    foot_geom_ids: dict[str, set[int]],
) -> None:
    lin_vel_b, ang_vel_b = root_velocity_body(data)
    contacts = foot_contact_states(model, data, foot_geom_ids)
    sample = {
        "time": float(data.time),
        "target_vx": float(state.command[0]),
        "actual_vx": float(lin_vel_b[0]),
        "target_vy": float(state.command[1]),
        "actual_vy": float(lin_vel_b[1]),
        "target_yaw_rate": float(state.command[2]),
        "actual_yaw_rate": float(ang_vel_b[2]),
        "left_hip_pitch": float(data.qpos[pitch_qpos_addr["joint_left_hip_pitch"]]),
        "left_knee_pitch": float(data.qpos[pitch_qpos_addr["joint_left_knee_pitch"]]),
        "right_hip_pitch": float(data.qpos[pitch_qpos_addr["joint_right_hip_pitch"]]),
        "right_knee_pitch": float(data.qpos[pitch_qpos_addr["joint_right_knee_pitch"]]),
        "left_contact": float(contacts["left_contact"]),
        "right_contact": float(contacts["right_contact"]),
    }
    samples.append(sample)


def truncate_initial_samples(samples: list[dict[str, float]], discard_initial_s: float) -> list[dict[str, float]]:
    if discard_initial_s <= 0.0:
        return samples
    return [sample for sample in samples if sample["time"] >= discard_initial_s]


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values.copy()

    kernel = np.ones(window_size, dtype=np.float64) / window_size
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_fft_window(size: int, window_name: str) -> np.ndarray:
    window_name = window_name.lower()
    if window_name == "hann":
        return np.hanning(size)
    if window_name == "hamming":
        return np.hamming(size)
    if window_name == "blackman":
        return np.blackman(size)
    if window_name == "rect":
        return np.ones(size, dtype=np.float64)
    raise ValueError(f"Unsupported STFT window '{window_name}'. Expected hann, hamming, blackman, or rect.")


def parabolic_peak_frequency(spectrum: np.ndarray, frequencies: np.ndarray, peak_index: int) -> float:
    if peak_index <= 0 or peak_index >= spectrum.size - 1:
        return float(frequencies[peak_index])

    alpha = float(spectrum[peak_index - 1])
    beta = float(spectrum[peak_index])
    gamma = float(spectrum[peak_index + 1])
    denominator = alpha - 2.0 * beta + gamma
    if abs(denominator) < 1e-12:
        return float(frequencies[peak_index])

    bin_offset = 0.5 * (alpha - gamma) / denominator
    bin_offset = float(np.clip(bin_offset, -0.5, 0.5))
    frequency_step = float(frequencies[1] - frequencies[0])
    return float(frequencies[peak_index] + bin_offset * frequency_step)


def dominant_frequency(
    signal: np.ndarray,
    sample_dt: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    fft_oversample: int,
    window_name: str,
) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    if np.max(np.abs(signal)) < 1e-4:
        return 0.0

    windowed_signal = signal * make_fft_window(signal.size, window_name)
    fft_size = int(2 ** np.ceil(np.log2(max(signal.size, 1)))) * max(1, fft_oversample)
    spectrum = np.abs(np.fft.rfft(windowed_signal, n=fft_size))
    frequencies = np.fft.rfftfreq(fft_size, d=sample_dt)
    valid = (frequencies >= min_frequency_hz) & (frequencies <= max_frequency_hz)
    if not np.any(valid):
        return 0.0

    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[np.argmax(spectrum[valid])]
    return parabolic_peak_frequency(spectrum, frequencies, int(peak_index))


def add_step_frequency(
    samples: list[dict[str, float]],
    stft_window_s: float,
    filter_window_s: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    fft_oversample: int,
    window_name: str,
    min_target_vx: float,
    min_actual_vx: float,
    min_hip_amplitude_rad: float,
) -> None:
    if len(samples) < 2:
        for sample in samples:
            sample["left_hip_frequency"] = 0.0
            sample["right_hip_frequency"] = 0.0
            sample["step_frequency_raw"] = 0.0
            sample["step_frequency"] = 0.0
        return

    t = np.asarray([sample["time"] for sample in samples], dtype=np.float64)
    sample_dt = float(np.median(np.diff(t)))
    window_size = max(8, int(round(stft_window_s / sample_dt)))
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    left_hip = np.asarray([sample["left_hip_pitch"] for sample in samples], dtype=np.float64)
    right_hip = np.asarray([sample["right_hip_pitch"] for sample in samples], dtype=np.float64)
    target_vx = np.asarray([sample["target_vx"] for sample in samples], dtype=np.float64)
    actual_vx = np.asarray([sample["actual_vx"] for sample in samples], dtype=np.float64)

    left_frequency = np.zeros(len(samples), dtype=np.float64)
    right_frequency = np.zeros(len(samples), dtype=np.float64)
    raw_frequency = np.zeros(len(samples), dtype=np.float64)

    for index in range(len(samples)):
        start = max(0, index - half_window)
        end = min(len(samples), index + half_window + 1)
        if end - start < 8:
            continue
        if abs(target_vx[index]) < min_target_vx:
            continue
        if np.mean(np.abs(actual_vx[start:end])) < min_actual_vx:
            continue

        left_window = left_hip[start:end]
        right_window = right_hip[start:end]
        if np.ptp(left_window) >= min_hip_amplitude_rad:
            current_left_frequency = dominant_frequency(
                left_window, sample_dt, min_frequency_hz, max_frequency_hz, fft_oversample, window_name
            )
        else:
            current_left_frequency = 0.0
        if np.ptp(right_window) >= min_hip_amplitude_rad:
            current_right_frequency = dominant_frequency(
                right_window, sample_dt, min_frequency_hz, max_frequency_hz, fft_oversample, window_name
            )
        else:
            current_right_frequency = 0.0
        left_frequency[index] = current_left_frequency
        right_frequency[index] = current_right_frequency

        valid_frequencies = [
            frequency for frequency in (current_left_frequency, current_right_frequency) if frequency > 0.0
        ]
        raw_frequency[index] = float(np.mean(valid_frequencies)) if valid_frequencies else 0.0

    for sample in samples:
        sample["left_hip_frequency"] = 0.0
        sample["right_hip_frequency"] = 0.0
        sample["step_frequency_raw"] = 0.0
        sample["step_frequency"] = 0.0

    filter_window_size = max(1, int(round(filter_window_s / sample_dt)))
    filtered_frequency = moving_average(raw_frequency, filter_window_size)
    for sample, left, right, raw, frequency in zip(
        samples, left_frequency, right_frequency, raw_frequency, filtered_frequency
    ):
        sample["left_hip_frequency"] = float(left)
        sample["right_hip_frequency"] = float(right)
        sample["step_frequency_raw"] = float(raw)
        sample["step_frequency"] = float(frequency)


def write_csv(samples: list[dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)


def contact_intervals(t: np.ndarray, contact: np.ndarray) -> list[tuple[float, float]]:
    if t.size == 0:
        return []

    intervals = []
    sample_dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    in_contact = bool(contact[0])
    start_time = float(t[0])

    for index in range(1, t.size):
        is_contact = bool(contact[index])
        if is_contact == in_contact:
            continue
        if in_contact:
            intervals.append((start_time, float(t[index] - start_time)))
        start_time = float(t[index])
        in_contact = is_contact

    if in_contact:
        intervals.append((start_time, float(t[-1] + sample_dt - start_time)))
    return intervals


def plot_contact_band(ax, t: np.ndarray, samples: list[dict[str, float]]) -> None:
    rows = (
        ("left_contact", "Left foot", 0.55, "#1f77b4", "#e8f1fb"),
        ("right_contact", "Right foot", 0.0, "#ff7f0e", "#fff0df"),
    )

    x_start = float(t[0])
    x_end = float(t[-1])
    for key, label, y_base, color, background in rows:
        ax.broken_barh([(x_start, x_end - x_start)], (y_base - 0.18, 0.36), facecolors=background, edgecolors="none")
        contact = np.asarray([sample[key] for sample in samples], dtype=bool)
        intervals = contact_intervals(t, contact)
        if intervals:
            ax.broken_barh(intervals, (y_base - 0.18, 0.36), facecolors=color, edgecolors="none")

    ax.set_title("Foot ground contact")
    ax.set_xlabel("time (s)")
    ax.set_ylim(-0.35, 0.9)
    ax.set_yticks([0.55, 0.0])
    ax.set_yticklabels(["Left foot", "Right foot"])
    ax.set_xlim(x_start, x_end)
    ax.grid(True, axis="x", alpha=0.3)


def target_velocity_segments(samples: list[dict[str, float]]) -> list[tuple[float, float, float]]:
    segments = []
    segment_start = float(samples[0]["time"])
    current_vx = float(samples[0]["target_vx"])

    for previous, current in zip(samples, samples[1:]):
        next_vx = float(current["target_vx"])
        if np.isclose(next_vx, current_vx):
            continue
        segment_end = float(current["time"])
        segments.append((segment_start, segment_end, current_vx))
        segment_start = segment_end
        current_vx = next_vx

    sample_dt = float(np.median(np.diff([sample["time"] for sample in samples]))) if len(samples) > 1 else 0.01
    segments.append((segment_start, float(samples[-1]["time"]) + sample_dt, current_vx))
    return segments


def shade_target_velocity_segments(ax, samples: list[dict[str, float]]) -> None:
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.06 * (y_max - y_min)
    for index, (start_time, end_time, target_vx) in enumerate(target_velocity_segments(samples)):
        color = "#f2f2f2" if index % 2 == 0 else "#e8eef7"
        ax.axvspan(start_time, end_time, facecolor=color, edgecolor="#aeb7c2", linewidth=0.8, alpha=0.55, zorder=0)
        if end_time - start_time < 0.4:
            continue
        ax.text(
            0.5 * (start_time + end_time),
            y_text,
            f"{target_vx:.1f} m/s",
            ha="center",
            va="top",
            fontsize=9,
            color="#3d4752",
        )


def plot_samples(samples: list[dict[str, float]], plot_path: Path, max_step_frequency_hz: float) -> None:
    t = np.asarray([sample["time"] for sample in samples])
    actual_vx = moving_average(np.asarray([sample["actual_vx"] for sample in samples]), max(1, round(0.15 / 0.01)))
    actual_vy = moving_average(np.asarray([sample["actual_vy"] for sample in samples]), max(1, round(0.15 / 0.01)))
    actual_yaw_rate = moving_average(
        np.asarray([sample["actual_yaw_rate"] for sample in samples]),
        max(1, round(0.15 / 0.01)),
    )

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    gs = fig.add_gridspec(5, 2)

    ax_vx = fig.add_subplot(gs[0, 0])
    ax_vy = fig.add_subplot(gs[0, 1])
    ax_yaw = fig.add_subplot(gs[1, 0])
    ax_joint = fig.add_subplot(gs[1, 1])
    ax_frequency = fig.add_subplot(gs[2, :])
    ax_contact = fig.add_subplot(gs[3:, :])

    ax_vx.plot(t, [sample["target_vx"] for sample in samples], label="target")
    ax_vx.plot(t, actual_vx, label="actual")
    ax_vx.set_title("Body-frame x velocity")
    ax_vx.set_ylabel("m/s")
    ax_vx.legend(loc="upper right")

    ax_vy.plot(t, [sample["target_vy"] for sample in samples], label="target")
    ax_vy.plot(t, actual_vy, label="actual")
    ax_vy.set_title("Body-frame y velocity")
    ax_vy.set_ylabel("m/s")
    ax_vy.legend(loc="upper right")

    ax_yaw.plot(t, [sample["target_yaw_rate"] for sample in samples], label="target")
    ax_yaw.plot(t, actual_yaw_rate, label="actual")
    ax_yaw.set_title("Body yaw rate")
    ax_yaw.set_ylabel("rad/s")
    ax_yaw.legend(loc="upper right")

    for key in ("left_hip_pitch", "left_knee_pitch", "right_hip_pitch", "right_knee_pitch"):
        ax_joint.plot(t, [sample[key] for sample in samples], label=key)
    ax_joint.set_title("Hip/knee pitch joint positions")
    ax_joint.set_ylabel("rad")
    ax_joint.legend(loc="upper right", fontsize=8)

    ax_frequency.plot(
        t,
        [sample["step_frequency"] for sample in samples],
        color="#2ca02c",
        linewidth=2.0,
        label="filtered step frequency",
    )
    ax_frequency.set_title("Detected step frequency by target x-velocity segment")
    ax_frequency.set_ylabel("Hz")
    ax_frequency.legend(loc="upper right")
    ax_frequency.set_axisbelow(True)
    y_max = max(1.0, min(max_step_frequency_hz, max(sample["step_frequency"] for sample in samples) * 1.15))
    ax_frequency.set_ylim(0.0, y_max)
    shade_target_velocity_segments(ax_frequency, samples)

    plot_contact_band(ax_contact, t, samples)

    for ax in (ax_vx, ax_vy, ax_yaw, ax_joint, ax_frequency):
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def run_fixed_command_test() -> tuple[Path, Path]:
    with (PROJECT_ROOT / "deploy" / "bsrl_hopf_deploy.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    policy_path = resolve_path(config["policy_path"])
    xml_path = resolve_path(config["xml_path"])
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    root_height = float(config.get("root_height", 0.8665))
    action_scale = float(config["action_scale"])

    kps = np.asarray(config["kps"], dtype=np.float64)
    kds = np.asarray(config["kds"], dtype=np.float64)
    default_angles = np.asarray(config["default_angles"], dtype=np.float64)
    peak_powers = np.asarray(config["peak_powers"], dtype=np.float64) if "peak_powers" in config else None
    velocity_limits = np.asarray(config["velocity_limits"], dtype=np.float64) if "velocity_limits" in config else None
    policy_joint_names = list(config.get("policy_joint_names", config.get("joint_names", [])))
    mujoco_joint_names = list(config.get("mujoco_joint_names", policy_joint_names))
    segments = parse_command_segments(None)
    total_duration = sum(segment[3] for segment in segments)
    sample_decimation = max(1, int(round(0.01 / simulation_dt)))

    output_root = PROJECT_ROOT / "logs" / "fixed_command_tests"
    csv_path = output_root / "fixed_command_trace.csv"
    plot_path = output_root / "fixed_command_trace.png"

    staged_assets, staged_xml_path = stage_assets_to_ascii_path(xml_path)
    try:
        model = mujoco.MjModel.from_xml_path(staged_xml_path)
        model.opt.timestep = simulation_dt
        data = mujoco.MjData(model)

        require_model_layout(model, mujoco_joint_names, policy_joint_names)
        binding = bind_joints(model, policy_joint_names)
        pitch_qpos_addr = bind_named_qpos(model, PITCH_JOINT_NAMES)
        foot_geom_ids = {
            contact_name: body_geom_ids(model, body_name) for contact_name, body_name in FOOT_BODY_NAMES.items()
        }

        session, input_name, output_name = load_policy(policy_path)
        state = RuntimeState(config, default_angles, kds)
        hopf = MasterHopf(config["hopf"])

        reset_to_default_pose(data, binding, default_angles, root_height)
        mujoco.mj_forward(model, data)

        samples = []
        counter = 0
        while data.time < total_duration:
            if counter % control_decimation == 0:
                state.command[:] = command_at_time(segments, data.time)
                hopf_xy = hopf.step_velocity(float(state.command[0]), simulation_dt * control_decimation)
                obs = build_observation(data, binding, state, default_angles, hopf_xy, config)
                state.action[:] = run_policy(session, input_name, output_name, obs)
                state.target_q[:] = state.action * action_scale + default_angles

            step_control(model, data, binding, state, kps, kds, peak_powers, velocity_limits)

            if counter % sample_decimation == 0:
                record_sample(samples, model, data, state, pitch_qpos_addr, foot_geom_ids)
            counter += 1

        samples = truncate_initial_samples(samples, 2.0)
        if not samples:
            raise RuntimeError("No samples were recorded. Check duration and sample_dt.")

        add_step_frequency(
            samples,
            4.0,
            0.25,
            0.3,
            3.0,
            16,
            "blackman",
            0.15,
            0.10,
            0.12,
        )
        write_csv(samples, csv_path)
        plot_samples(samples, plot_path, 3.0)
    finally:
        staged_assets.cleanup()

    return csv_path, plot_path


def main() -> None:
    csv_path, plot_path = run_fixed_command_test()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
