from __future__ import annotations

import argparse
import ctypes
import math
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_DIR / "assets" / "bsrl" / "urdf" / "export_floating.xml"
DEFAULT_POLICY = PROJECT_DIR / "policy" / "plane" / "hopf_joint" / "exported" / "policy.onnx"

JOINT_NAMES = [
    "joint_right_hip_yaw",
    "joint_right_hip_roll",
    "joint_right_hip_pitch",
    "joint_right_knee_pitch",
    "joint_right_ankle_pitch",
    "joint_right_ankle_roll",
    "joint_left_hip_yaw",
    "joint_left_hip_roll",
    "joint_left_hip_pitch",
    "joint_left_knee_pitch",
    "joint_left_ankle_pitch",
    "joint_left_ankle_roll",
]

KP = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 100.0, 100.0, 100.0, 150.0, 40.0, 40.0])
KD = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0])
PEAK_POWER = np.array([1250.0, 1050.0, 1050.0, 1050.0, 1250.0, 1250.0, 1250.0, 1050.0, 1050.0, 1050.0, 1250.0, 1250.0])
ACTION_SCALE = 0.25
KEYFRAME = "default_stand"
DURATION = 0.0
RENDER_FPS = 60.0
COMMAND = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WALK_VX = 0.8
WALK_KEY = ord("V")
DECIMATION = 20
KP_SCALE = 1.0
KD_SCALE = 1.0
SHOW_COLLISION = False
PLOT_HISTORY = 1000
PLOT_HZ = 30.0
PLOT_SIM_STEPS = 10


class MasterHopf:
    def __init__(self) -> None:
        self.mu = 1.0
        self.gamma = 20.0
        self.stop_eps = 1e-6
        self.phase_eps = 2e-2
        self.return_omega_min = 0.0
        self.return_omega_max = 2.0 * math.pi * 5.0
        self.x = 0.0
        self.y = 0.0
        self.omega = 0.0
        self.last_omega = self.return_omega_max
        self.reset()

    @property
    def phase(self) -> float:
        return math.atan2(self.y, self.x) % (2.0 * math.pi)

    def reset(self, phase: float = 0.0) -> None:
        radius = math.sqrt(self.mu)
        self.x = radius * math.cos(phase)
        self.y = radius * math.sin(phase)
        self.omega = 0.0
        self.last_omega = self.return_omega_max

    def step(self, omega_cmd: float, dt: float) -> np.ndarray:
        phase = self.phase
        is_walking = abs(omega_cmd) > self.stop_eps
        is_near_start = phase <= self.phase_eps or phase >= 2.0 * math.pi - self.phase_eps

        if is_walking:
            self.omega = omega_cmd
            self.last_omega = abs(omega_cmd)
        elif is_near_start:
            self.reset()
            return np.array([self.x, self.y], dtype=np.float32)
        else:
            self.omega = min(max(self.last_omega, self.return_omega_min), self.return_omega_max)

        r2 = self.x * self.x + self.y * self.y
        dx = self.gamma * (self.mu - r2) * self.x - self.omega * self.y
        dy = self.gamma * (self.mu - r2) * self.y + self.omega * self.x
        self.x += dx * dt
        self.y += dy * dt

        phase = self.phase
        if not is_walking and (phase <= self.phase_eps or phase >= 2.0 * math.pi - self.phase_eps):
            self.reset()
        return np.array([self.x, self.y], dtype=np.float32)

    def step_velocity(self, vx: float, dt: float) -> np.ndarray:
        freq = 0.0 if abs(vx) < 1e-6 else 0.45 * abs(vx) + 0.60
        return self.step(2.0 * math.pi * freq, dt)


def stage_bsrl_to_ascii(xml_path: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix="bsrl_check_")
    staged_bsrl_dir = Path(temp_dir.name) / "bsrl"
    shutil.copytree(xml_path.parents[1], staged_bsrl_dir)
    return temp_dir, staged_bsrl_dir / "urdf" / xml_path.name


def reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData, key_name: str) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        raise ValueError(f"Keyframe '{key_name}' not found. Available keyframes: {names}")
    data.qpos[:] = model.key_qpos[key_id]
    data.qvel[:] = 0.0
    data.ctrl[:] = model.key_ctrl[key_id]
    model.qpos0[:] = model.key_qpos[key_id]
    mujoco.mj_forward(model, data)
    return key_id


def joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = []
    qpos_addr = []
    qvel_addr = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found.")
        joint_ids.append(joint_id)
        qpos_addr.append(model.jnt_qposadr[joint_id])
        qvel_addr.append(model.jnt_dofadr[joint_id])
    return np.array(joint_ids), np.array(qpos_addr), np.array(qvel_addr)


def check_actuator_order(model: mujoco.MjModel) -> None:
    actuator_joints = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    if actuator_joints != JOINT_NAMES:
        raise ValueError(f"Actuator order mismatch.\nExpected: {JOINT_NAMES}\nActual: {actuator_joints}")


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
    q = q / max(np.linalg.norm(q), 1e-12)
    return quat_mul(quat_mul(quat_conj(q), np.array([0.0, *v], dtype=np.float64)), q)[1:]


def build_obs(
    data: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    default_q: np.ndarray,
    command: np.ndarray,
    last_action: np.ndarray,
    joint_effort: np.ndarray,
    hopf_xy: np.ndarray,
) -> np.ndarray:
    base_quat = data.qpos[3:7].copy()
    base_ang_vel = quat_rotate_inverse(base_quat, data.qvel[3:6].copy()) * 0.2
    projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
    obs = np.concatenate(
        [
            base_ang_vel,
            projected_gravity,
            command,
            data.qpos[qpos_addr] - default_q,
            data.qvel[qvel_addr] * 0.05,
            joint_effort * 0.01,
            last_action,
            hopf_xy,
        ]
    )
    return obs.astype(np.float32)


def base_ang_vel(data: mujoco.MjData) -> np.ndarray:
    return quat_rotate_inverse(data.qpos[3:7].copy(), data.qvel[3:6].copy())


def projected_gravity(data: mujoco.MjData) -> np.ndarray:
    return quat_rotate_inverse(data.qpos[3:7].copy(), np.array([0.0, 0.0, -1.0]))


def pd_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    target_q: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    torque = kp * (target_q - data.qpos[qpos_addr]) - kd * data.qvel[qvel_addr]
    torque = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    power_torque_limit = PEAK_POWER / np.maximum(np.abs(data.qvel[qvel_addr]), 1e-3)
    torque_limit = np.minimum(np.maximum(np.abs(model.actuator_ctrlrange[:, 0]), np.abs(model.actuator_ctrlrange[:, 1])), power_torque_limit)
    torque = np.clip(torque, -torque_limit, torque_limit)
    data.ctrl[:] = torque
    mujoco.mj_step(model, data)
    return torque


def load_policy(policy_path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"policy_input={input_info.name} shape={input_info.shape}")
    print(f"policy_output={output_info.name} shape={output_info.shape}")
    return session, input_info.name


def is_walk_key_pressed() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(WALK_KEY) & 0x8000)


def update_command_from_keyboard(command: np.ndarray) -> None:
    target_vx = WALK_VX if is_walk_key_pressed() else 0.0
    if abs(command[0] - target_vx) > 1e-6:
        command[0] = target_vx
        state = "hold V" if target_vx > 0.0 else "release V"
        print(f"command {state}: [vx={command[0]:.3f}, vy={command[1]:.3f}, wz={command[2]:.3f}]")


def run_obs_plot(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    default_q: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    command: np.ndarray,
    hopf: MasterHopf,
) -> None:
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets
    except ImportError as exc:
        raise RuntimeError("Install PyQtGraph first: pip install pyqtgraph PyQt5") from exc

    if args.plot == "base_ang_vel":
        title = "base_ang_vel"
        title_suffix = "raw body frame"
        y_label = "rad/s"
        curve_names = ("x", "y", "z")

        def sample() -> np.ndarray:
            return base_ang_vel(data)
    elif args.plot == "projected_gravity":
        title = "projected_gravity"
        title_suffix = "policy obs[3:6]"
        y_label = "gravity projection"
        curve_names = ("x", "y", "z")

        def sample() -> np.ndarray:
            return projected_gravity(data)
    elif args.plot == "velocity_commands":
        title = "velocity_commands"
        title_suffix = "policy obs[6:9]"
        y_label = "command"
        curve_names = ("vx", "vy", "wz")

        def sample() -> np.ndarray:
            return command.copy()
    elif args.plot == "hopf_xy":
        title = "hopf_master_xy"
        title_suffix = "policy obs[57:59] with command vx"
        y_label = "Hopf state / vx"
        curve_names = ("hopf_x", "hopf_y", "command_vx")

        def sample() -> np.ndarray:
            hopf_xy = hopf.step_velocity(command[0], model.opt.timestep * DECIMATION)
            return np.array([hopf_xy[0], hopf_xy[1], command[0]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported plot group: {args.plot}")

    times = deque(maxlen=PLOT_HISTORY)
    xs = deque(maxlen=PLOT_HISTORY)
    ys = deque(maxlen=PLOT_HISTORY)
    zs = deque(maxlen=PLOT_HISTORY)

    app = QtWidgets.QApplication([])
    window = pg.GraphicsLayoutWidget(title=f"BSRL {title} - {title_suffix}")
    window.resize(1000, 520)
    plot = window.addPlot(title=f"{title} {title_suffix}")
    plot.setLabel("bottom", "time", "s")
    plot.setLabel("left", y_label)
    plot.getAxis("bottom").enableAutoSIPrefix(False)
    plot.getAxis("left").enableAutoSIPrefix(False)
    plot.addLegend()
    plot.showGrid(x=True, y=True, alpha=0.3)
    curve_x = plot.plot(pen=pg.mkPen("#e74c3c", width=2), name=curve_names[0])
    curve_y = plot.plot(pen=pg.mkPen("#2ecc71", width=2), name=curve_names[1])
    curve_z = plot.plot(pen=pg.mkPen("#3498db", width=2), name=curve_names[2])
    window.show()

    start_time = time.time()
    last_plot_time = 0.0
    with viewer.launch_passive(model, data) as sim_viewer:
        sim_viewer.opt.geomgroup[3] = 1 if SHOW_COLLISION else 0

        def update() -> None:
            nonlocal last_plot_time
            update_command_from_keyboard(command)
            for _ in range(PLOT_SIM_STEPS):
                pd_step(model, data, qpos_addr, qvel_addr, default_q, kp, kd)

            if data.time < last_plot_time:
                times.clear()
                xs.clear()
                ys.clear()
                zs.clear()
                hopf.reset()

            value = sample()
            times.append(data.time)
            xs.append(value[0])
            ys.append(value[1])
            zs.append(value[2] if value.shape[0] > 2 else 0.0)
            last_plot_time = data.time

            t = np.asarray(times)
            curve_x.setData(t, np.asarray(xs))
            curve_y.setData(t, np.asarray(ys))
            curve_z.setData(t, np.asarray(zs))
            sim_viewer.sync()

            if not sim_viewer.is_running():
                app.quit()
            if DURATION > 0.0 and time.time() - start_time >= DURATION:
                app.quit()

        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(max(1, round(1000.0 / PLOT_HZ)))
        app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BSRL MuJoCo standing action test.")
    parser.add_argument(
        "--plot",
        choices=["none", "base_ang_vel", "projected_gravity", "velocity_commands", "hopf_xy"],
        default="none",
    )
    args = parser.parse_args()

    temp_dir, staged_xml = stage_bsrl_to_ascii(DEFAULT_XML)
    try:
        model = mujoco.MjModel.from_xml_path(str(staged_xml))
        data = mujoco.MjData(model)
        key_id = reset_to_keyframe(model, data, KEYFRAME)
        check_actuator_order(model)

        _, qpos_addr, qvel_addr = joint_addresses(model)
        default_q = model.key_qpos[key_id, qpos_addr].copy()
        kp = KP * KP_SCALE
        kd = KD * KD_SCALE
        command = COMMAND.copy()

        session, input_name = load_policy(DEFAULT_POLICY)
        hopf = MasterHopf()
        last_action = np.zeros(12, dtype=np.float64)
        torque = np.zeros(12, dtype=np.float64)

        hopf_xy = hopf.step_velocity(command[0], model.opt.timestep * DECIMATION)
        obs = build_obs(data, qpos_addr, qvel_addr, default_q, command, last_action, torque, hopf_xy)
        first_action = session.run(None, {input_name: obs[None, :]})[0].reshape(-1).astype(np.float64)
        print(f"first_action shape={first_action.shape} values={first_action}")

        if args.plot != "none":
            run_obs_plot(args, model, data, qpos_addr, qvel_addr, default_q, kp, kd, command, hopf)
            return

        frame_dt = 1.0 / max(RENDER_FPS, 1.0)
        steps_per_frame = max(1, round(frame_dt / model.opt.timestep))
        start_time = time.time()
        with viewer.launch_passive(model, data) as sim_viewer:
            sim_viewer.opt.geomgroup[3] = 1 if SHOW_COLLISION else 0
            while sim_viewer.is_running():
                frame_start = time.time()
                update_command_from_keyboard(command)
                for _ in range(steps_per_frame):
                    torque = pd_step(model, data, qpos_addr, qvel_addr, default_q, kp, kd)

                sim_viewer.sync()
                if DURATION > 0.0 and time.time() - start_time >= DURATION:
                    break
                sleep_time = frame_dt - (time.time() - frame_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
