"""Parallel ankle demo. Frame: +X toe, +Y left, +Z up."""
import argparse
import ctypes
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

MODEL = Path(__file__).with_name("parallel_ankle.xml")
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "logs" / "parallel_ankle"
KP, KD, TORQUE_LIMIT = 120.0, 2.0, 20.0
VK_UP = 0x26


def up_arrow_pressed():
    """Return whether the keyboard Up Arrow key is currently held on Windows."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_UP) & 0x8000)


class TargetActions:
    """Eight smooth, six-second ankle-space excursions."""
    duration_per_action = 1.0
    actions_deg = ((34, 0), (-34, 0), (0, 30), (0, -30),
                   (34, 30), (34, -30), (-34, 30), (-34, -30))
    names = ("P+", "P-", "R+", "R-", "P+R+", "P+R-", "P-R+", "P-R-")

    @property
    def duration(self):
        return len(self.actions_deg) * self.duration_per_action

    def get(self, t):
        index = min(int(t / self.duration_per_action), len(self.actions_deg) - 1)
        u = np.clip((t % self.duration_per_action) / self.duration_per_action, 0, 1)
        extreme = np.deg2rad(self.actions_deg[index])
        return (extreme * np.sin(np.pi * u) ** 2,
                extreme * np.pi / self.duration_per_action * np.sin(2 * np.pi * u),
                self.names[index])


class ForwardKinematics:
    """Numerically solve ankle angles from measured motor angles."""
    def __init__(self, model):
        self.origin = model.body("pitch_carrier").pos.copy()
        self.pivots = np.array([model.body(f"crank_{i}").pos for i in (1, 2)])
        self.mounts = np.array([model.site(f"foot_end_{i}").pos for i in (1, 2)])
        self.radii = np.array([-model.body(f"rod_{i}").pos[0] for i in (1, 2)])
        self.lengths = np.array([np.linalg.norm(model.site(f"rod_end_{i}").pos) for i in (1, 2)])
        self.limits = np.array([model.joint(n).range for n in ("ankle_pitch", "ankle_roll")])
        self.previous = np.zeros(2)

    @staticmethod
    def rotation(q):
        p, r = q
        cp, sp, cr, sr = np.cos(p), np.sin(p), np.cos(r), np.sin(r)
        return np.array(((cp, sp * sr, sp * cr), (0, cr, -sr), (-sp, cp * sr, cp * cr)))

    def solve(self, motors):
        motors = np.asarray(motors)
        crank_ends = self.pivots + np.column_stack((-self.radii * np.cos(motors), np.zeros(2), -self.radii * np.sin(motors)))
        
        def residual(q):
            ends = self.mounts @ self.rotation(q).T + self.origin
            return np.linalg.norm(ends - crank_ends, axis=1) - self.lengths
        
        q = self.previous.copy()
        eye, h = np.eye(2), 1e-6
        for _ in range(30):
            error = residual(q)
            if np.max(np.abs(error)) < 1e-10:
                self.previous = q.copy()
                return q, float(np.max(np.abs(error)))
            
            jacobian = np.column_stack([(residual(q + eye[i] * h)-residual(q - eye[i] * h)) / (2 * h) for i in range(2)])
            if np.linalg.cond(jacobian) > 1e8:
                raise RuntimeError("Forward kinematics singularity")
            
            step = np.linalg.solve(jacobian, -error)
            for scale in (1, .5, .25, .125, .0625, .03125):
                candidate = np.clip(q + scale * step, self.limits[:, 0], self.limits[:, 1])
                if np.linalg.norm(residual(candidate)) < np.linalg.norm(error):
                    q = candidate
                    break
            else:
                raise RuntimeError("Forward kinematics did not converge")
            
        raise RuntimeError("Forward kinematics iteration limit")


class InverseKinematics:
    """Solve motor angles and J=d(motor angles)/d(ankle angles)."""
    def __init__(self, model):
        self.model = model
        self.origin = model.body("pitch_carrier").pos.copy()
        self.pivots = np.array([model.body(f"crank_{i}").pos for i in (1, 2)])
        self.mounts = np.array([model.site(f"foot_end_{i}").pos for i in (1, 2)])
        self.radii = np.array([-model.body(f"rod_{i}").pos[0] for i in (1, 2)])
        self.lengths = np.array([np.linalg.norm(model.site(f"rod_end_{i}").pos) for i in (1, 2)])

    def solve(self, ankle):
        rotation = ForwardKinematics.rotation(ankle)
        result = []
        for i in range(2):
            v = rotation @ self.mounts[i] + self.origin - self.pivots[i]
            a, b, radius = -v[0], -v[2], self.radii[i]
            ratio = (v @ v + radius ** 2 - self.lengths[i] ** 2) / (2 * radius * np.hypot(a, b))
            if abs(ratio) > 1 + 1e-10:
                raise ValueError("Unreachable ankle target")
            
            center = np.arctan2(b, a)
            roots = (center + np.array((-1, 1)) * np.arccos(np.clip(ratio, -1, 1))+np.pi) % (2 * np.pi) - np.pi
            result.append(min(roots, key=abs))
        return np.asarray(result)

    def jacobian(self, ankle):
        eye, h = np.eye(2), 1e-5
        return np.column_stack([(self.solve(ankle + eye[i] * h) - self.solve(ankle - eye[i] * h)) / (2 * h) for i in range(2)])


class Evaluation:
    """Keep samples in memory, print metrics, and show waveforms."""
    def __init__(self, mode):
        self.mode, self.rows, self.next_sample = mode, [], 0.0

    def record(self, t, target, feedback, simulation, motor_target, motor,
               torque, residual, condition):
        if t+1e-9 < self.next_sample:
            return
        self.rows.append([t, *target, *feedback, *simulation, *motor_target,
                          *motor, *torque, residual, condition])
        self.next_sample = t+.01

    def show(self):
        v = np.asarray(self.rows)
        tracking = v[:, 1:3] - v[:, 3:5]
        fk_error = v[:, 3:5] - v[:, 5:7]
        print(f"\n{self.mode} evaluation")
        print("  ankle tracking RMSE [pitch, roll] rad:", np.sqrt(np.mean(tracking ** 2, axis=0)))
        print("  ankle tracking max abs [pitch, roll] rad:", np.max(np.abs(tracking), axis=0))
        print("  FK vs MuJoCo max abs [pitch, roll] rad:", np.max(np.abs(fk_error), axis=0))
        print("  peak motor torque [M1, M2] Nm:", np.max(np.abs(v[:, 11:13]), axis=0))
        print("  max FK residual m:", np.max(v[:, 13]))
        print("  max Jacobian condition:", np.max(v[:, 14]))
        figure, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        for i, name in enumerate(("Pitch", "Roll")):
            axes[0, i].plot(v[:, 0], v[:, 1 + i], label="Target")
            axes[0, i].plot(v[:, 0], v[:, 3 + i], "--", label="FK feedback")
            axes[0, i].plot(v[:, 0], v[:, 5 + i], ":", label="MuJoCo")
            axes[0, i].set_ylabel(f"{name} (rad)")
            # axes[1, i].plot(v[:, 0], tracking[:, i], label="Target - FK")
            axes[1, i].plot(v[:, 0], fk_error[:, i], label="FK - MuJoCo")
            axes[1, i].set_ylabel("Error (rad)")
            axes[2, i].plot(v[:, 0], v[:, 7 + i], label="Target")
            axes[2, i].plot(v[:, 0], v[:, 9 + i], "--", label="Actual")
            torque_axis = axes[2, i].twinx()
            torque_axis.plot(v[:, 0], v[:, 11 + i], "r", alpha=.45, label="Torque")
            axes[2, i].set_ylabel(f"Motor {i + 1} (rad)")
            torque_axis.set_ylabel("Torque (Nm)")
            for row in range(3):
                axes[row, i].grid(alpha=.3)
                axes[row, i].set_xlabel("Time (s)")
                axes[row, i].legend(loc="upper left")
            torque_axis.legend(loc="upper right")
        figure.suptitle(f"Parallel ankle {self.mode} control")
        figure.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{self.mode.lower()}_control_evaluation.png"
        figure.savefig(output_path, dpi=150)
        print("  waveform:", output_path)
        plt.show()


def pdcontroller(target, feedback, target_velocity, feedback_velocity, kp, kd):
    del target_velocity
    return kp * (target - feedback) + kd * (-feedback_velocity)


def joint_state(data, names):
    return (np.array([data.joint(n).qpos[0] for n in names]),
            np.array([data.joint(n).qvel[0] for n in names]))


def control_step(mode, data, targets, fk, ik):
    ankle_target, ankle_target_velocity, phase = targets.get(data.time)
    motor, motor_velocity = joint_state(data, ("motor_1", "motor_2"))
    ankle, residual = fk.solve(motor)
    jacobian = ik.jacobian(ankle)
    condition = np.linalg.cond(jacobian)
    if condition > 1e6:
        raise RuntimeError("Transmission Jacobian singularity")
    motor_target = ik.solve(ankle_target)
    if mode == "PR":
        ankle_velocity = np.linalg.solve(jacobian, motor_velocity)
        ankle_torque = pdcontroller(ankle_target, ankle, ankle_target_velocity,
                                    ankle_velocity, KP, KD)
        motor_torque = np.linalg.solve(jacobian.T, ankle_torque)
    else:
        motor_target_velocity = ik.jacobian(ankle_target) @ ankle_target_velocity
        motor_torque = pdcontroller(motor_target, motor, motor_target_velocity,
                                    motor_velocity, KP, KD)
    motor_torque = np.clip(motor_torque, -TORQUE_LIMIT, TORQUE_LIMIT)
    data.ctrl[:] = motor_torque
    return phase, ankle_target, ankle, motor_target, motor, motor_torque, residual, condition


def run(mode, no_view):
    model = mujoco.MjModel.from_xml_string(MODEL.read_text(encoding="utf-8"))
    data, targets = mujoco.MjData(model), TargetActions()
    fk, ik, evaluation = ForwardKinematics(model), InverseKinematics(model), Evaluation(mode)

    def step():
        result = control_step(mode, data, targets, fk, ik)
        simulation, _ = joint_state(data, ("ankle_pitch", "ankle_roll"))
        evaluation.record(data.time, result[1], result[2], simulation, result[3],
                          result[4], result[5], result[6], result[7])
        mujoco.mj_step(model, data)
        return result

    if no_view:
        while data.time < targets.duration:
            step()
    else:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:], viewer.cam.distance = (0, 0, .26), .95
            viewer.cam.azimuth, viewer.cam.elevation = 25, -15
            wall_start, next_print = time.perf_counter(), 0.0
            started = False
            print(f"{mode}: press the Up Arrow key to start motion.", flush=True)
            while viewer.is_running() and data.time < targets.duration:
                frame_start = time.perf_counter()

                # Keep the mechanism at its initial pose and do not advance
                # simulation time until the user explicitly starts the test.
                if not started:
                    viewer.sync()
                    if up_arrow_pressed():
                        started = True
                        wall_start = time.perf_counter()
                        next_print = 0.0
                        print(f"{mode}: motion started.", flush=True)
                    else:
                        time.sleep(max(0, 1/60-(time.perf_counter()-frame_start)))
                    continue

                target_time = min(frame_start-wall_start, data.time+.1)
                with viewer.lock():
                    while data.time < target_time and data.time < targets.duration:
                        result = step()
                    mujoco.mj_forward(model, data)
                if data.time >= next_print:
                    print(f"{mode} t={data.time:5.2f}s {result[0]} | target/FK rad: "
                          f"{result[1]} / {result[2]}", flush=True)
                    next_print = data.time+.5
                viewer.sync()
                time.sleep(max(0, 1/60-(time.perf_counter()-frame_start)))
            viewer.close()
    evaluation.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--mode", choices=("PR", "AB"), type=str.upper)
    args = parser.parse_args()
    for mode in ((args.mode,) if args.mode else ("PR", "AB")):
        run(mode, args.no_view)


if __name__ == "__main__":
    main()
