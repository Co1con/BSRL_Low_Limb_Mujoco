import numpy as np

TWO_PI = 2.0 * np.pi


def _check_dt(dt):
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be greater than zero")
    return dt


def _wrap_to_pi(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class RhythmHopf:
    """根据目标速度生成全局节律，并在停车时返回初始相位。"""

    def __init__(self):
        self.mu = 1.0
        self.gamma = 20.0
        self.frequency_slope = 0.3756343915
        self.frequency_intercept = 0.5563912798
        self.reset()

    def reset(self):
        # 静止时使用唯一的规范相位；起步腿在收到非零速度命令时再随机选择。
        self.reset_phase = 0.0
        radius = np.sqrt(self.mu)
        self.x = float(radius * np.cos(self.reset_phase))
        self.y = float(radius * np.sin(self.reset_phase))
        self.omega = 0.0
        self.is_running = False
        self.stop_requested = False

    def _prepare_start(self):
        self.reset_phase = float(np.random.choice((0.0, np.pi)))
        radius = np.sqrt(self.mu)
        self.x = float(radius * np.cos(self.reset_phase))
        self.y = float(radius * np.sin(self.reset_phase))

    @property
    def phase(self):
        return float(np.mod(np.arctan2(self.y, self.x), TWO_PI))

    @property
    def frequency(self):
        return self.omega / TWO_PI

    def velocity_to_frequency(self, velocity):
        speed = abs(float(velocity))
        if speed == 0.0:
            return 0.0
        return self.frequency_slope * speed + self.frequency_intercept

    def _stop_at_reset_phase(self):
        radius = np.sqrt(self.mu)
        self.x = float(radius * np.cos(self.reset_phase))
        self.y = float(radius * np.sin(self.reset_phase))
        self.omega = 0.0
        self.is_running = False
        self.stop_requested = False

    def step(self, velocity, dt):
        velocity = float(velocity)
        dt = _check_dt(dt)

        # 速度幅值决定步频，符号决定极限环旋转方向。
        if velocity != 0.0:
            if not self.is_running:
                self._prepare_start()
            self.omega = np.sign(velocity) * TWO_PI * self.velocity_to_frequency(velocity)
            self.is_running = True
            self.stop_requested = False
        elif self.is_running:
            self.stop_requested = True

        if not self.is_running:
            return self.x, self.y

        radius_squared = self.x * self.x + self.y * self.y
        radial = self.gamma * (self.mu - radius_squared)
        dx = radial * self.x - self.omega * self.y
        dy = radial * self.y + self.omega * self.x
        next_x = self.x + dx * dt
        next_y = self.y + dy * dt

        # 只在本次 reset 选择的 0 或 pi 相位停车。
        reset_direction = np.cos(self.reset_phase)
        at_reset_phase = (
            np.isclose(self.y, 0.0, atol=1e-12)
            and self.x * reset_direction > 0.0
        )
        crosses_reset_phase = (
            self.y * next_y <= 0.0
            and next_x * reset_direction > 0.0
        )

        if self.stop_requested and (at_reset_phase or crosses_reset_phase):
            self._stop_at_reset_phase()
        else:
            self.x, self.y = next_x, next_y
        return self.x, self.y


class JointOscillator:
    """以连续变化的自身频率软跟踪 Rhythm 的目标相位。"""

    def __init__(self, phase_offset):
        self.mu = 1.0
        self.gamma = 20.0
        self.phase_gain = 3.5
        self.frequency_response = 14.0
        self.phase_offset = float(phase_offset)
        self.reset()

    def reset(self):
        self.x = float(np.sqrt(self.mu))
        self.y = 0.0
        self.omega = 0.0
        self.phase_error = 0.0
        self.initialized = False

    @property
    def phase(self):
        return float(np.mod(np.arctan2(self.y, self.x), TWO_PI))

    @property
    def frequency(self):
        return self.omega / TWO_PI

    def _set_phase(self, phase):
        radius = np.sqrt(self.mu)
        self.x = float(radius * np.cos(phase))
        self.y = float(radius * np.sin(phase))

    def step(self, rhythm_phase, rhythm_frequency, dt):
        rhythm_phase = float(rhythm_phase)
        rhythm_frequency = float(rhythm_frequency)
        dt = _check_dt(dt)
        target_phase = rhythm_phase + self.phase_offset
        if not self.initialized:
            # 首帧直接建立目标相位差，避免启动时追相造成大幅超调。
            self._set_phase(target_phase)
            self.initialized = True

        self.phase_error = _wrap_to_pi(target_phase - self.phase)
        target_omega = (
            TWO_PI * rhythm_frequency
            + self.phase_gain * np.sin(self.phase_error)
        )
        if rhythm_frequency != 0.0:
            target_omega = np.sign(rhythm_frequency) * max(
                0.0, np.sign(rhythm_frequency) * target_omega
            )

        # Joint 的频率连续逼近目标频率，不复制 Rhythm 的频率阶跃。
        decay = np.exp(-self.frequency_response * dt)
        self.omega = target_omega + (self.omega - target_omega) * decay

        radius_squared = self.x * self.x + self.y * self.y
        radial = self.gamma * (self.mu - radius_squared)
        dx = radial * self.x - self.omega * self.y
        dy = radial * self.y + self.omega * self.x
        self.x += dx * dt
        self.y += dy * dt

        # Rhythm 停止后允许短路径回相，收敛后精确固定在目标相位。
        if rhythm_frequency == 0.0:
            remaining_error = _wrap_to_pi(target_phase - self.phase)
            if abs(remaining_error) < 1e-4 and abs(self.omega) < 1e-3:
                self._set_phase(target_phase)
                self.omega = 0.0
                self.phase_error = 0.0
        return self.x, self.y


class JointShaper:
    """使用髋或膝的三阶傅里叶系数将相位整形为关节角。"""

    COEFFICIENTS = {
        "hip": (
            0.0078388447,
            0.0000546934, 0.2613439175,
            -0.0482368723, -0.0312843541,
            0.0047676238, -0.0142935173,
        ),
        "knee": (
            0.2367006798,
            -0.0001668887, 0.3473199593,
            -0.0100145053, 0.2530838817,
            0.0287142304, 0.0462009950,
        ),
    }

    def __init__(self, joint_type):
        joint_type = str(joint_type).lower()
        if joint_type not in self.COEFFICIENTS:
            raise ValueError("joint_type must be 'hip' or 'knee'")
        self.coefficients = self.COEFFICIENTS[joint_type]
        self.angle = 0.0

    def step(self, phase):
        phase = float(phase)
        angle = self.coefficients[0]
        for order in range(1, 4):
            angle += self.coefficients[2 * order - 1] * np.sin(order * phase)
            angle += self.coefficients[2 * order] * np.cos(order * phase)
        self.angle = float(angle)
        return self.angle


class LowLimbCPG:
    """组合速度驱动的节律、四关节相位协调和关节波形整形。"""

    def __init__(self):
        self.rhythm = RhythmHopf()

        self.phase_speed_slope = -0.2865263613
        self.phase_speed_intercept = 5.1776800215
        self.max_integration_dt = 0.005
        self.start_transition_time = 0.3
        self.stop_transition_time = 0.3
        self.standing_angles = {
            "left_hip": 0.10,
            "right_hip": 0.10,
            "left_knee": 0.20,
            "right_knee": 0.20,
        }

        initial_phase_difference = self.phase_speed_intercept
        self.joints = {
            "left_hip": JointOscillator(0.0),
            "right_hip": JointOscillator(np.pi),
            "left_knee": JointOscillator(-initial_phase_difference),
            "right_knee": JointOscillator(-initial_phase_difference + np.pi),
        }
        self.shapers = {
            "left_hip": JointShaper("hip"),
            "right_hip": JointShaper("hip"),
            "left_knee": JointShaper("knee"),
            "right_knee": JointShaper("knee"),
        }

        self.velocity = 0.0
        self.motion_blend = 0.0
        self.hip_knee_phase_difference = initial_phase_difference
        self.gait_angles = self.standing_angles.copy()
        self.angles = self.standing_angles.copy()

    def reset(self):
        """恢复对称站姿，并清空四关节的振荡状态。"""
        self.rhythm.reset()
        self.velocity = 0.0
        self.motion_blend = 0.0
        self.hip_knee_phase_difference = self.phase_speed_intercept
        self.joints["left_hip"].phase_offset = 0.0
        self.joints["right_hip"].phase_offset = np.pi
        self.joints["left_knee"].phase_offset = -self.hip_knee_phase_difference
        self.joints["right_knee"].phase_offset = -self.hip_knee_phase_difference + np.pi
        for name, joint in self.joints.items():
            joint.reset()
            self.shapers[name].angle = self.standing_angles[name]
            self.gait_angles[name] = self.standing_angles[name]
            self.angles[name] = self.standing_angles[name]

    def _update_motion_blend(self, moving, dt):
        transition_time = (
            self.start_transition_time if moving else self.stop_transition_time
        )
        target = 1.0 if moving else 0.0
        if transition_time <= 0.0:
            self.motion_blend = target
            return
        step = dt / transition_time
        if moving:
            self.motion_blend = min(1.0, self.motion_blend + step)
        else:
            self.motion_blend = max(0.0, self.motion_blend - step)

    def _update_knee_phase_offsets(self):
        self.hip_knee_phase_difference = (
            self.phase_speed_slope * abs(self.velocity)
            + self.phase_speed_intercept
        )
        knee_offset = -self.hip_knee_phase_difference
        self.joints["left_knee"].phase_offset = knee_offset
        self.joints["right_knee"].phase_offset = knee_offset + np.pi

    def _step_once(self, target_velocity, dt):
        self.velocity = target_velocity
        moving = target_velocity != 0.0
        starting = moving and not self.rhythm.is_running
        self.rhythm.step(target_velocity, dt)
        self._update_motion_blend(moving, dt)

        self._update_knee_phase_offsets()
        if starting:
            for joint in self.joints.values():
                joint.reset()
        for name, joint in self.joints.items():
            joint.step(self.rhythm.phase, self.rhythm.frequency, dt)
            gait_angle = self.shapers[name].step(joint.phase)
            self.gait_angles[name] = gait_angle
            standing_angle = self.standing_angles[name]
            self.angles[name] = (
                standing_angle
                + self.motion_blend * (gait_angle - standing_angle)
            )

    def step(self, target_velocity, dt):
        target_velocity = float(target_velocity)
        dt = _check_dt(dt)
        # 训练侧每 0.02 s 刷新一次 OBS，内部按物理步长拆成 4 次更新。
        substeps = max(1, int(np.ceil(dt / self.max_integration_dt)))
        substep_dt = dt / substeps
        for _ in range(substeps):
            self._step_once(target_velocity, substep_dt)

        return self.angles.copy()
