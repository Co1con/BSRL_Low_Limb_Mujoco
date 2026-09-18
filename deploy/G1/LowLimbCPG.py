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
        self.frequency_slope = 0.48749031
        self.frequency_intercept = 0.60252246
        self.reset()

    def reset(self):
        # Match Isaac Lab: reset to phase zero; choose the leading leg on start.
        self.reset_phase = 0.0
        radius = np.sqrt(self.mu)
        self.x = float(radius * np.cos(self.reset_phase))
        self.y = float(radius * np.sin(self.reset_phase))
        self.omega = 0.0
        self.is_running = False
        self.stop_requested = False

    @property
    def phase(self):
        return float(np.mod(np.arctan2(self.y, self.x), TWO_PI))

    @property
    def frequency(self):
        return self.omega / TWO_PI

    def velocity_to_frequency(self, velocity):
        velocity = float(velocity)
        if velocity <= 0.0:
            return 0.0
        return self.frequency_slope * velocity + self.frequency_intercept

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

        # 正速度立即更新节律；零速度只提出停车请求。
        if velocity > 0.0:
            if not self.is_running:
                self.reset_phase = float(np.random.choice((0.0, np.pi)))
                radius = np.sqrt(self.mu)
                self.x = float(radius * np.cos(self.reset_phase))
                self.y = float(radius * np.sin(self.reset_phase))
            self.omega = TWO_PI * self.velocity_to_frequency(velocity)
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
        if rhythm_frequency < 0.0:
            raise ValueError("rhythm_frequency must be greater than or equal to zero")

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
        if rhythm_frequency > 0.0:
            target_omega = max(0.0, target_omega)

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

        self.phase_speed_slope = -0.3307487167
        self.phase_speed_intercept = 5.2494736659
        self.max_integration_dt = 0.005

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
        self.hip_knee_phase_difference = initial_phase_difference
        self.angles = {name: 0.0 for name in self.joints}

    def reset(self):
        """与训练环境的 episode reset 一致，清除全部振荡器状态。"""
        self.rhythm.reset()
        self.velocity = 0.0
        self.hip_knee_phase_difference = self.phase_speed_intercept
        self.joints["left_hip"].phase_offset = 0.0
        self.joints["right_hip"].phase_offset = np.pi
        self.joints["left_knee"].phase_offset = -self.hip_knee_phase_difference
        self.joints["right_knee"].phase_offset = -self.hip_knee_phase_difference + np.pi
        for name, joint in self.joints.items():
            joint.reset()
            self.shapers[name].angle = 0.0
            self.angles[name] = 0.0

    def _update_knee_phase_offsets(self):
        self.hip_knee_phase_difference = (
            self.phase_speed_slope * self.velocity
            + self.phase_speed_intercept
        )
        knee_offset = -self.hip_knee_phase_difference
        self.joints["left_knee"].phase_offset = knee_offset
        self.joints["right_knee"].phase_offset = knee_offset + np.pi

    def _step_once(self, target_velocity, dt):
        self.velocity = target_velocity
        self.rhythm.step(target_velocity, dt)

        self._update_knee_phase_offsets()
        for name, joint in self.joints.items():
            joint.step(self.rhythm.phase, self.rhythm.frequency, dt)
            self.angles[name] = self.shapers[name].step(joint.phase)

    def step(self, target_velocity, dt):
        target_velocity = float(target_velocity)
        dt = _check_dt(dt)
        if target_velocity < 0.0:
            raise ValueError("target_velocity must be greater than or equal to zero")

        # 训练侧每 0.02 s 刷新一次 OBS，内部按物理步长拆成 4 次更新。
        substeps = max(1, int(np.ceil(dt / self.max_integration_dt)))
        substep_dt = dt / substeps
        for _ in range(substeps):
            self._step_once(target_velocity, substep_dt)

        return self.angles.copy()
