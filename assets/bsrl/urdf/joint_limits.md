# BSRL Joint Limits

Source: `urdf/export.urdf`

Motor limits use gearbox-output values from the supplied SETZ datasheet:

| Motor | Joints | Ratio | Rated torque | Peak torque | Rated speed | Peak speed | Rated / peak power |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| SETZ70-1EB-J (small) | hip yaw, ankle pitch, ankle roll | 16 | 20 N*m | 112 N*m | 14.66 rad/s | 26.18 rad/s | 300 / 1250 W |
| SETZ90-1FF-J (large) | hip roll, hip pitch, knee pitch | 18 | 45 N*m | 216 N*m | 9.32 rad/s | 15.13 rad/s | 400 / 1050 W |

The URDF table below records peak torque and peak speed as independent hard limits. They are not simultaneously available; downstream actuator models must also enforce the peak-power envelope.

| Joint Name | Type | Position Lower (rad) | Position Upper (rad) | Position Lower (deg) | Position Upper (deg) | Velocity Limit (rad/s) | Torque Limit (N*m) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `joint_right_hip_yaw` | revolute | -0.4000 | 0.6000 | -22.92 | 34.38 | 26.18 | 112.00 |
| `joint_right_hip_roll` | revolute | -0.3000 | 0.3000 | -17.19 | 17.19 | 15.13 | 216.00 |
| `joint_right_hip_pitch` | revolute | -1.5000 | 0.3000 | -85.94 | 17.19 | 15.13 | 216.00 |
| `joint_right_knee_pitch` | revolute | -0.1000 | 2.2000 | -5.73 | 126.05 | 15.13 | 216.00 |
| `joint_right_ankle_pitch` | revolute | -0.6000 | 0.6000 | -34.38 | 34.38 | 26.18 | 112.00 |
| `joint_right_ankle_roll` | revolute | -0.5200 | 0.5200 | -29.79 | 29.79 | 26.18 | 112.00 |
| `joint_left_hip_yaw` | revolute | -0.6000 | 0.4000 | -34.38 | 22.92 | 26.18 | 112.00 |
| `joint_left_hip_roll` | revolute | -0.3000 | 0.3000 | -17.19 | 17.19 | 15.13 | 216.00 |
| `joint_left_hip_pitch` | revolute | -1.5000 | 0.3000 | -85.94 | 17.19 | 15.13 | 216.00 |
| `joint_left_knee_pitch` | revolute | -0.1000 | 2.2000 | -5.73 | 126.05 | 15.13 | 216.00 |
| `joint_left_ankle_pitch` | revolute | -0.6000 | 0.6000 | -34.38 | 34.38 | 26.18 | 112.00 |
| `joint_left_ankle_roll` | revolute | -0.5200 | 0.5200 | -29.79 | 29.79 | 26.18 | 112.00 |


存在问题
1. 足底碰撞下移加宽14mm urdf里已经带了
2. 去掉电路板建模
3. 调整奖励和观测 ？是否加 estimator ？ 
4. 限位是否需要再调整
--5. 足底碰撞等高是否需要 --