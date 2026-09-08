# 并联踝关节运动学与控制

本目录包含独立的 MuJoCo 并联踝关节模型：

- `parallel_ankle.xml`：机构、关节、闭环约束和执行器模型。
- `parallel_ankle_demo.py`：目标轨迹、正逆运动学、PR/AB 控制和结果评估。

## 运行方式

```powershell
python deploy/parallel_ankle/parallel_ankle_demo.py --mode PR
python deploy/parallel_ankle/parallel_ankle_demo.py --mode AB
python deploy/parallel_ankle/parallel_ankle_demo.py --no-view
```

`--mode PR` 使用踝关节角度 PD 控制，`--mode AB` 使用电机角度 PD
控制。不指定 `--mode` 时，依次运行一轮 PR 和一轮 AB。加入
`--no-view` 后不显示 MuJoCo 窗口，仿真以最快速度运行。

每轮包含四个单轴动作和四个双轴耦合动作。每个动作持续 6 秒，从
零位平滑运动到极值，再返回零位：

1. Pitch：+34°；
2. Pitch：-34°；
3. Roll：+30°；
4. Roll：-30°；
5. Pitch +34°、Roll +30°；
6. Pitch +34°、Roll -30°；
7. Pitch -34°、Roll +30°；
8. Pitch -34°、Roll -30°。

运行完成后打印误差指标、显示波形，并将波形导出到
`logs/parallel_ankle/pr_control_evaluation.png` 或
`logs/parallel_ankle/ab_control_evaluation.png`。

## 坐标系与变量

采用右手坐标系：

- +X 指向脚尖；
- +Y 指向机器人左侧；
- +Z 指向上方；
- Pitch 角 `p` 绕 +Y 轴旋转；
- Roll 角 `r` 绕经过 Pitch 旋转后的局部 +X 轴旋转。

对第 `i` 条驱动支链定义：

- `O`：踝关节中心；
- `B_i`：电机转轴中心；
- `C_i`：电机摇臂末端；
- `D_i`：长连杆与足底的连接点；
- `a_i`：电机摇臂长度；
- `L_i`：长连杆长度；
- `alpha_i`：电机角度；
- `d_i`：`D_i` 在足底坐标系中的固定坐标。

足底相对小腿的旋转矩阵为：

```math
R(p,r)=R_y(p)R_x(r)
=
\begin{bmatrix}
\cos p & \sin p\sin r & \sin p\cos r\\
0 & \cos r & -\sin r\\
-\sin p & \cos p\sin r & \cos p\cos r
\end{bmatrix}.
```

足底连接点在小腿坐标系中的位置为：

```math
D_i(p,r)=O+R(p,r)d_i.
```

摇臂中立位置指向 -X，电机绕 Y 轴旋转，因此摇臂末端位置为：

```math
C_i(\alpha_i)=B_i+
\begin{bmatrix}
-a_i\cos\alpha_i\\
0\\
a_i\sin\alpha_i
\end{bmatrix}.
```

长连杆长度保持不变，机构的核心闭环方程为：

```math
\boxed{\left\|D_i(p,r)-C_i(\alpha_i)\right\|=L_i}.
```

两条支链提供两个独立约束：

```math
\begin{cases}
f_1(p,r,\alpha_1)=\|D_1-C_1\|-L_1=0,\\
f_2(p,r,\alpha_2)=\|D_2-C_2\|-L_2=0.
\end{cases}
```

正运动学和逆运动学都以这两个定长约束为基础。

## 正运动学

正运动学已知两个电机实际角度：

```math
\boldsymbol{\alpha}=
\begin{bmatrix}\alpha_1&\alpha_2\end{bmatrix}^{T},
```

求踝关节角度：

```math
\boldsymbol q=
\begin{bmatrix}p&r\end{bmatrix}^{T}.
```

先由电机角度计算 `C_1` 和 `C_2`，然后构造残差：

```math
F(\boldsymbol q)=
\begin{bmatrix}
\|O+R(\boldsymbol q)d_1-C_1\|-L_1\\
\|O+R(\boldsymbol q)d_2-C_2\|-L_2
\end{bmatrix}.
```

正解即寻找满足下式的 `p` 和 `r`：

```math
F(\boldsymbol q)=0.
```

当前 `ForwardKinematics` 使用牛顿迭代：

```math
\boldsymbol q_{k+1}
=\boldsymbol q_k-J_F(\boldsymbol q_k)^{-1}F(\boldsymbol q_k),
```

其中：

```math
J_F=
\frac{\partial(f_1,f_2)}{\partial(p,r)}.
```

残差雅可比使用中心差分计算：

```math
\frac{\partial F}{\partial q_j}
\approx
\frac{F(\boldsymbol q+h\boldsymbol e_j)-F(\boldsymbol q-h\boldsymbol e_j)}{2h},
\qquad h=10^{-6}.
```

为避免牛顿步过大，程序依次尝试以下缩放系数：

```math
\lambda\in\left\{1,\frac12,\frac14,\frac18,\frac1{16},\frac1{32}\right\}.
```

只有新残差变小时才接受该步，同时将结果限制在 Pitch/Roll 关节范围内。
每次求解使用上一次正解作为初值：

```math
\boldsymbol q_0(t)=\boldsymbol q_{FK}(t-\Delta t).
```

这种方式有助于维持解的连续性，并保持在同一机构装配分支上。

## 逆运动学

逆运动学已知 Pitch 和 Roll，分别求两个电机角 `alpha_i`。令：

```math
\boldsymbol v_i=D_i(p,r)-B_i
=\begin{bmatrix}v_{ix}&v_{iy}&v_{iz}\end{bmatrix}^{T}.
```

将摇臂端点代入定长约束并展开：

```math
\boldsymbol v_i^T\boldsymbol v_i+a_i^2
+2a_iv_{ix}\cos\alpha_i
-2a_iv_{iz}\sin\alpha_i=L_i^2.
```

定义：

```math
A_i=-v_{ix},\qquad B'_i=v_{iz},
```

```math
H_i=\frac{\boldsymbol v_i^T\boldsymbol v_i+a_i^2-L_i^2}{2a_i}.
```

原约束可写为：

```math
A_i\cos\alpha_i+B'_i\sin\alpha_i=H_i.
```

进一步定义：

```math
M_i=\sqrt{A_i^2+B_i'^2},\qquad
\delta_i=\operatorname{atan2}(B'_i,A_i),
```

得到：

```math
M_i\cos(\alpha_i-\delta_i)=H_i,
```

所以每条支链的解析逆解为：

```math
\boxed{
\alpha_i=\delta_i\pm\arccos\left(\frac{H_i}{M_i}\right)
}.
```

当前 `InverseKinematics` 选择绝对值较小的候选解，以对应电机零位附近的
装配分支。目标的可达条件为：

```math
\left|\frac{H_i}{M_i}\right|\leq 1.
```

## 传动雅可比

代码使用的传动雅可比定义为：

```math
J(\boldsymbol q)
=\frac{\partial\boldsymbol\alpha}{\partial\boldsymbol q}
=\frac{\partial(\alpha_1,\alpha_2)}{\partial(p,r)}.
```

因此速度关系为：

```math
\dot{\boldsymbol\alpha}=J\dot{\boldsymbol q},
\qquad
\dot{\boldsymbol q}=J^{-1}\dot{\boldsymbol\alpha}.
```

根据虚功一致性：

```math
\boldsymbol\tau_q^T\dot{\boldsymbol q}
=\boldsymbol\tau_\alpha^T\dot{\boldsymbol\alpha},
```

可以得到：

```math
\boldsymbol\tau_q=J^T\boldsymbol\tau_\alpha,
\qquad
\boxed{\boldsymbol\tau_\alpha=J^{-T}\boldsymbol\tau_q}.
```

`InverseKinematics.jacobian()` 通过中心差分计算雅可比：

```math
J_{:,j}\approx
\frac{IK(\boldsymbol q+h\boldsymbol e_j)-IK(\boldsymbol q-h\boldsymbol e_j)}{2h},
\qquad h=10^{-5}.
```

## PD 控制器

两种模式共用同一个 `pdcontroller()`。目标速度统一按零处理，因此控制律为：

```math
\boxed{
\boldsymbol\tau
=K_p(\boldsymbol q_d-\boldsymbol q)-K_d\dot{\boldsymbol q}
}.
```

函数仍保留 `target_velocity` 参数以维持统一调用接口，但该参数不参与计算。
当前默认参数为 `Kp=120`、`Kd=2`，电机输出力矩限制为 ±20 N·m。

## PR：踝关节角度控制

PR 模式的控制流程为：

```text
TargetActions 生成 Pitch/Roll 目标
        ↓
读取两个电机的实际角度和速度
        ↓
ForwardKinematics 由电机角度计算 Pitch/Roll 反馈
        ↓
通过逆解雅可比将电机速度转换为踝关节速度
        ↓
踝关节空间 PD 计算目标 Pitch/Roll 力矩
        ↓
通过逆转置雅可比转换为两个电机力矩
        ↓
MuJoCo 电机执行器
```

角度和速度反馈为：

```math
\boldsymbol q_{FK}=FK(\boldsymbol\alpha),
\qquad
\dot{\boldsymbol q}_{FK}=J^{-1}\dot{\boldsymbol\alpha}.
```

踝关节目标力矩为：

```math
\boldsymbol\tau_q
=K_p(\boldsymbol q_d-\boldsymbol q_{FK})
-K_d\dot{\boldsymbol q}_{FK}.
```

最终电机力矩为：

```math
\boldsymbol\tau_\alpha=J^{-T}\boldsymbol\tau_q.
```

## AB：电机角度控制

AB 模式的控制流程为：

```text
TargetActions 生成 Pitch/Roll 目标
        ↓
InverseKinematics 计算两个电机目标角度
        ↓
读取两个电机的实际角度和速度
        ↓
电机空间 PD 计算两个电机目标力矩
        ↓
MuJoCo 电机执行器
```

电机目标角度为：

```math
\boldsymbol\alpha_d=IK(\boldsymbol q_d).
```

电机目标力矩为：

```math
\boldsymbol\tau_\alpha
=K_p(\boldsymbol\alpha_d-\boldsymbol\alpha)
-K_d\dot{\boldsymbol\alpha}.
```

## Evaluation 评估内容

`Evaluation` 以 100 Hz 在内存中记录：

- Pitch/Roll 目标角度；
- 正运动学反馈角度；
- MuJoCo Pitch/Roll 关节角度；
- 两个电机的逆解目标角度和实际角度；
- 两个电机的输出力矩；
- 正运动学连杆长度残差；
- 传动雅可比条件数。

每轮结束后打印以下指标：

- Pitch/Roll 跟踪 RMSE；
- Pitch/Roll 最大跟踪误差；
- 正运动学与 MuJoCo 关节角度的最大差值；
- 两个电机的峰值力矩；
- 最大正运动学残差；
- 最大雅可比条件数。

波形图包含目标、正解反馈、MuJoCo 结果、跟踪误差、电机角度和电机力矩。
