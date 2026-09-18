# CPG 无界面评估

在项目根目录运行：

```powershell
& 'C:\Users\xie23\.conda\envs\bipedal_mujoco\python.exe' deploy/BSRL/evaluate_cpg.py
```

脚本使用 `deploy/BSRL/bsrl_cpg_deploy.yaml` 中的 MuJoCo 模型、ONNX 策略、PD 参数和 CPG 实现；不启动 viewer。默认指令序列依次为：静止 2 秒、0.5 m/s 前进 4 秒、1.0 m/s 前进 4 秒、静止 2 秒、0.8 m/s 且左右偏航各 4 秒、最后静止 3 秒。固定随机种子为 0。

输出位于 `logs/cpg_evaluation/`：

- `trace.csv`：每个策略控制周期记录一次指令、机体系实际速度、四个髋膝关节的 CPG 参考角和实际角、左右足触地、机身高度/姿态、CPG 节律和关节力矩 RMS。
- `summary.json`：整体与分段指标，并保存种子、配置路径和指令序列。
- `joint_trajectories.png`、`motion_and_contacts.png`、`balance_and_effort.png`：对应的时序图。图中不包含 policy target。

足端接触在足部刚体与 `floor` 几何体发生接触时记为 1，否则为 0。实际线速度与角速度均为机体系；每个样本在执行完该控制周期的 MuJoCo 物理步后记录。

可用 `--schedule path/to/schedule.yaml` 提供自定义速度序列，文件内容是若干 `[持续秒数, 前进速度m/s, 偏航角速度rad/s]`，例如：

```yaml
- [2.0, 0.0, 0.0]
- [4.0, 0.5, 0.0]
- [4.0, 0.8, 0.25]
- [2.0, 0.0, 0.0]
```

其他参数可通过 `--config`、`--output-dir` 和 `--seed` 指定。


