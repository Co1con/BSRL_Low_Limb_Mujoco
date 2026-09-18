# MuJoCo deployment

Run commands from the repository root with the `bipedal_mujoco` Python environment.

## Shared controls and output

Both deployment scripts start the policy immediately with a 0.5 m/s forward command and use the same G1-style sky, checker ground, and lighting. Press **Up/Down** to choose forward-speed levels `[0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]` m/s, **Left/Right** to change yaw command by 0.1 rad/s, **U** to stop, and **R** to reset the robot and controller. The initial command can also be set with `--vx`, `--vy`, and `--yaw`.

Both print target and measured body-frame `vx`, `vy`, and yaw rate every 0.5 s. Closing the viewer opens the same three plots for either robot: `joint_angles` (four CPG hip/knee joints), `motion` (velocity, yaw, height), and `balance` (body roll, pitch, lateral displacement). Add `--plot-dir PATH` to save PNG files, or `--no-plot` to skip interactive plots. `--headless --duration N` runs without the viewer; `--check` validates model loading and one policy inference.

The viewer refreshes at 30 FPS by default. `--render-fps` changes this setting. `--diagnostics` prints viewer sync time, simulation lag, and simulated-time/wall-time rate once per simulated second. If rendering is slow on another machine, try `--render-fps 25` or `20`.

## BSRL

The BSRL code, configuration, CPG implementation, and headless evaluation are entirely in `deploy/BSRL/`. The deployed XML is `assets/BSRL/xml/bsrl_scene.xml`, based on the updated URDF project XML with MuJoCo motors and a collidable floor added.

```powershell
python deploy/BSRL/bsrl_cpg_deploy.py
python deploy/BSRL/evaluate_cpg.py
```

The second command runs the separate fixed-command evaluation and writes results to `logs/cpg_evaluation/` by default. The BSRL simulation uses a 1 ms physics step and a 20 ms policy step. On the local machine, a 3 s viewer test at 30 FPS measured about 10–12 ms per viewer sync and a simulation rate of 0.98–0.99× real time. A 5 s headless trace at 0.5 m/s showed body roll from -2.69° to +2.92° and yaw rate from -0.65 to +0.74 rad/s. Roll changes with the supporting foot, so some visible side-to-side motion is present in the simulated state; `balance.png` separates it from frame stutter.

## G1 29DoF

The G1 code, configuration, and its own CPG implementation are in `deploy/G1/`. It uses `assets/G1/scene_29dof.xml` and the ONNX export under `policy/plane/unitreeG1_cpg_hk_model/`.

```powershell
python deploy/G1/g1_cpg_deploy.py --check
python deploy/G1/g1_cpg_deploy.py --headless --duration 20 --vx 0.5
python deploy/G1/g1_cpg_deploy.py --duration 0 --vx 0.5
```

The G1 controller uses the observation sequence saved in `policy/plane/unitreeG1_cpg_hk_model/params/env.yaml`: body angular velocity, projected gravity, velocity command, joint position offset, joint velocity, joint effort, last action, and CPG rhythm. The controller period is 0.02 s, action scale is 0.25, and PD gains, velocity limits, and effort limits follow that export. `--stand-only` holds the default pose without balance control and is intended for model diagnostics.

The saved `joint_sdk_names` order is different from the PhysX articulation order used for the trained policy's observations and actions. `deploy/G1/g1_cpg_deploy.yaml` now uses the breadth-first joint order of the training USD articulation and maps each element to its MuJoCo joint and motor by name. The 0.5 m/s headless rollout stayed upright for 20 s and traveled 9.22 m; a 1.0 m/s rollout stayed upright for 10 s and traveled 8.97 m. A zero-velocity run stayed in place for 10 s. Use `--diagnostics` to print one-second pose/contact summaries; a headless rollout raises an error if pelvis height drops below 0.5 m. Yaw response and long-run drift have not been calibrated against Isaac Lab.
