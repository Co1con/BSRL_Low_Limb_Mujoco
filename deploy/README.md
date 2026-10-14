# MuJoCo deployment

Run commands from the repository root with the `bipedal_mujoco` Python environment.

## Shared controls and output

The G1 HALO script starts at zero command by default. Press **Up/Down** to change signed x velocity, **Left/Right** to change yaw by 0.1 rad/s, **U** to stop, and **R** to reset. The initial command can also be set with `--vx`, `--vy`, and `--yaw`.

Both deployment scripts print target and measured body-frame `vx`, `vy`, and yaw rate every 0.5 s. Closing the viewer opens `joint_angles`, `motion`, and `balance` plots. Add `--plot-dir PATH` to save PNG files, or `--no-plot` to skip interactive plots. `--headless --duration N` runs without the viewer; `--check` validates model loading and one policy inference.

The viewer refreshes at 30 FPS by default. `--render-fps` changes this setting. `--diagnostics` prints viewer sync time, simulation lag, and simulated-time/wall-time rate once per simulated second. If rendering is slow on another machine, try `--render-fps 25` or `20`.

## BSRL

The BSRL code, configuration, CPG implementation, and headless evaluation are entirely in `deploy/BSRL/`. The deployed XML is `assets/BSRL/xml/bsrl_scene.xml`, based on the updated URDF project XML with MuJoCo motors and a collidable floor added.

```powershell
python deploy/BSRL/bsrl_cpg_deploy.py
python deploy/BSRL/evaluate_cpg.py
```

The second command runs the separate fixed-command evaluation and writes results to `logs/cpg_evaluation/` by default. The BSRL simulation uses a 1 ms physics step and a 20 ms policy step. On the local machine, a 3 s viewer test at 30 FPS measured about 10–12 ms per viewer sync and a simulation rate of 0.98–0.99× real time. A 5 s headless trace at 0.5 m/s showed body roll from -2.69° to +2.92° and yaw rate from -0.65 to +0.74 rad/s. Roll changes with the supporting foot, so some visible side-to-side motion is present in the simulated state; `balance.png` separates it from frame stutter.

## G1 29DoF

The G1 HALO deployment is in `deploy/G1/`. It uses `assets/G1/scene_29dof.xml` and the V2 ONNX export synchronized from the PPO project at `policy/plane/unitreeG1_HALO/V2/exported/policy.onnx`.

```powershell
python deploy/G1/g1_halo_deploy.py --check --vx 0.8 --yaw 0.3
python deploy/G1/g1_halo_deploy.py --headless --duration 20 --vx 0.8 --yaw 0.3 --no-plot
python deploy/G1/g1_halo_deploy.py --duration 0 --vx 0.8 --yaw 0.3
python deploy/G1/evaluate_halo.py
```

The G1 controller uses the observation sequence saved in `policy/plane/unitreeG1_HALO/V2/params/env.yaml`: body angular velocity, projected gravity, velocity command, joint position offset, joint velocity, joint effort, last action, and HALO rhythm. The controller period is 0.02 s and action scale is 0.25. Its HALO uses the same yaw gain (0.6), left/right amplitude bounds (0.7–1.3), and 0.1 s amplitude transition as the PPO training code. Left/Right change the yaw command by 0.1 rad/s within ±0.5 rad/s. The policy and HALO both receive the yaw command on each control step. `--stand-only` holds the default pose for model diagnostics.

The saved `joint_sdk_names` order differs from the PhysX articulation order used for policy observations and actions. `deploy/G1/g1_halo_deploy.yaml` uses the training joint order and maps each element to its MuJoCo joint and motor by name. `evaluate_halo.py` runs a forward/backward and left/right turn schedule and writes per-segment yaw and x-velocity errors to `logs/g1_halo_evaluation/`. A headless rollout raises an error if pelvis height drops below 0.5 m. Differences between MuJoCo and Isaac Lab dynamics remain relevant when interpreting tracking quality.

In the local 26 s MuJoCo evaluation, all four `vx=±0.8 m/s`, `yaw=±0.3 rad/s` turn segments remained upright (minimum base height 0.770 m across the schedule). Mean yaw rates had the commanded sign but reached only about ±0.11–0.13 rad/s. This verifies the deployment path and directional response; yaw magnitude tracking still differs from the command.
