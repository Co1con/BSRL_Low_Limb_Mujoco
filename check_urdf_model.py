from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import mujoco


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = PROJECT_DIR / "bsrl" / "urdf" / "export_floating.urdf"


def print_model_summary(model: mujoco.MjModel) -> None:
    print("MuJoCo model loaded.")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, ngeom={model.ngeom}, njnt={model.njnt}")
    print("joints:")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qvel_addr = int(model.jnt_dofadr[joint_id])
        print(f"  {joint_id:02d}: {name} type={joint_type} qpos={qpos_addr} qvel={qvel_addr}")


def run_steps(model: mujoco.MjModel, steps: int) -> None:
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    print(f"stepped_time={data.time:.4f}s")
    if model.nq >= 3:
        print(f"base_position=({data.qpos[0]:.4f}, {data.qpos[1]:.4f}, {data.qpos[2]:.4f})")


def show_viewer(model: mujoco.MjModel) -> None:
    from mujoco import viewer

    data = mujoco.MjData(model)
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


def load_urdf_with_ascii_fallback(urdf_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(urdf_path))
    except ValueError as first_error:
        print("Direct load failed. Retrying from an ASCII temp path.")
        print(f"direct_load_error={first_error}")

    with tempfile.TemporaryDirectory(prefix="bsrl_urdf_check_") as temp_dir:
        temp_bsrl_dir = Path(temp_dir) / "bsrl"
        shutil.copytree(urdf_path.parents[1], temp_bsrl_dir)
        temp_urdf_path = temp_bsrl_dir / "urdf" / urdf_path.name
        print(f"Temporary URDF: {temp_urdf_path}")
        return mujoco.MjModel.from_xml_path(str(temp_urdf_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly load and inspect the BSRL URDF in MuJoCo.")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="Path to the URDF file.")
    parser.add_argument("--steps", type=int, default=0, help="Run this many uncontrolled MuJoCo steps after loading.")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer after loading.")
    args = parser.parse_args()

    urdf_path = Path(args.urdf).expanduser().resolve()
    print(f"Loading URDF: {urdf_path}")

    model = load_urdf_with_ascii_fallback(urdf_path)
    print_model_summary(model)

    if args.steps > 0:
        run_steps(model, args.steps)
    if args.viewer:
        show_viewer(model)


if __name__ == "__main__":
    main()
