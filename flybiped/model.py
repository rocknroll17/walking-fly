"""Build the bipedal fruit-fly MJCF from the original flybody assets.

All six legs, the wings, abdomen and head stay actuated. Only the mouth,
antennae and halteres (irrelevant to locomotion) are frozen. A mocap "goal" marker and a
floor plane are added. The result is written to ``build/biped.xml`` together
with the mesh assets so both MuJoCo (CPU/GPU) and the web viewer can load it.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf

ROOT = Path(__file__).resolve().parents[1]
FLYBODY_XML = ROOT / "ext/flybody/flybody/fruitfly/assets/fruitfly.xml"
BUILD_DIR = ROOT / "build"
BIPED_XML = BUILD_DIR / "biped.xml"

# Physics / control timing (flybody walking defaults).
PHYSICS_DT = 4e-4        # Euler at 4e-4 is NaN-free on Warp (implicitfast at 4e-4 was not); 2x throughput vs 2e-4
CONTROL_DT = 2e-3
JOINT_FILTER = 0.01      # s, first-order filter on position actuators
ADHESION_FILTER = 0.007  # s

WING_FLUIDCOEF = (1.0, 0.5, 1.5, 1.7, 1.0)  # flybody _WING_PARAMS
GOAL_RADIUS = 0.12       # cm, visual sphere radius of the goal marker

FROZEN_PARTS = ("rostrum", "haustellum", "labrum", "antenna", "haltere")


def _has(substrings, name: str) -> bool:
    return any(s in name for s in substrings)


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _joint_springs(xml_path: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Map joint name -> (axis, spring reference angle) from the compiled model."""
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    out = {}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        out[name] = (m.jnt_axis[j].copy(), float(m.qpos_spring[m.jnt_qposadr[j]]))
    return out


def _body_quat_from_springrefs(body, springs) -> np.ndarray:
    """Bake the joint spring-reference angles of ``body`` into its quat.

    This reproduces flybody's leg-retraction trick: joints are removed and the
    body is rotated to the pose the joint springs would hold it at.
    """
    quat = np.array([1.0, 0, 0, 0])
    for joint in reversed(body.joint):
        axis, theta = springs[joint.name]
        quat = _quat_mul(np.hstack((np.cos(theta / 2), np.sin(theta / 2) * axis)), quat)
    if body.quat is not None:
        quat = _quat_mul(body.quat, quat)
    return quat


def _freeze(root, parts, springs) -> None:
    """Remove joints/actuators/tendons/sensors of body parts, baking their pose.

    """
    for body in root.find_all("body"):
        if _has(parts, body.name) and body.joint:
            body.quat = _body_quat_from_springrefs(body, springs)
    for tendon in root.find_all("tendon"):
        if _has(parts, tendon.name):
            act = root.find("actuator", tendon.name)
            if act is not None:
                act.remove()
            tendon.remove()
    for joint in root.find_all("joint"):
        if _has(parts, joint.name):
            act = root.find("actuator", joint.name)
            if act is not None:
                act.remove()
            joint.remove()
    for act in root.find_all("actuator"):
        if _has(parts, act.name):
            act.remove()
    for sensor in root.find_all("sensor"):
        if _has(parts, sensor.name):
            sensor.remove()


def _enable_wing_fluid(root) -> None:
    for geom in root.find_all("geom"):
        if "fluid" in (geom.name or ""):
            geom.fluidshape = "ellipsoid"
            geom.fluidcoef = WING_FLUIDCOEF


def _set_actuator_filters(root) -> None:
    for act in root.find_all("actuator"):
        if act.tag == "adhesion":
            act.dclass.parent.general.dyntype = "filterexact"
            act.dclass.parent.general.dynprm = (ADHESION_FILTER,)
        else:
            act.dyntype = "filterexact"
            act.dynprm = (JOINT_FILTER,)


WING_KP = 0.1     # position-servo gain for the wing joints (dyn*cm/rad); wing spring stiffness is 0.01
WING_KV = 0.004


def _wings_as_position_servos(root) -> None:
    """Turn the flight torque motors on the wing joints into position servos.

    Torque control is right for 200 Hz flapping but hopeless for holding a
    pose: the motors overpower the wing springs ~300x, so any bias pins the
    wing at a joint limit. Position servos let the policy pose the wings
    (folded by default) and use them deliberately.
    """
    for act in root.find_all("actuator"):
        if "wing" not in act.name or act.tag != "general":
            continue
        joint = root.find("joint", act.name)
        rng = joint.range if joint.range is not None else joint.dclass.joint.range
        act.biastype = "affine"
        act.gainprm = (WING_KP,)
        act.biasprm = (0.0, -WING_KP, -WING_KV)
        act.ctrlrange = tuple(rng)


def _add_scene(root) -> None:
    """Floor plane at z=0, a visible goal marker (mocap body), lighting."""
    root.asset.add("texture", name="grid", type="2d", builtin="checker",
                   rgb1=(0.15, 0.2, 0.3), rgb2=(0.2, 0.3, 0.4),
                   width=300, height=300, mark="edge", markrgb=(0.2, 0.3, 0.4))
    root.asset.add("material", name="grid", texture="grid", texrepeat=(1, 1),
                   texuniform=True, reflectance=0.2)
    root.asset.add("texture", name="skybox", type="skybox", builtin="gradient",
                   rgb1=(0.4, 0.6, 0.8), rgb2=(0, 0, 0), width=100, height=100)
    root.worldbody.add("geom", name="floor", type="plane", size=(20, 20, 0.1),
                       material="grid", solref=(0.0002, 1))
    root.worldbody.add("light", name="top", pos=(0, 0, 3), dir=(0, 0, -1),
                       diffuse=(0.6, 0.6, 0.6))
    goal = root.worldbody.add("body", name="goal", mocap=True, pos=(2, 0, 0.3))
    goal.add("geom", name="goal", type="sphere", size=(GOAL_RADIUS,),
             rgba=(1, 0.2, 0.1, 0.6), contype=0, conaffinity=0, group=0)
    goal.add("site", name="goal", size=(0.02,), rgba=(1, 1, 0, 1))

def _add_counterweight(root) -> None:
    pass  # Removed for handstand: natural front-heavy CoM is better for balancing on front legs.


def _set_options(root) -> None:
    root.option.timestep = PHYSICS_DT
    root.option.noslip_iterations = 0  # unsupported on MuJoCo Warp
    root.option.iterations = 30        # Newton iterations: bounded cost on GPU
    root.option.ls_iterations = 15
    root.option.tolerance = 1e-6
    root.size.njmax = None
    root.size.nconmax = None
    root.visual.map.znear = 0.001
    root.visual.map.zfar = 50.0
    root.statistic.extent = 4.0
    root.compiler.boundmass = 0.0
    root.compiler.boundinertia = 0.0


def build(out_dir: Path = BUILD_DIR) -> Path:
    """Create the bipedal fly model and export it to ``out_dir/biped.xml``."""
    root = mjcf.from_path(str(FLYBODY_XML))
    root.model = "flybiped"
    _freeze(root, FROZEN_PARTS, _joint_springs(FLYBODY_XML))
    _enable_wing_fluid(root)
    _wings_as_position_servos(root)
    _set_actuator_filters(root)
    _set_options(root)
    _add_scene(root)
    _add_counterweight(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    mjcf.export_with_assets(root, str(out_dir), "biped.xml")
    return out_dir / "biped.xml"


def load(path: Path = BIPED_XML) -> mujoco.MjModel:
    if not path.exists():
        build(path.parent)
    return mujoco.MjModel.from_xml_path(str(path))


if __name__ == "__main__":
    xml = build()
    m = mujoco.MjModel.from_xml_path(str(xml))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print(f"wrote {xml}: nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} "
          f"nbody={m.nbody} mass={m.body_subtreemass[1]:.2e} g")
    print("actuators:", names)
    print("joints:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
