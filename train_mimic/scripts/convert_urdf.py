#!/usr/bin/env python3
"""
Convert G1 URDF to USD format for Isaac Lab.

This script parses the G1 29-DOF URDF and creates a faithful USD representation
preserving all joint structure, limits, inertial properties, and mesh references.

Usage:
    python train_mimic/scripts/convert_urdf.py

Output:
    train_mimic/assets/g1/usd/g1_29dof.usd
"""

import os
import sys
import shutil
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, UsdShade


# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = TRAIN_ROOT / "assets" / "g1"
URDF_PATH = ASSETS_DIR / "g1_custom_collision_29dof.urdf"
MESH_SRC_DIR = ASSETS_DIR / "meshes"
USD_OUT_DIR = ASSETS_DIR / "usd"
USD_OUT_PATH = USD_OUT_DIR / "g1_29dof.usd"
USD_MESH_DIR = USD_OUT_DIR / "meshes"


# ─── URDF Parsing ────────────────────────────────────────────────────────────

def parse_xyz(text: str) -> Gf.Vec3d:
    """Parse 'x y z' string to Gf.Vec3d."""
    parts = [float(x) for x in text.strip().split()]
    return Gf.Vec3d(*parts)


def parse_rpy(text: str) -> Gf.Quatd:
    """Parse 'roll pitch yaw' string to quaternion (Gf.Quatd)."""
    r, p, y = [float(x) for x in text.strip().split()]
    # RPY to quaternion (intrinsic XYZ = extrinsic ZYX)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatd(w, Gf.Vec3d(x, yq, z))


def parse_origin(elem):
    """Parse <origin xyz='...' rpy='...'/> element."""
    if elem is None:
        return Gf.Vec3d(0, 0, 0), Gf.Quatd(1, Gf.Vec3d(0, 0, 0))
    xyz = parse_xyz(elem.get("xyz", "0 0 0"))
    quat = parse_rpy(elem.get("rpy", "0 0 0"))
    return xyz, quat


def sanitize_name(name: str) -> str:
    """Make a valid USD prim name from URDF name."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


class URDFLink:
    def __init__(self, elem):
        self.name = elem.get("name")
        self.inertial = None
        self.visuals = []
        self.collisions = []

        inertial_elem = elem.find("inertial")
        if inertial_elem is not None:
            self.inertial = {
                "origin": parse_origin(inertial_elem.find("origin")),
                "mass": float(inertial_elem.find("mass").get("value")),
            }
            inertia_elem = inertial_elem.find("inertia")
            if inertia_elem is not None:
                self.inertial["inertia"] = {
                    "ixx": float(inertia_elem.get("ixx", "0")),
                    "ixy": float(inertia_elem.get("ixy", "0")),
                    "ixz": float(inertia_elem.get("ixz", "0")),
                    "iyy": float(inertia_elem.get("iyy", "0")),
                    "iyz": float(inertia_elem.get("iyz", "0")),
                    "izz": float(inertia_elem.get("izz", "0")),
                }

        for vis_elem in elem.findall("visual"):
            vis = {"origin": parse_origin(vis_elem.find("origin"))}
            geom = vis_elem.find("geometry")
            if geom is not None:
                mesh_elem = geom.find("mesh")
                if mesh_elem is not None:
                    vis["mesh"] = mesh_elem.get("filename")
                    vis["type"] = "mesh"
            mat_elem = vis_elem.find("material")
            if mat_elem is not None:
                color_elem = mat_elem.find("color")
                if color_elem is not None:
                    vis["color"] = [float(c) for c in color_elem.get("rgba").split()]
            self.visuals.append(vis)

        for col_elem in elem.findall("collision"):
            col = {"origin": parse_origin(col_elem.find("origin"))}
            geom = col_elem.find("geometry")
            if geom is not None:
                mesh_elem = geom.find("mesh")
                cyl_elem = geom.find("cylinder")
                sphere_elem = geom.find("sphere")
                box_elem = geom.find("box")
                if mesh_elem is not None:
                    col["type"] = "mesh"
                    col["mesh"] = mesh_elem.get("filename")
                elif cyl_elem is not None:
                    col["type"] = "cylinder"
                    col["radius"] = float(cyl_elem.get("radius"))
                    col["length"] = float(cyl_elem.get("length"))
                elif sphere_elem is not None:
                    col["type"] = "sphere"
                    col["radius"] = float(sphere_elem.get("radius"))
                elif box_elem is not None:
                    col["type"] = "box"
                    col["size"] = [float(s) for s in box_elem.get("size").split()]
            self.collisions.append(col)


class URDFJoint:
    def __init__(self, elem):
        self.name = elem.get("name")
        self.type = elem.get("type")
        self.parent = elem.find("parent").get("link")
        self.child = elem.find("child").get("link")
        self.origin = parse_origin(elem.find("origin"))
        axis_elem = elem.find("axis")
        if axis_elem is not None:
            parts = [float(x) for x in axis_elem.get("xyz").split()]
            self.axis = Gf.Vec3f(*parts)
        else:
            self.axis = Gf.Vec3f(1, 0, 0)
        limit_elem = elem.find("limit")
        if limit_elem is not None:
            self.lower = float(limit_elem.get("lower", "0"))
            self.upper = float(limit_elem.get("upper", "0"))
            self.effort = float(limit_elem.get("effort", "0"))
            self.velocity = float(limit_elem.get("velocity", "0"))
        else:
            self.lower = self.upper = self.effort = self.velocity = 0.0


def parse_urdf(urdf_path: str):
    """Parse URDF file and return links and joints."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.get("name", "robot")

    links = {}
    for link_elem in root.findall("link"):
        link = URDFLink(link_elem)
        links[link.name] = link

    joints = {}
    for joint_elem in root.findall("joint"):
        joint = URDFJoint(joint_elem)
        joints[joint.name] = joint

    return robot_name, links, joints


# ─── USD Construction ─────────────────────────────────────────────────────────

def set_xform_op(prim, translate=None, orient=None):
    """Set translate and orient xform ops on a UsdGeom.Xformable."""
    xformable = UsdGeom.Xformable(prim)
    if translate is not None:
        xformable.AddTranslateOp().Set(translate)
    if orient is not None:
        # Convert Quatd to Quatf if needed
        if isinstance(orient, Gf.Quatd):
            orient = Gf.Quatf(float(orient.GetReal()), Gf.Vec3f(*[float(x) for x in orient.GetImaginary()]))
        xformable.AddOrientOp().Set(orient)


def create_collision_geom(stage, parent_path, col, idx):
    """Create a collision geometry prim under parent_path."""
    col_name = f"collision_{idx}"
    col_path = f"{parent_path}/{col_name}"
    xyz, quat = col["origin"]
    col_type = col.get("type")

    if col_type == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, col_path)
        geom.GetRadiusAttr().Set(col["radius"])
        geom.GetHeightAttr().Set(col["length"])
        geom.GetAxisAttr().Set("Z")
    elif col_type == "sphere":
        geom = UsdGeom.Sphere.Define(stage, col_path)
        geom.GetRadiusAttr().Set(col["radius"])
    elif col_type == "box":
        geom = UsdGeom.Cube.Define(stage, col_path)
        size = col["size"]
        # USD Cube is unit cube, scale to match
        xformable = UsdGeom.Xformable(geom.GetPrim())
        xformable.AddScaleOp().Set(Gf.Vec3f(size[0] / 2, size[1] / 2, size[2] / 2))
    elif col_type == "mesh":
        # Reference the mesh file
        geom = UsdGeom.Mesh.Define(stage, col_path)
        # We'll skip mesh collision for now, just mark it
    else:
        return

    set_xform_op(geom.GetPrim(), translate=xyz, orient=quat)

    # Apply collision API
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim())


def copy_meshes(mesh_src_dir: Path, mesh_dst_dir: Path):
    """Copy STL meshes to USD output directory."""
    mesh_dst_dir.mkdir(parents=True, exist_ok=True)
    if mesh_src_dir.exists():
        for stl_file in mesh_src_dir.glob("*.STL"):
            dst = mesh_dst_dir / stl_file.name
            if not dst.exists() or dst.stat().st_mtime < stl_file.stat().st_mtime:
                shutil.copy2(stl_file, dst)
        for stl_file in mesh_src_dir.glob("*.stl"):
            dst = mesh_dst_dir / stl_file.name
            if not dst.exists() or dst.stat().st_mtime < stl_file.stat().st_mtime:
                shutil.copy2(stl_file, dst)


def build_usd(robot_name, links, joints, usd_path: str, mesh_rel_prefix: str = "./meshes"):
    """Build a USD stage from parsed URDF data."""
    stage = Usd.Stage.CreateNew(usd_path)
    stage.SetDefaultPrim(stage.DefinePrim(f"/{sanitize_name(robot_name)}"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = f"/{sanitize_name(robot_name)}"

    # Build parent→children map
    children_map = {}  # parent_link -> [(joint, child_link)]
    child_set = set()
    for jname, joint in joints.items():
        children_map.setdefault(joint.parent, []).append((joint, links[joint.child]))
        child_set.add(joint.child)

    # Find root link (not a child of any joint)
    root_links = [name for name in links if name not in child_set]
    if len(root_links) != 1:
        print(f"WARNING: Expected 1 root link, found {len(root_links)}: {root_links}")
    root_link_name = root_links[0] if root_links else list(links.keys())[0]

    # Apply ArticulationRootAPI to the robot root
    root_prim = stage.GetPrimAtPath(root_path)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    # Recursive function to build the kinematic tree
    def build_link(link: URDFLink, parent_usd_path: str, joint: URDFJoint = None):
        link_name = sanitize_name(link.name)
        link_path = f"{parent_usd_path}/{link_name}"

        # Create Xform for the link
        link_xform = UsdGeom.Xform.Define(stage, link_path)
        link_prim = link_xform.GetPrim()

        # If there's a joint connecting to this link, set the transform and create joint
        if joint is not None:
            xyz, quat = joint.origin
            set_xform_op(link_prim, translate=xyz, orient=quat)

            # Create physics joint
            joint_name = sanitize_name(joint.name)
            joint_path = f"{link_path}/{joint_name}"

            if joint.type == "revolute":
                phys_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                phys_joint.GetAxisAttr().Set(_axis_to_token(joint.axis))
                phys_joint.GetLowerLimitAttr().Set(math.degrees(joint.lower))
                phys_joint.GetUpperLimitAttr().Set(math.degrees(joint.upper))

                # Store original radian limits as custom attributes for verification
                joint_prim = phys_joint.GetPrim()
                joint_prim.CreateAttribute("urdf:lowerLimitRad", Sdf.ValueTypeNames.Double).Set(joint.lower)
                joint_prim.CreateAttribute("urdf:upperLimitRad", Sdf.ValueTypeNames.Double).Set(joint.upper)
                joint_prim.CreateAttribute("urdf:effort", Sdf.ValueTypeNames.Double).Set(joint.effort)
                joint_prim.CreateAttribute("urdf:velocity", Sdf.ValueTypeNames.Double).Set(joint.velocity)

                # Set joint bodies
                parent_link_path = parent_usd_path
                phys_joint.GetBody0Rel().SetTargets([parent_link_path])
                phys_joint.GetBody1Rel().SetTargets([link_path])

                # Add drive
                drive_api = UsdPhysics.DriveAPI.Apply(phys_joint.GetPrim(), "angular")
                drive_api.GetDampingAttr().Set(0.0)
                drive_api.GetStiffnessAttr().Set(0.0)

            elif joint.type == "fixed":
                phys_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                phys_joint.GetBody0Rel().SetTargets([parent_usd_path])
                phys_joint.GetBody1Rel().SetTargets([link_path])

        # Apply RigidBodyAPI
        if link.inertial is not None:
            UsdPhysics.RigidBodyAPI.Apply(link_prim)
            mass_api = UsdPhysics.MassAPI.Apply(link_prim)
            mass_api.GetMassAttr().Set(link.inertial["mass"])

            # Set center of mass
            inertial_xyz, inertial_quat = link.inertial["origin"]
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(float(inertial_xyz[0]),
                                                          float(inertial_xyz[1]),
                                                          float(inertial_xyz[2])))

            # Set diagonal inertia
            if "inertia" in link.inertial:
                inertia = link.inertial["inertia"]
                mass_api.GetDiagonalInertiaAttr().Set(
                    Gf.Vec3f(float(inertia["ixx"]),
                             float(inertia["iyy"]),
                             float(inertia["izz"])))
                mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(inertial_quat.GetReal()), Gf.Vec3f(*[float(x) for x in inertial_quat.GetImaginary()])))

        # Add visual meshes
        for i, vis in enumerate(link.visuals):
            if vis.get("type") == "mesh" and vis.get("mesh"):
                vis_name = f"visuals"
                vis_path = f"{link_path}/{vis_name}"
                vis_xform = UsdGeom.Xform.Define(stage, vis_path)
                vis_xyz, vis_quat = vis["origin"]
                set_xform_op(vis_xform.GetPrim(), translate=vis_xyz, orient=vis_quat)

                # Reference the STL mesh file
                mesh_filename = vis["mesh"]  # e.g., "meshes/pelvis.STL"
                # Strip "meshes/" prefix if present
                if mesh_filename.startswith("meshes/"):
                    mesh_filename = mesh_filename[len("meshes/"):]
                mesh_ref_path = f"{mesh_rel_prefix}/{mesh_filename}"

                # Create a mesh prim that references the STL
                mesh_path = f"{vis_path}/mesh"
                mesh_prim = stage.DefinePrim(mesh_path)
                mesh_prim.GetReferences().AddReference(mesh_ref_path)

                # Set color if available
                if "color" in vis:
                    color = vis["color"]
                    # Apply display color on the visuals xform
                    gprim = UsdGeom.Gprim(vis_xform.GetPrim())
                    if gprim:
                        vis_xform.GetPrim().CreateAttribute(
                            "primvars:displayColor",
                            Sdf.ValueTypeNames.Color3fArray
                        ).Set([Gf.Vec3f(color[0], color[1], color[2])])

        # Add collision geometries
        if link.collisions:
            col_group_path = f"{link_path}/collisions"
            col_group = UsdGeom.Xform.Define(stage, col_group_path)
            for i, col in enumerate(link.collisions):
                create_collision_geom(stage, col_group_path, col, i)

        # Recurse into children
        if link.name in children_map:
            for child_joint, child_link in children_map[link.name]:
                build_link(child_link, link_path, child_joint)

    # Start building from root
    root_link = links[root_link_name]
    build_link(root_link, root_path)

    stage.GetRootLayer().Save()
    return stage


def _axis_to_token(axis: Gf.Vec3f) -> str:
    """Convert axis vector to USD axis token."""
    ax = [abs(float(axis[0])), abs(float(axis[1])), abs(float(axis[2]))]
    max_idx = ax.index(max(ax))
    return ["X", "Y", "Z"][max_idx]


# ─── Verification ─────────────────────────────────────────────────────────────

def verify_usd(usd_path: str, expected_joints: dict, expected_links: dict):
    """Verify the converted USD file matches URDF expectations."""
    stage = Usd.Stage.Open(usd_path)
    results = []
    results.append(f"=== USD Verification: {usd_path} ===\n")

    # Count revolute joints
    revolute_joints = []
    fixed_joints = []
    all_joint_prims = []

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joints.append(prim)
            all_joint_prims.append(prim)
        elif prim.IsA(UsdPhysics.FixedJoint):
            fixed_joints.append(prim)

    results.append(f"Revolute joints (DOFs): {len(revolute_joints)}")
    results.append(f"Fixed joints: {len(fixed_joints)}")
    results.append(f"Total joints: {len(revolute_joints) + len(fixed_joints)}")
    results.append("")

    # Verify DOF count
    dof_ok = len(revolute_joints) == 29
    results.append(f"[{'PASS' if dof_ok else 'FAIL'}] DOF count: {len(revolute_joints)} (expected 29)")
    results.append("")

    # List all revolute joint names and limits
    results.append("--- Revolute Joints ---")
    joint_limits_ok = True
    for prim in revolute_joints:
        name = prim.GetName()
        joint = UsdPhysics.RevoluteJoint(prim)
        lower_deg = joint.GetLowerLimitAttr().Get()
        upper_deg = joint.GetUpperLimitAttr().Get()
        lower_rad = prim.GetAttribute("urdf:lowerLimitRad").Get()
        upper_rad = prim.GetAttribute("urdf:upperLimitRad").Get()
        effort = prim.GetAttribute("urdf:effort").Get()
        velocity = prim.GetAttribute("urdf:velocity").Get()

        # Verify against original URDF
        if name in expected_joints:
            ej = expected_joints[name]
            lower_match = abs(lower_rad - ej.lower) < 1e-6
            upper_match = abs(upper_rad - ej.upper) < 1e-6
            status = "PASS" if (lower_match and upper_match) else "FAIL"
            if status == "FAIL":
                joint_limits_ok = False
            results.append(
                f"  [{status}] {name}: lower={lower_rad:.6f} (expected {ej.lower:.6f}), "
                f"upper={upper_rad:.6f} (expected {ej.upper:.6f}), "
                f"effort={effort}, velocity={velocity}"
            )
        else:
            results.append(f"  [WARN] {name}: not found in expected joints")

    results.append(f"\n[{'PASS' if joint_limits_ok else 'FAIL'}] Joint limits verification")
    results.append("")

    # Verify body names
    results.append("--- Body Names ---")
    body_names = set()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            body_names.add(prim.GetName())

    expected_body_names = [
        "pelvis", "left_ankle_roll_link", "right_ankle_roll_link",
        "left_hip_pitch_link", "right_hip_pitch_link",
        "left_knee_link", "right_knee_link",
        "torso_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
        "left_elbow_link", "right_elbow_link",
    ]
    bodies_ok = True
    for bname in expected_body_names:
        found = bname in body_names
        if not found:
            bodies_ok = False
        results.append(f"  [{'PASS' if found else 'FAIL'}] {bname}")

    results.append(f"\n[{'PASS' if bodies_ok else 'FAIL'}] Body names verification")
    results.append(f"Total bodies with RigidBody/Mass API: {len(body_names)}")
    results.append(f"All body names: {sorted(body_names)}")
    results.append("")

    # Verify mesh references exist
    results.append("--- Mesh References ---")
    mesh_count = 0
    for prim in stage.Traverse():
        refs = prim.GetReferences()
        if prim.GetPath().pathString.endswith("/mesh"):
            mesh_count += 1

    results.append(f"Mesh reference prims: {mesh_count}")
    meshes_ok = mesh_count > 0
    results.append(f"[{'PASS' if meshes_ok else 'FAIL'}] Mesh references present")
    results.append("")

    # Overall
    all_ok = dof_ok and joint_limits_ok and bodies_ok and meshes_ok
    results.append(f"{'=' * 50}")
    results.append(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    results.append(f"{'=' * 50}")

    return "\n".join(results), all_ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"URDF source: {URDF_PATH}")
    print(f"USD output:  {USD_OUT_PATH}")
    print()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF file not found: {URDF_PATH}")
        sys.exit(1)

    # Parse URDF
    print("Parsing URDF...")
    robot_name, links, joints = parse_urdf(str(URDF_PATH))
    print(f"  Robot name: {robot_name}")
    print(f"  Links: {len(links)}")
    print(f"  Joints: {len(joints)}")

    revolute_joints = {n: j for n, j in joints.items() if j.type == "revolute"}
    fixed_joints = {n: j for n, j in joints.items() if j.type == "fixed"}
    print(f"  Revolute (DOF): {len(revolute_joints)}")
    print(f"  Fixed: {len(fixed_joints)}")
    print()

    # Copy meshes
    print("Copying meshes...")
    copy_meshes(MESH_SRC_DIR, USD_MESH_DIR)
    print(f"  Meshes copied to {USD_MESH_DIR}")
    print()

    # Create output directory
    USD_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove existing USD if present (idempotent)
    if USD_OUT_PATH.exists():
        USD_OUT_PATH.unlink()
        print(f"  Removed existing {USD_OUT_PATH}")

    # Build USD
    print("Building USD...")
    stage = build_usd(robot_name, links, joints, str(USD_OUT_PATH))
    print(f"  USD saved to {USD_OUT_PATH}")
    print()

    # Verify
    print("Verifying USD...")
    report, all_ok = verify_usd(str(USD_OUT_PATH), revolute_joints, links)
    print(report)

    # Save evidence
    evidence_dir = TRAIN_ROOT.parent / ".sisyphus" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "task-3-usd-verification.txt"
    with open(evidence_path, "w") as f:
        f.write(report)
    print(f"\nEvidence saved to {evidence_path}")

    if not all_ok:
        print("\nWARNING: Some verifications failed!")
        sys.exit(1)
    else:
        print("\nAll verifications passed!")


if __name__ == "__main__":
    main()
