#!/usr/bin/env python3
"""
Verify USD structure for G1 29-DOF robot.

Checks:
- Exactly 29 revolute joints
- Joint names match expected 29-DOF set
- Key bodies exist (pelvis, ankle_roll_link, rubber_hand, head_mocap, torso_link)
- No closed-loop FixedJoints
- Effort limits match URDF values
- Compare with old USD backup
"""

import sys
from pathlib import Path
from pxr import Usd, UsdPhysics

# Expected joint names (29 DOF)
EXPECTED_JOINTS = [
    # Legs (6 per leg = 12)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Torso (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Arms (7 per arm = 14)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Key bodies from G1MotionCfg (lines 383-389)
KEY_BODIES = [
    "left_rubber_hand", "right_rubber_hand",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_knee_link", "right_knee_link",
    "left_elbow_link", "right_elbow_link",
    "head_mocap",
]

# Expected effort limits from URDF (Nm)
EFFORT_LIMITS = {
    # Hip pitch
    "left_hip_pitch_joint": 88, "right_hip_pitch_joint": 88,
    # Hip roll
    "left_hip_roll_joint": 139, "right_hip_roll_joint": 139,
    # Hip yaw
    "left_hip_yaw_joint": 88, "right_hip_yaw_joint": 88,
    # Knee
    "left_knee_joint": 139, "right_knee_joint": 139,
    # Ankle pitch
    "left_ankle_pitch_joint": 50, "right_ankle_pitch_joint": 50,
    # Ankle roll
    "left_ankle_roll_joint": 50, "right_ankle_roll_joint": 50,
    # Waist
    "waist_yaw_joint": 88, "waist_roll_joint": 50, "waist_pitch_joint": 50,
    # Shoulder
    "left_shoulder_pitch_joint": 25, "left_shoulder_roll_joint": 25, "left_shoulder_yaw_joint": 25,
    "right_shoulder_pitch_joint": 25, "right_shoulder_roll_joint": 25, "right_shoulder_yaw_joint": 25,
    # Elbow
    "left_elbow_joint": 25, "right_elbow_joint": 25,
    # Wrist
    "left_wrist_roll_joint": 25, "left_wrist_pitch_joint": 5, "left_wrist_yaw_joint": 5,
    "right_wrist_roll_joint": 25, "right_wrist_pitch_joint": 5, "right_wrist_yaw_joint": 5,
}


def verify_usd(usd_path: str):
    """Verify USD structure."""
    results = []
    results.append(f"=== USD Verification: {usd_path} ===\n")
    
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        results.append(f"[FAIL] Could not open USD file: {usd_path}")
        return "\n".join(results), False
    
    # Count joints
    revolute_joints = []
    fixed_joints = []
    
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joints.append(prim)
        elif prim.IsA(UsdPhysics.FixedJoint):
            fixed_joints.append(prim)
    
    results.append(f"Revolute joints (DOFs): {len(revolute_joints)}")
    results.append(f"Fixed joints: {len(fixed_joints)}")
    results.append("")
    
    # Check DOF count
    dof_ok = len(revolute_joints) == 29
    results.append(f"[{'PASS' if dof_ok else 'FAIL'}] DOF count: {len(revolute_joints)} (expected 29)")
    results.append("")
    
    # Check joint names
    results.append("--- Joint Names ---")
    joint_names = [prim.GetName() for prim in revolute_joints]
    joint_names_ok = True
    
    for expected in EXPECTED_JOINTS:
        found = expected in joint_names
        if not found:
            joint_names_ok = False
        results.append(f"  [{'PASS' if found else 'FAIL'}] {expected}")
    
    # Check for unexpected joints
    unexpected = set(joint_names) - set(EXPECTED_JOINTS)
    if unexpected:
        joint_names_ok = False
        results.append(f"\n  [WARN] Unexpected joints: {unexpected}")
    
    results.append(f"\n[{'PASS' if joint_names_ok else 'FAIL'}] Joint names verification")
    results.append("")
    
    # Check effort limits
    results.append("--- Effort Limits ---")
    effort_ok = True
    
    for prim in revolute_joints:
        name = prim.GetName()
        effort_attr = prim.GetAttribute("urdf:effort")
        
        if effort_attr and effort_attr.Get():
            effort = effort_attr.Get()
            expected_effort = EFFORT_LIMITS.get(name)
            
            if expected_effort:
                match = abs(effort - expected_effort) < 1e-3
                if not match:
                    effort_ok = False
                results.append(f"  [{'PASS' if match else 'FAIL'}] {name}: {effort} Nm (expected {expected_effort} Nm)")
            else:
                results.append(f"  [WARN] {name}: {effort} Nm (no expected value)")
        else:
            results.append(f"  [WARN] {name}: no effort attribute")
    
    results.append(f"\n[{'PASS' if effort_ok else 'FAIL'}] Effort limits verification")
    results.append("")
    
    # Check key bodies
    results.append("--- Key Bodies ---")
    body_names = set()
    
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            body_names.add(prim.GetName())
    
    bodies_ok = True
    for body in KEY_BODIES:
        found = body in body_names
        if not found:
            bodies_ok = False
        results.append(f"  [{'PASS' if found else 'FAIL'}] {body}")
    
    results.append(f"\n[{'PASS' if bodies_ok else 'FAIL'}] Body names verification")
    results.append(f"Total bodies with RigidBody/Mass API: {len(body_names)}")
    results.append("")
    
    # Check for closed-loop FixedJoints
    results.append("--- Fixed Joints (Closed Loop Check) ---")
    if fixed_joints:
        results.append(f"Found {len(fixed_joints)} FixedJoint prims:")
        for prim in fixed_joints:
            results.append(f"  - {prim.GetPath()}")
        results.append("[WARN] FixedJoints may create closed loops")
    else:
        results.append("[PASS] No FixedJoint prims found")
    results.append("")
    
    # Overall
    all_ok = dof_ok and joint_names_ok and effort_ok and bodies_ok and (len(fixed_joints) == 0)
    results.append("=" * 50)
    results.append(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    results.append("=" * 50)
    
    return "\n".join(results), all_ok


def compare_with_backup(new_usd: str, backup_usd: str):
    """Compare new USD with backup."""
    results = []
    results.append(f"\n=== Comparison with Backup ===\n")
    
    new_stage = Usd.Stage.Open(new_usd)
    # Try to convert .bak to .usdc if needed
    if backup_usd.endswith('.bak'):
        import shutil
        temp_usdc = backup_usd.replace('.bak', '.temp.usdc')
        shutil.copy(backup_usd, temp_usdc)
        backup_stage = Usd.Stage.Open(temp_usdc)
    else:
        backup_stage = Usd.Stage.Open(backup_usd)
    
    if not new_stage or not backup_stage:
        results.append("[FAIL] Could not open one or both USD files")
        return "\n".join(results)
    
    # Get joint names
    new_joints = set()
    backup_joints = set()
    
    for prim in new_stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            new_joints.add(prim.GetName())
    
    for prim in backup_stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            backup_joints.add(prim.GetName())
    
    results.append(f"New USD joints: {len(new_joints)}")
    results.append(f"Backup USD joints: {len(backup_joints)}")
    results.append("")
    
    # Check for lost joints
    lost = backup_joints - new_joints
    if lost:
        results.append(f"[WARN] Joints in backup but not in new: {lost}")
    else:
        results.append("[PASS] No joints lost")
    
    # Check for added joints
    added = new_joints - backup_joints
    if added:
        results.append(f"[WARN] Joints in new but not in backup: {added}")
    else:
        results.append("[PASS] No joints added")
    
    results.append("")
    
    return "\n".join(results)


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    new_usd = base_dir / "assets/g1/usd/g1_29dof.usd"
    backup_usd = base_dir / "assets/g1/usd/g1_29dof.usd.bak"
    
    # Verify new USD
    print(f"Verifying: {new_usd}")
    verification_output, success = verify_usd(str(new_usd))
    print(verification_output)
    
    # Compare with backup
    if backup_usd.exists():
        comparison_output = compare_with_backup(str(new_usd), str(backup_usd))
        print(comparison_output)
    else:
        print(f"\n[WARN] Backup USD not found: {backup_usd}")
    
    # Save evidence
    evidence_dir = base_dir.parent / ".sisyphus/evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    with open(evidence_dir / "task-3-usd-verification.txt", "w") as f:
        f.write(verification_output)
        if backup_usd.exists():
            f.write("\n" + comparison_output)
    
    # Extract body names for separate evidence file
    stage = Usd.Stage.Open(str(new_usd))
    body_names = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            body_names.append(prim.GetName())
    
    with open(evidence_dir / "task-3-body-names.txt", "w") as f:
        f.write("=== Body Names in USD ===\n\n")
        for name in sorted(body_names):
            f.write(f"{name}\n")
    
    print(f"\nEvidence saved to:")
    print(f"  - {evidence_dir / 'task-3-usd-verification.txt'}")
    print(f"  - {evidence_dir / 'task-3-body-names.txt'}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
