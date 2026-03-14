from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.bvh_provider import BVHInputProvider
from teleopit.retargeting.core import RetargetingModule
from teleopit.retargeting.gmr.params import ROBOT_XML_DICT


def mocap_xml_path(project_root: Path, robot_name: str = "unitree_g1") -> Path:
    del project_root
    try:
        return Path(ROBOT_XML_DICT[robot_name]).resolve()
    except KeyError as exc:
        raise ValueError(
            f"unsupported robot_name for BVH retarget export: {robot_name}. "
            f"Supported robots: {sorted(ROBOT_XML_DICT)}"
        ) from exc


def convert_bvh_to_retarget_pkl(
    bvh_path: Path,
    output_pkl: Path,
    bvh_format: str,
    robot_name: str,
    max_frames: int,
    model: mujoco.MjModel,
) -> None:
    provider = BVHInputProvider(bvh_path=str(bvh_path), human_format=bvh_format)
    retargeter = RetargetingModule(
        robot_name=robot_name,
        human_format=f"bvh_{provider.human_format}",
        actual_human_height=provider.human_height,
    )

    n_total = len(provider)
    n_frames = n_total if max_frames <= 0 else min(n_total, max_frames)
    if n_frames <= 0:
        raise ValueError(f"No frames in BVH: {bvh_path}")

    expected_qpos_dim = model.nq
    num_actions = expected_qpos_dim - 7
    data = mujoco.MjData(model)
    link_body_list = [model.body(i).name for i in range(1, model.nbody)]
    n_bodies = len(link_body_list)

    root_pos = np.zeros((n_frames, 3), dtype=np.float32)
    root_rot_xyzw = np.zeros((n_frames, 4), dtype=np.float32)
    dof_pos = np.zeros((n_frames, num_actions), dtype=np.float32)
    body_pos_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)

    for i in range(n_frames):
        human_frame = provider.get_frame()
        qpos = np.asarray(retargeter.retarget(human_frame), dtype=np.float64).reshape(-1)
        if qpos.shape[0] != expected_qpos_dim:
            raise ValueError(
                f"Retargeted qpos dim mismatch at frame {i}: got {qpos.shape[0]}, expected {expected_qpos_dim}"
            )

        root_pos[i] = qpos[0:3].astype(np.float32)
        root_quat_wxyz = qpos[3:7]
        root_rot_xyzw[i] = np.array(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
            dtype=np.float32,
        )
        dof_pos[i] = qpos[7: 7 + num_actions].astype(np.float32)

        data.qpos[:] = 0.0
        data.qpos[: 7 + num_actions] = qpos[: 7 + num_actions]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos_w[i] = np.asarray(data.xpos[1: 1 + n_bodies], dtype=np.float32)

    root_rot_local = R.from_quat(root_rot_xyzw).inv().as_matrix()
    delta = body_pos_w - root_pos[:, None, :]
    local_body_pos = np.einsum("tij,tbj->tbi", root_rot_local, delta)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": int(provider.fps),
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.astype(np.float32),
        "link_body_list": link_body_list,
    }
    with output_pkl.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
