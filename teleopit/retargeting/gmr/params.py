from pathlib import Path

from teleopit.runtime.assets import (
    GMR_ASSETS_ROOT,
    UNITREE_G1_AVP_O6_XML,
    UNITREE_G1_DEX3_XML,
    UNITREE_G1_XML,
)

BASE_DIR = Path(__file__).parent


def _resolve_path(relative_path):
    return BASE_DIR / relative_path


def _resolve_gmr_asset_path(relative_path):
    return GMR_ASSETS_ROOT / relative_path


IK_CONFIG_ROOT = _resolve_path("ik_configs")
ASSET_ROOT = GMR_ASSETS_ROOT

ROBOT_XML_DICT = {
    "unitree_g1": UNITREE_G1_XML,
    "unitree_g1_with_hands": UNITREE_G1_DEX3_XML,
    "unitree_g1_avp_o6": UNITREE_G1_AVP_O6_XML,
    "unitree_h1": _resolve_gmr_asset_path("unitree_h1/h1.xml"),
    "unitree_h1_2": _resolve_gmr_asset_path("unitree_h1_2/h1_2_handless.xml"),
    "booster_t1": _resolve_gmr_asset_path("booster_t1/T1_serial.xml"),
    "booster_t1_29dof": _resolve_gmr_asset_path("booster_t1_29dof/t1_mocap.xml"),
    "stanford_toddy": _resolve_gmr_asset_path("stanford_toddy/toddy_mocap.xml"),
    "fourier_n1": _resolve_gmr_asset_path("fourier_n1/n1_mocap.xml"),
    "engineai_pm01": _resolve_gmr_asset_path("engineai_pm01/pm_v2.xml"),
    "kuavo_s45": _resolve_gmr_asset_path("kuavo_s45/biped_s45_collision.xml"),
    "hightorque_hi": _resolve_gmr_asset_path("hightorque_hi/hi_25dof.xml"),
    "galaxea_r1pro": _resolve_gmr_asset_path("galaxea_r1pro/r1_pro.xml"),
    "berkeley_humanoid_lite": _resolve_gmr_asset_path("berkeley_humanoid_lite/bhl_scene.xml"),
    "booster_k1": _resolve_gmr_asset_path("booster_k1/K1_serial.xml"),
    "pnd_adam_lite": _resolve_gmr_asset_path("pnd_adam_lite/scene.xml"),
    "tienkung": _resolve_gmr_asset_path("tienkung/mjcf/tienkung.xml"),
    "pal_talos": _resolve_gmr_asset_path("pal_talos/talos.xml"),
    "fourier_gr3": _resolve_gmr_asset_path("fourier_gr3v2_1_1/mjcf/gr3v2_1_1_dummy_hand.xml"),
}

IK_CONFIG_DICT = {
    # offline data
    "smplx": {
        "unitree_g1": _resolve_path("ik_configs/smplx_to_g1.json"),
        "unitree_g1_with_hands": _resolve_path("ik_configs/smplx_to_g1.json"),
        "unitree_g1_avp_o6": _resolve_path("ik_configs/smplx_to_g1.json"),
        "unitree_h1": _resolve_path("ik_configs/smplx_to_h1.json"),
        "unitree_h1_2": _resolve_path("ik_configs/smplx_to_h1_2.json"),
        "booster_t1": _resolve_path("ik_configs/smplx_to_t1.json"),
        "booster_t1_29dof": _resolve_path("ik_configs/smplx_to_t1_29dof.json"),
        "stanford_toddy": _resolve_path("ik_configs/smplx_to_toddy.json"),
        "fourier_n1": _resolve_path("ik_configs/smplx_to_n1.json"),
        "engineai_pm01": _resolve_path("ik_configs/smplx_to_pm01.json"),
        "kuavo_s45": _resolve_path("ik_configs/smplx_to_kuavo.json"),
        "hightorque_hi": _resolve_path("ik_configs/smplx_to_hi.json"),
        "galaxea_r1pro": _resolve_path("ik_configs/smplx_to_r1pro.json"),
        "berkeley_humanoid_lite": _resolve_path("ik_configs/smplx_to_bhl.json"),
        "booster_k1": _resolve_path("ik_configs/smplx_to_k1.json"),
        "pnd_adam_lite": _resolve_path("ik_configs/smplx_to_adam.json"),
        "tienkung": _resolve_path("ik_configs/smplx_to_tienkung.json"),
        "fourier_gr3": _resolve_path("ik_configs/smplx_to_gr3.json"),
    },
    "bvh_lafan1": {
        "unitree_g1": _resolve_path("ik_configs/bvh_lafan1_to_g1.json"),
        "unitree_g1_with_hands": _resolve_path("ik_configs/bvh_lafan1_to_g1.json"),
        "unitree_g1_avp_o6": _resolve_path("ik_configs/bvh_lafan1_to_g1.json"),
        "booster_t1_29dof": _resolve_path("ik_configs/bvh_lafan1_to_t1_29dof.json"),
        "fourier_n1": _resolve_path("ik_configs/bvh_lafan1_to_n1.json"),
        "stanford_toddy": _resolve_path("ik_configs/bvh_lafan1_to_toddy.json"),
        "engineai_pm01": _resolve_path("ik_configs/bvh_lafan1_to_pm01.json"),
        "pal_talos": _resolve_path("ik_configs/bvh_to_talos.json"),
    },
    "bvh_nokov": {
        "unitree_g1": _resolve_path("ik_configs/bvh_nokov_to_g1.json"),
    },
    "bvh_hc_mocap": {
        "unitree_g1": _resolve_path("ik_configs/bvh_hc_mocap_to_g1.json"),
    },
    "bvh_xsens": {
        "unitree_g1": _resolve_path("ik_configs/bvh_xsens_to_g1.json"),
        "unitree_h1_2": _resolve_path("ik_configs/bvh_xsens_to_h1_2.json"),
    },
    "fbx": {
        "unitree_g1": _resolve_path("ik_configs/fbx_to_g1.json"),
        "unitree_g1_with_hands": _resolve_path("ik_configs/fbx_to_g1.json"),
        "unitree_g1_avp_o6": _resolve_path("ik_configs/fbx_to_g1.json"),
    },
    "fbx_offline": {
        "unitree_g1": _resolve_path("ik_configs/fbx_offline_to_g1.json"),
    },
    "pico_bridge": {
        "unitree_g1": _resolve_path("ik_configs/pico_bridge_to_g1.json"),
    },
}


ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "unitree_g1_with_hands": "pelvis",
    "unitree_g1_avp_o6": "pelvis",
    "unitree_h1": "pelvis",
    "unitree_h1_2": "pelvis",
    "booster_t1": "Waist",
    "booster_t1_29dof": "Waist",
    "stanford_toddy": "waist_link",
    "fourier_n1": "base_link",
    "engineai_pm01": "LINK_BASE",
    "kuavo_s45": "base_link",
    "hightorque_hi": "base_link",
    "galaxea_r1pro": "torso_link4",
    "berkeley_humanoid_lite": "imu_2",
    "booster_k1": "Trunk",
    "pnd_adam_lite": "pelvis",
    "tienkung": "Base_link",
    "pal_talos": "base_link",
    "fourier_gr3": "base_link",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "unitree_g1_with_hands": 2.0,
    "unitree_g1_avp_o6": 2.0,
    "unitree_h1": 3.0,
    "unitree_h1_2": 3.0,
    "booster_t1": 2.0,
    "booster_t1_29dof": 2.0,
    "stanford_toddy": 1.0,
    "fourier_n1": 2.0,
    "engineai_pm01": 2.0,
    "kuavo_s45": 3.0,
    "hightorque_hi": 2.0,
    "galaxea_r1pro": 3.0,
    "berkeley_humanoid_lite": 2.0,
    "booster_k1": 2.0,
    "pnd_adam_lite": 3.0,
    "tienkung": 3.0,
    "pal_talos": 3.0,
    "fourier_gr3": 2.0,
}
