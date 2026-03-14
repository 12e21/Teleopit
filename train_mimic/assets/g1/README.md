# G1 Assets

This directory no longer mirrors the full set of historical G1 MJCF/URDF
variants.

Current status:

- The training environment itself uses the G1 asset shipped in `mjlab`.
- The only local MJCF kept here is `g1_sim2sim_29dof.xml`, which is still used
  by `train_mimic.data.motion_fk.MotionFkExtractor` during NPZ FK extraction and
  consistency checks.
- Meshes are kept because the remaining MJCF still references them.
- Images are left untouched as auxiliary documentation assets.
