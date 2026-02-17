# Teleopit

Teleoperation framework for humanoid robots with motion retargeting.

## Installation

```bash
pip install -e .
```

## Project Structure

```
Teleopit/
├── teleopit/
│   ├── retargeting/      # Motion retargeting modules
│   ├── controllers/      # Robot controllers (RL policy, etc.)
│   ├── robots/           # Robot-specific implementations
│   ├── inputs/           # Input sources (BVH, SMPL-X, etc.)
│   ├── bus/              # Communication bus
│   ├── recording/        # Data recording utilities
│   └── configs/          # Hydra configuration files
│       ├── default.yaml
│       ├── robot/
│       ├── controller/
│       └── input/
└── tests/
```

## Configuration

Uses Hydra for configuration management. Default configs in `teleopit/configs/`.
