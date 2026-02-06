# CTC-SAC-Control: Hybrid Robotic Manipulator Control

**Computed Torque Control (CTC) + Soft Actor-Critic (SAC) for Enhanced Trajectory Tracking**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-green.svg)](https://docs.ros.org/en/humble/)

## Overview

This repository implements a hybrid control approach combining:
- **CTC (Computed Torque Control)**: Physics-informed base controller providing ~90% of control effort
- **SAC (Soft Actor-Critic)**: Residual RL agent learning to correct model uncertainties

The approach significantly improves trajectory tracking accuracy for robotic manipulators by leveraging analytical dynamics for the majority of control while using RL to learn residual corrections.

---

## Project Structure

```
CTC-SAC-Control/
├── ros2_ws/
│   └── src/
│       ├── robot_dynamics.py         # Lagrangian dynamics for Kinova Mico (2-DOF)
│       ├── ctc_controller.py         # CTC implementation with PD feedback
│       ├── sac_agent.py              # SAC agent (actor-critic architecture)
│       ├── train_in_coppeliasim.py   # Training script for CoppeliaSim
│       ├── ctc_sac_experiment.py     # Evaluation and testing script
│       └── coppeliasim_experiment.py # Alternative experiment script
├── results/
│   ├── ctc_sac_agent.pt              # Trained SAC model
│   ├── ctc_sac_results.json          # Performance metrics
│   ├── ctc_sac_data.csv              # Time-series data
│   └── ctc_sac_results.png           # Visualization plots
├── config/
│   └── params.yaml                   # System parameters configuration
├── docker-compose.yml                # Docker setup for ROS2 + CoppeliaSim
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## Key Features

### ✅ Core Components

1. **Robot Dynamics Module** (`robot_dynamics.py`)
   - Full Lagrangian formulation for 2-DOF planar manipulator
   - Kinova Mico parameters (m₁=2.072kg, m₂=1.072kg, L₁=L₂=0.15m)
   - Includes friction modeling and parameter uncertainty handling

2. **CTC Controller** (`ctc_controller.py`)
   - Inverse dynamics with PD feedback linearization
   - Gravity, Coriolis, and friction compensation
   - Provides ~90% of required control torque

3. **SAC Agent** (`sac_agent.py`)
   - Twin Q-networks for stable learning
   - Gaussian actor with entropy regularization
   - Residual action space: ±5 N·m (vs ±20 N·m for full control)

4. **CoppeliaSim Integration**
   - ZMQ-based communication (host.docker.internal:23000)
   - Real-time trajectory tracking experiments
   - 50Hz control loop with dynamic simulation

---

## Installation

### Prerequisites

- Docker and Docker Compose
- CoppeliaSim installed on host machine
- Python 3.10+

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Mona-Sbakhi/CTC-SAC-Control.git
cd CTC-SAC-Control
```

2. **Start Docker environment**
```bash
docker-compose up -d
```

3. **Enter Docker container**
```bash
docker exec -it ctc-sac-control bash
```

4. **Install dependencies** (inside container)
```bash
cd /workspace/ros2_ws/src
pip install -r requirements.txt --break-system-packages
```

5. **Start CoppeliaSim** (on host machine)
   - Open CoppeliaSim
   - Load scene: `Mico_2DOF_vertical.ttt`
   - Enable ZMQ Remote API Server (port 23000)

---

## Usage

### 1. Training CTC+SAC Agent

```bash
cd /workspace/ros2_ws/src
python train_in_coppeliasim.py
```

**Training Configuration:**
- Episodes: 200
- Steps per episode: 200
- Control frequency: 50Hz (dt=0.02s)
- Trajectory: Sinusoidal (freq=0.5Hz, amp=0.5rad)
- Reward: Position error + velocity error + control effort penalty

**Outputs:**
- `results/best_ctc_sac_coppeliasim.pt` - Best model checkpoint
- `results/training_curves.png` - Reward and error plots

### 2. Running Experiments

```bash
cd /workspace/ros2_ws/src
python ctc_sac_experiment.py
```

**Experiment outputs:**
```
results/
├── ctc_sac_metrics.json     # RMS errors, max errors, mean tracking error
├── ctc_sac_data.csv         # Complete time-series data
└── ctc_sac_results.png      # 4-subplot visualization
    ├── Trajectory tracking
    ├── Torque decomposition (CTC vs RL)
    ├── Tracking error over time
    └── CTC contribution ratio
```

### 3. Using Pre-trained Model

```python
from sac_agent import SACAgent, SACConfig
from ctc_controller import CTCController, CTCGains
from robot_dynamics import create_nominal_robot

# Load robot and CTC
robot = create_nominal_robot()
gains = CTCGains(Kp=[100.0, 50.0], Kd=[20.0, 10.0])
ctc = CTCController(robot, gains)

# Load SAC agent
config = SACConfig(action_low=-5.0, action_high=5.0)
sac = SACAgent(state_dim=8, action_dim=2, config=config)
sac.load('results/ctc_sac_agent.pt')

# Control loop
state = get_robot_state()  # [q, q_dot, e, e_dot]
tau_ctc = ctc.compute_torque(q, q_dot, q_d, q_dot_d, q_ddot_d)
tau_rl = sac.select_action(state, deterministic=True)
tau_total = np.clip(tau_ctc + tau_rl, -20.0, 20.0)
```

---

## Configuration

Edit `config/params.yaml` to customize:

### Robot Parameters
```yaml
robot:
  m1: 2.072        # Link 1 mass [kg]
  m2: 1.072        # Link 2 mass [kg]
  L1: 0.15         # Link 1 length [m]
  L2: 0.15         # Link 2 length [m]
  I1: 0.001        # Link 1 inertia [kg·m²]
  I2: 0.00098      # Link 2 inertia [kg·m²]
```

### CTC Gains
```yaml
ctc:
  Kp: [100.0, 50.0]   # Position gains
  Kd: [20.0, 10.0]    # Velocity gains
```

### SAC Hyperparameters
```yaml
sac:
  hidden_dims: [256, 256]
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 256
  action_bounds: [-5.0, 5.0]  # Residual torque limits
```

---

## Docker Configuration

### Docker Compose (`docker-compose.yml`)

```yaml
services:
  ros2:
    image: osrf/ros:humble-desktop
    container_name: ctc-sac-control
    network_mode: bridge
    volumes:
      - ./ros2_ws:/workspace/ros2_ws
      - ./results:/workspace/results
      - ./config:/workspace/config
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1
```

### Key Points:
- **Image**: `osrf/ros:humble-desktop` (not Jazzy as initially planned)
- **Network**: `host.docker.internal:23000` for CoppeliaSim communication
- **Volumes**: Mounted for code, results, and configuration
- **No ROS2 workspace build**: Python scripts run directly

---

## File Descriptions

### Core Implementation Files

| File | Description | Key Functions |
|------|-------------|---------------|
| `robot_dynamics.py` | Lagrangian dynamics model | `mass_matrix()`, `coriolis_matrix()`, `gravity_vector()`, `inverse_dynamics()` |
| `ctc_controller.py` | CTC with PD feedback | `compute_torque()`, `compute_torque_decomposed()` |
| `sac_agent.py` | SAC RL agent | `GaussianActor`, `TwinQNetwork`, `select_action()` |
| `train_in_coppeliasim.py` | Training pipeline | Episode loop, reward calculation, model checkpointing |
| `ctc_sac_experiment.py` | Evaluation script | Data collection, metrics computation, visualization |

### Configuration Files

| File | Purpose |
|------|---------|
| `params.yaml` | Central configuration for robot, CTC, SAC, and trajectory |
| `requirements.txt` | Python dependencies (torch, numpy, scipy, matplotlib, pyzmq, etc.) |
| `docker-compose.yml` | Docker environment setup |

---

## Results Format

### JSON Metrics (`ctc_sac_results.json`)
```json
{
  "rms_error_q1": 0.0245,
  "rms_error_q2": 0.0189,
  "max_error_q1": 0.0521,
  "max_error_q2": 0.0398,
  "mean_error": 0.0217,
  "ctc_contribution_q1": 91.3,
  "ctc_contribution_q2": 89.7
}
```

### CSV Data (`ctc_sac_data.csv`)
Columns: `time, q1_desired, q2_desired, q1_actual, q2_actual, q1_dot, q2_dot, error_q1, error_q2, error_dot_q1, error_dot_q2, tau_ctc_1, tau_ctc_2, tau_rl_1, tau_rl_2, tau_total_1, tau_total_2`

---

## Troubleshooting

### Common Issues

1. **CoppeliaSim Connection Failed**
   ```bash
   # Check if CoppeliaSim is running with ZMQ server on port 23000
   # Verify host.docker.internal resolves correctly
   docker exec -it ctc-sac-control ping host.docker.internal
   ```

2. **Module Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd /workspace/ros2_ws/src
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

3. **CUDA/GPU Issues**
   - This implementation uses CPU by default (`device='cpu'`)
   - For GPU training, modify `SACAgent` initialization

4. **Docker Volume Permissions**
   ```bash
   # Fix permissions if needed
   sudo chown -R $USER:$USER ./ros2_ws ./results ./config
   ```

---

## Performance Benchmarks

### Kinova Mico (2-DOF Vertical Configuration)

| Method | RMS Error q₁ | RMS Error q₂ | Mean Error | Training Episodes |
|--------|--------------|--------------|------------|-------------------|
| CTC-Only | 0.0312 rad | 0.0267 rad | 0.0289 rad | N/A |
| **CTC+SAC** | **0.0245 rad** | **0.0189 rad** | **0.0217 rad** | 200 |
| **Improvement** | **21.5%** | **29.2%** | **24.9%** | - |

**Key Insights:**
- CTC provides ~90% of control torque (analytical dynamics)
- SAC learns ±5 N·m residual corrections (~10% of total)
- Sample efficiency: 40K timesteps (200 episodes × 200 steps)
- Converges in ~2 hours on CPU

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{alsbakhi2026ctc_sac,
  title={Hybrid CTC-SAC Control for Robotic Manipulator Trajectory Tracking},
  author={Alsbakhi, Mona and Alsalool, Anas and Tabash, Majed and Sbaih, Asma},
  booktitle={2026 International Conference on Control, Decision and Information Technologies (CoDIT)},
  year={2026},
  organization={IEEE}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

Copyright © 2026 Mona Alsbakhi, Anas Alsalool, Majed Tabash, Asma Sbaih

---

## Authors

- **Mona Alsbakhi** - Islamic University of Gaza
- **Anas Alsalool** - Islamic University of Gaza
- **Majed Tabash** - Islamic University of Gaza
- **Asma Sbaih** - Islamic University of Gaza

---

## Acknowledgments

- Kinova Robotics for Mico robot parameters
- CoppeliaSim for simulation environment
- OpenAI Gym/Stable-Baselines3 for RL inspiration
- ROS2 Humble for robotics middleware

---

## Archived/Legacy Files

The following files are from previous experiments on different robots and are **not part of the current implementation**:

```
archive/ (not included in main repository)
├── test_phantomx.py          # PhantomX Pincher experiments (old)
├── train_phantomx.py         # PhantomX training (old)
├── test_mtb.py               # MTB robot experiments (old)
├── train_mtb.py              # MTB training (old)
├── mtb_dynamics.py           # MTB dynamics (old)
├── train_rl_only.py          # RL-only baseline (old)
└── test_complete_comparison.py  # Multi-robot comparison (old)
```

These files are kept for reference but are **not required** for the current Mico robot implementation.

---

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: [Your institutional email]

---

**Last Updated**: February 6, 2026
