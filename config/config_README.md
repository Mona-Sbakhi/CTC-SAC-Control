# Configuration Directory

This directory contains configuration files for the CTC-SAC hybrid control system.

## Files

### params.yaml
Main configuration file containing all parameters for:
- **Robot Parameters**: Kinova Mico physical properties (masses, lengths, inertias)
- **CTC Controller**: PD gains, friction compensation settings
- **SAC Agent**: Network architecture, learning rates, hyperparameters
- **Training**: Episode count, batch size, checkpoint settings
- **Trajectory**: Reference trajectory parameters (circle, lemniscate, etc.)
- **Reward Function**: Weight parameters for different reward components
- **CoppeliaSim**: Simulation connection and settings
- **ROS 2**: Node names, topics, QoS settings
- **Logging**: Debug and output settings

## Usage

Parameters can be modified directly in `params.yaml` before running experiments. The configuration is loaded automatically by:
- Training scripts (`training/train_sac.py`)
- ROS 2 nodes (`ros2_ws/src/ctc_sac_control/`)
- Evaluation scripts (`training/evaluate.py`)

## Important Parameters to Adjust

### For Different Robots
Modify the `robot` section with your robot's specifications.

### For Training Tuning
- `sac.learning_rate`: Adjust if training is unstable
- `training.num_episodes`: Increase for more thorough training
- `sac.batch_size`: Increase for more stable learning (if you have enough memory)

### For Trajectory Tracking
- `trajectory.type`: Change trajectory type ('circle', 'lemniscate', 'sine', 'step')
- `trajectory.<type>.*`: Adjust specific trajectory parameters

### For Performance
- `ctc.kp` and `ctc.kd`: Tune PD gains for better CTC performance
- `sac.action_bounds`: Adjust residual torque limits
- `reward.*_weight`: Tune reward function weights
