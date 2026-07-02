# GOHAN

GOHAN means Guided Optimization for High-speed Autonomous Navigation.

This repository is now the GOHAN project root. It contains a telemetry-only reinforcement learning autonomy stack for Autonoma Labs AWSIM Racing Simulator and the Dallara AV-21R. AWSIM runs separately and publishes ROS2 telemetry; GOHAN converts that telemetry into a normalized vector observation, runs a Stable-Baselines3 policy, and publishes steering/throttle/brake commands back through ROS2.

GOHAN does not include AWSIM, clone AWSIM, modify simulator source, use camera perception, or assume CUDA.

## Architecture

```text
AWSIM Racing Simulator
  -> ROS2 telemetry
GOHAN Ros2Bridge
  -> telemetry feature extractor
Stable-Baselines3 PPO/SAC-ready policy
  -> action mapper and safety filter
GOHAN Ros2Bridge
  -> ROS2 vehicle command
AWSIM Dallara AV-21R
  -> logging, reward, and training loop
```

## Layout

```text
AG26/
  README.md
  pyproject.toml
  requirements.txt
  configs/
    awsim.yaml
    reward.yaml
    training.yaml
    tracks/default_track.yaml
  scripts/
  src/gohan/
  tests/
  runs/
  ros2_ws/
    src/sim-msgs/
    build/
    install/
    log/
```

`ros2_ws` is only for external ROS2 message packages such as `autonoma_msgs`. The Python package lives directly in this root under `src/gohan`.

## Prerequisites

- Linux Mint 22.x or another ROS2 Jazzy-compatible Linux system.
- ROS2 installed externally, for example `/opt/ros/jazzy`.
- Python 3.10+.
- AWSIM Racing Simulator installed separately and running in a ROS2-controlled racing scene.
- Autonoma message packages built in `ros2_ws`.
- CPU-first training. CUDA is not required.

`rclpy` is intentionally not in `requirements.txt`; it comes from the sourced ROS2 installation.

## Build ROS2 Message Workspace

Run this once from the project root. Use system Python, not Anaconda, so ROS2 CMake finds the right Python packages.

```bash
cd /home/osama/Documents/RL/AG26
mkdir -p ros2_ws/src
cd ros2_ws/src

git clone https://github.com/autonomalabs/sim-msgs.git

cd /home/osama/Documents/RL/AG26/ros2_ws
conda deactivate
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

If a previous configure was polluted by Anaconda, rebuild with:

```bash
cd /home/osama/Documents/RL/AG26/ros2_ws
conda deactivate
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-clean-cache --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

## Set Up GOHAN Python

Create the Python environment in the root:

```bash
cd /home/osama/Documents/RL/AG26
conda deactivate

/usr/bin/python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Every GOHAN Terminal

Run this before any ROS-facing script:

```bash
cd /home/osama/Documents/RL/AG26
conda deactivate

source /opt/ros/jazzy/setup.bash
source /home/osama/Documents/RL/AG26/ros2_ws/install/setup.bash
source .venv/bin/activate
export ROS_DOMAIN_ID=0
```

## Start AWSIM

AWSIM does not need to live in this repository. Start it however you installed it, then enter the racing Drive scene with ROS2 control enabled.

```bash
export ROS_DOMAIN_ID=0
# launch AWSIM using your installed executable, launcher, or desktop shortcut
```

Topics will not appear while AWSIM is closed or still outside the running Drive scenario. If GOHAN sees only `/parameter_events` and `/rosout`, ROS2 is alive but AWSIM is not visible on that ROS graph.

Manual reset is the default. Automatic reset needs an AWSIM-side reset service or simulator integration; add that later in `AWSIMRacingEnv.reset()` when a reset API is available.

## Inspect ROS2 Topics

```bash
ros2 topic list -t
python scripts/inspect_ros_topics.py --config configs/awsim.yaml
```

Expected high-level racing topics include:

```text
/novatel_top/inspva or /novatel_bottom/inspva
/novatel_top/rawimux or /novatel_bottom/rawimux
/vehicle_data
/powertrain_data
/race_control
/vehicle_inputs
/to_raptor
```

If a command topic type is unsupported, inspect it:

```bash
ros2 topic info /vehicle_inputs
```

Then update:

```text
src/gohan/ros/message_adapters.py
```

GOHAN fails clearly instead of publishing an unknown command shape.

## Record Telemetry

```bash
python scripts/record_telemetry.py --config configs/awsim.yaml --run-name telemetry_test
```

This verifies that AWSIM telemetry is visible and that the adapters can build GOHAN telemetry rows.

## Test Command Publishing

```bash
python scripts/drive_manual_test.py --config configs/awsim.yaml --mode slow_forward
```

Supported modes:

- `neutral`
- `brake`
- `slow_forward`
- `sine_steer_low_speed`

Run this before training.

## Check Environment

```bash
python scripts/check_awsim_env.py --config configs/awsim.yaml
```

This waits for telemetry, creates one observation vector, publishes one neutral command, and exits.

## Train PPO

Train only after topic inspection, telemetry recording, environment check, and manual command publishing work.

```bash
python scripts/train_ppo.py \
  --config configs/awsim.yaml \
  --training-config configs/training.yaml \
  --reward-config configs/reward.yaml \
  --run-name awsim_ppo_v1 \
  --total-timesteps 100000
```

Models and logs are written under:

```text
runs/awsim_ppo_v1/
```

## Evaluate

```bash
python scripts/evaluate_policy.py \
  --config configs/awsim.yaml \
  --model runs/awsim_ppo_v1/models/final_model.zip \
  --episodes 3
```

## Live Controller

```bash
python scripts/run_inference_controller.py \
  --config configs/awsim.yaml \
  --model runs/awsim_ppo_v1/models/final_model.zip
```

Ctrl+C sends a brake command before shutdown.

## Plot Telemetry

```bash
python scripts/plot_telemetry.py --input runs/awsim_ppo_v1/telemetry.csv --output runs/awsim_ppo_v1/plots
```

## Observation Vector

GOHAN uses a 20D `np.float32` vector clipped to `[-1, 1]`:

```text
[
  lateral_error_norm,
  heading_error_sin,
  heading_error_cos,
  speed_norm,
  longitudinal_velocity_norm,
  lateral_velocity_norm,
  yaw_rate_norm,
  track_curvature_norm,
  distance_to_left_boundary_norm,
  distance_to_right_boundary_norm,
  lookahead_1_dx_norm,
  lookahead_1_dy_norm,
  lookahead_2_dx_norm,
  lookahead_2_dy_norm,
  lookahead_3_dx_norm,
  lookahead_3_dy_norm,
  previous_steering,
  previous_throttle_brake,
  lap_progress_sin,
  lap_progress_cos
]
```

The observation is built from ROS2 telemetry and track-relative features, not images.

## Action Mapping

The action space is:

```python
gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

`action[0]` is steering. `action[1]` is combined throttle/brake:

```text
if action[1] >= 0:
  throttle = action[1]
  brake = 0
else:
  throttle = 0
  brake = abs(action[1])
```

## Reward

`RacingReward` combines progress, speed, line tracking, heading alignment, yaw stability, action smoothness, off-track, collision, reverse-progress, and stuck penalties.

## Tests

```bash
cd /home/osama/Documents/RL/AG26
source .venv/bin/activate
pytest
```

Unit tests do not require AWSIM or ROS2.

## Limitations

- Automatic reset requires AWSIM-side reset support or manual reset.
- Command message types may need adapter updates if AWSIM changes the ROS2 interface.
- The first training goal is stable lap completion, not optimal lap time.
- The default track file is a placeholder; replace it with AWSIM track waypoints or a racing line before serious training.

## Roadmap

- v1 direct telemetry-to-control PPO.
- v1.5 reward and curriculum refinement.
- v2 telemetry-guided racing target correction.
- v3 hybrid controller plus RL target optimizer.
