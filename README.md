🧠 Intelligence Layer
The AI & Intelligence backbone of NeloStrix Robotics

Overview
Intelligence Layer is the centralized AI and cognitive processing framework that powers autonomous decision-making, perception, and adaptive behavior across the entire NeloStrix robotics ecosystem. It serves as the "brain" that sits between raw sensor data and physical actuation — transforming observations into intelligent action.
This repository contains the core modules for perception, planning, learning, and reasoning that enable NeloStrix robots to understand, adapt, and operate autonomously in complex real-world environments.

Architecture
┌─────────────────────────────────────────────────────────────┐
│ INTELLIGENCE LAYER │
│ │
│ ┌─────────────┐ ┌──────────────┐ ┌───────────────────┐ │
│ │ Perception │ │ Planning │ │ Decision Engine │ │
│ │ Pipeline │→ │ & Policy │→ │ & Reasoning │ │
│ └─────────────┘ └──────────────┘ └───────────────────┘ │
│ ↑ │ │
│ ┌─────────────┐ ┌──────────────┐ ↓ │
│ │ Sensor │ │ Learning & │ ┌───────────────────┐ │
│ │ Fusion │ │ Adaptation │ │ Action Interface │ │
│ └─────────────┘ └──────────────┘ └───────────────────┘ │
│ │
└────────────────────────┬────────────────────────────────────┘
│
┌──────────┴──────────┐
│ NeloStrix Hardware │
│ Abstraction Layer │
└─────────────────────┘

Core Modules
Perception Pipeline
Computer vision, LiDAR processing, depth estimation, and multi-modal sensor fusion. Converts raw sensory input into structured world representations that downstream modules can reason over.
Planning & Policy
Task decomposition, motion planning, path optimization, and behavioral policy execution. Generates feasible action sequences from high-level goals using both classical planners and learned policies.
Decision Engine & Reasoning
Real-time decision-making under uncertainty. Incorporates probabilistic reasoning, constraint satisfaction, and safety-aware arbitration to select optimal actions in dynamic environments.
Learning & Adaptation
Online and offline learning pipelines — including reinforcement learning, imitation learning, and continual fine-tuning — that allow robots to improve performance over time and generalize to novel scenarios.
Sensor Fusion
Temporal and spatial alignment of heterogeneous sensor streams (camera, LiDAR, IMU, proprioception) into a unified state estimate with calibrated uncertainty.
Action Interface
Translates high-level decisions into low-level motor commands, bridging the Intelligence Layer to the NeloStrix Hardware Abstraction Layer with real-time safety checks and trajectory smoothing.

Tech Stack
DomainTechnologiesDeep LearningPyTorch, ONNX Runtime, TensorRTComputer VisionOpenCV, Open3D, DepthAnythingPlanningOMPL, MoveIt, Custom MCTSRL & TrainingStable-Baselines3, Isaac GymInferenceTensorRT, Triton Inference ServerCommunicationROS 2 (Humble/Iron), gRPC, ZeroMQDeploymentDocker, Kubernetes, NVIDIA Jetson

Repository Structure
intelligence-layer/
├── perception/ # Vision, LiDAR, and sensor processing
├── planning/ # Task & motion planning
├── decision/ # Reasoning and decision engine
├── learning/ # RL, IL, and training pipelines
├── fusion/ # Multi-sensor fusion
├── action/ # Action interface & safety layer
├── models/ # Pre-trained model weights & configs
├── protos/ # gRPC & message definitions
├── configs/ # Environment and module configs
├── tests/ # Unit, integration, and simulation tests
├── docs/ # Architecture docs and API reference
├── scripts/ # Utilities, benchmarks, and tooling
└── docker/ # Container definitions

Getting Started
Prerequisites

Python 3.10+
CUDA 12.x with cuDNN 8+
ROS 2 Humble or Iron
Docker (recommended for reproducible environments)

Installation
bash# Clone the repository
git clone git@github.com:NeloStrix/intelligence-layer.git
cd intelligence-layer

# Set up the environment

make setup # Creates venv, installs deps, pulls model weights

# Or use Docker

docker compose up -d intelligence-layer
Running Tests
bashmake test # Unit tests
make test-sim # Simulation integration tests

Integration with NeloStrix Ecosystem
The Intelligence Layer is designed to integrate seamlessly with other NeloStrix subsystems:

HAL (Hardware Abstraction Layer) — receives actuator capabilities, sends motor commands
Fleet Manager — reports agent status, receives mission assignments
Data Pipeline — streams telemetry and training data to cloud storage
Simulation Engine — supports hardware-in-the-loop and pure sim evaluation

Contributing
This is an internal NeloStrix repository. Please refer to our Contributing Guidelines and ensure all changes pass CI before opening a merge request. All modules require accompanying tests and documentation updates.

License
Proprietary — © NeloStrix Robotics. All rights reserved. See LICENSE for details.

<p align="center">
  <b>NeloStrix Robotics</b> — Building intelligent machines that see, think, and act.
</p>
