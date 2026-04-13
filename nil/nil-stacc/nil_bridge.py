"""NIL-STACC Bridge — connect any AI model to any robot.

This is the integration layer between NIL (Nelo Intelligence Layer)
and stacc (Universal Robotics Platform). It provides:

1. Model Registry — discover and load NIL models dynamically
2. Capability Interface — standardized API for all AI capabilities
3. Robot Adapter — connect NIL policies to stacc robots
4. Training Pipeline — train NIL models inside stacc simulation
5. Deployment Pipeline — deploy NIL models to stacc hardware targets

Architecture:
    NIL (AI models, training, inference)
        ↕  nil.bridge (this module)
    stacc (simulation, sensors, control, hardware)

Usage:
    from nil.bridge import NILBridge, ModelRegistry

    # Initialize bridge
    bridge = NILBridge(robot)

    # Load any NIL capability
    bridge.load("vision")        # object detection, segmentation
    bridge.load("embedding")     # visual/text embeddings
    bridge.load("policy")        # learned control policy
    bridge.load("planner")       # motion planning with learned model
    bridge.load("speech")        # voice commands
    bridge.load("generation")    # image/video generation

    # Use in robot control loop
    @robot.on_update
    def control(dt):
        # NIL processes sensor data, stacc executes actions
        image = bridge.camera.read()
        detections = bridge.vision.detect(image)
        action = bridge.policy.predict(robot.state, detections)
        robot.set_joint_torques(action)
"""

from __future__ import annotations

import importlib
import json
import os
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

# ══════════════════════════════════════════
# CAPABILITY TYPES
# ══════════════════════════════════════════


class CapabilityType(Enum):
    """All AI capabilities that NIL can provide to stacc."""

    # Perception
    VISION = "vision"  # object detection, segmentation
    EMBEDDING = "embedding"  # visual/text/multimodal embeddings
    DEPTH_ESTIMATION = "depth_estimation"  # monocular/stereo depth
    SCENE_UNDERSTANDING = "scene"  # scene graphs, spatial reasoning
    POINT_CLOUD = "point_cloud"  # 3D point cloud processing

    # Language & Speech
    NLP = "nlp"  # command parsing, intent
    SPEECH_TO_TEXT = "stt"  # speech recognition
    TEXT_TO_SPEECH = "tts"  # speech synthesis
    LLM = "llm"  # large language model inference

    # Generation
    IMAGE_GENERATION = "image_gen"  # text-to-image, image editing
    VIDEO_GENERATION = "video_gen"  # text-to-video, video prediction
    MOTION_GENERATION = "motion_gen"  # trajectory/motion synthesis

    # Control & Planning
    POLICY = "policy"  # learned control policy
    PLANNER = "planner"  # learned motion planner
    WORLD_MODEL = "world_model"  # predictive dynamics model
    REWARD_MODEL = "reward_model"  # learned reward function

    # Meta-Learning
    META_LEARNER = "meta_learner"  # few-shot adaptation
    SKILL_LIBRARY = "skill_library"  # reusable skill composition
    CURIOSITY = "curiosity"  # intrinsic motivation

    # Robotics-Specific
    SLAM = "slam"  # simultaneous localization & mapping
    GRASP_PLANNER = "grasp_planner"  # grasp pose estimation
    MANIPULATION = "manipulation"  # manipulation policy
    NAVIGATION = "navigation"  # autonomous navigation

    # Transfer & Deployment
    SIM2REAL = "sim2real"  # simulation to real transfer
    DOMAIN_ADAPTATION = "domain_adapt"  # adapt to new domains
    QUANTIZATION = "quantization"  # model compression for edge


# ══════════════════════════════════════════
# STANDARD MODEL INTERFACE
# ══════════════════════════════════════════


@runtime_checkable
class NILModel(Protocol):
    """Standard interface that every NIL model must implement.

    Any model — whether it's a vision transformer, an LLM, a policy
    network, or an embedding model — exposes this interface. stacc
    only calls these methods. It never touches model internals.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def capability(self) -> str: ...

    def predict(self, inputs: Any, **kwargs) -> Any:
        """Run inference. Input/output types depend on capability."""
        ...

    def load(self, path: str) -> None:
        """Load model weights from path."""
        ...

    def save(self, path: str) -> None:
        """Save model weights to path."""
        ...

    @property
    def device(self) -> str:
        """Current device: 'cpu', 'cuda', 'mps', 'tpu'."""
        ...

    def to(self, device: str) -> "NILModel":
        """Move model to device."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model metadata: params, FLOPs, latency, etc."""
        ...


class BaseNILModel:
    """Base implementation of NILModel interface.

    All NIL models inherit from this. Provides default implementations
    for common methods. Subclasses override predict().
    """

    def __init__(self, name: str, version: str = "0.1.0", capability: str = "generic"):
        self._name = name
        self._version = version
        self._capability = capability
        self._device = "cpu"
        self._loaded = False
        self._config: Dict[str, Any] = {}
        self._weights_path: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def capability(self) -> str:
        return self._capability

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, inputs: Any, **kwargs) -> Any:
        raise NotImplementedError(f"{self._name}.predict() not implemented")

    def train(self, data: Any, **kwargs) -> Dict[str, float]:
        """Train or fine-tune the model. Returns metrics."""
        raise NotImplementedError(f"{self._name}.train() not implemented")

    def adapt(self, support_data: Any, **kwargs) -> "BaseNILModel":
        """Few-shot adaptation. Returns adapted copy."""
        raise NotImplementedError(f"{self._name}.adapt() not implemented")

    def load(self, path: str) -> None:
        self._weights_path = path
        self._loaded = True

    def save(self, path: str) -> None:
        self._weights_path = path

    def to(self, device: str) -> "BaseNILModel":
        self._device = device
        return self

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "capability": self._capability,
            "device": self._device,
            "loaded": self._loaded,
            "config": self._config,
        }

    def __repr__(self):
        return f"NILModel('{self._name}', v{self._version}, {self._capability})"


# ══════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════


class ModelRegistry:
    """Central registry for all NIL models.

    Models register themselves here. stacc queries the registry
    to find the best model for a given capability.

    Usage:
        registry = ModelRegistry()

        # Register a model
        registry.register(MyVisionModel())

        # Find models by capability
        vision_models = registry.find("vision")
        best = registry.get_best("vision")

        # List everything available
        registry.list_all()
    """

    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        self._models: Dict[str, List[BaseNILModel]] = {}
        self._default_models: Dict[str, str] = {}  # capability → model name
        self._model_paths: Dict[str, str] = {}  # name → weights path

    @classmethod
    def global_registry(cls) -> "ModelRegistry":
        """Get or create the global singleton registry."""
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def register(self, model: BaseNILModel, default: bool = False):
        """Register a model in the registry."""
        cap = model.capability
        if cap not in self._models:
            self._models[cap] = []

        # Avoid duplicates
        existing_names = [m.name for m in self._models[cap]]
        if model.name not in existing_names:
            self._models[cap].append(model)

        if default or cap not in self._default_models:
            self._default_models[cap] = model.name

    def unregister(self, name: str):
        """Remove a model from the registry."""
        for cap in self._models:
            self._models[cap] = [m for m in self._models[cap] if m.name != name]

    def find(self, capability: str) -> List[BaseNILModel]:
        """Find all models with a given capability."""
        return list(self._models.get(capability, []))

    def get(self, name: str) -> Optional[BaseNILModel]:
        """Get a specific model by name."""
        for models in self._models.values():
            for model in models:
                if model.name == name:
                    return model
        return None

    def get_best(self, capability: str) -> Optional[BaseNILModel]:
        """Get the default/best model for a capability."""
        default_name = self._default_models.get(capability)
        if default_name:
            return self.get(default_name)
        models = self.find(capability)
        return models[0] if models else None

    def set_default(self, capability: str, model_name: str):
        """Set the default model for a capability."""
        self._default_models[capability] = model_name

    def list_capabilities(self) -> List[str]:
        """List all registered capabilities."""
        return list(self._models.keys())

    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        result = []
        for cap, models in self._models.items():
            for model in models:
                is_default = self._default_models.get(cap) == model.name
                result.append(
                    {
                        "name": model.name,
                        "version": model.version,
                        "capability": cap,
                        "device": model.device,
                        "default": is_default,
                    }
                )
        return result

    def summary(self) -> str:
        """Print a summary of the registry."""
        lines = [
            f"NIL Model Registry: {sum(len(v) for v in self._models.values())} models"
        ]
        for cap in sorted(self._models.keys()):
            models = self._models[cap]
            default = self._default_models.get(cap, "")
            for m in models:
                marker = " (default)" if m.name == default else ""
                lines.append(f"  {cap}: {m.name} v{m.version}{marker}")
        return "\n".join(lines)


# ══════════════════════════════════════════
# NIL BRIDGE — connects NIL to stacc
# ══════════════════════════════════════════


class NILBridge:
    """Main integration point between NIL and stacc.

    Attach to a stacc Robot. Loads NIL capabilities on demand.
    Routes sensor data to NIL models, routes NIL outputs to
    robot actuators.

    Usage:
        from nil.bridge import NILBridge

        bridge = NILBridge(robot)
        bridge.load("vision")
        bridge.load("policy")

        @robot.on_update
        def control(dt):
            obs = bridge.observe()
            action = bridge.act(obs)
            robot.set_joint_torques(action)
    """

    def __init__(self, robot=None, registry: Optional[ModelRegistry] = None):
        self.robot = robot
        self.registry = registry or ModelRegistry.global_registry()
        self._active_models: Dict[str, BaseNILModel] = {}
        self._sensor_cache: Dict[str, Any] = {}

        # Auto-discover NIL models if package is installed
        self._auto_discover()

    def _auto_discover(self):
        """Try to import and register NIL models automatically."""
        try:
            import nil

            if hasattr(nil, "register_models"):
                nil.register_models(self.registry)
        except ImportError:
            pass  # NIL not installed, use stacc's built-in AI

    def load(
        self, capability: str, model_name: Optional[str] = None, device: str = "cpu"
    ) -> Optional[BaseNILModel]:
        """Load a NIL capability.

        If model_name is specified, loads that exact model.
        Otherwise, loads the default model for the capability.
        Falls back to stacc's built-in AI if NIL model not available.

        Args:
            capability: e.g. "vision", "policy", "embedding"
            model_name: specific model name (optional)
            device: "cpu", "cuda", "mps"

        Returns:
            The loaded model, or None if unavailable
        """
        if model_name:
            model = self.registry.get(model_name)
        else:
            model = self.registry.get_best(capability)

        if model is None:
            # Fall back to stacc built-in
            model = self._fallback_model(capability)
            if model:
                self.registry.register(model)

        if model:
            model.to(device)
            self._active_models[capability] = model
            return model

        return None

    def get(self, capability: str) -> Optional[BaseNILModel]:
        """Get an active model by capability."""
        return self._active_models.get(capability)

    def observe(self) -> Dict[str, Any]:
        """Collect observations from all robot sensors.

        Returns a unified observation dict that any NIL model can consume.
        """
        obs = {}

        if self.robot:
            obs["joint_positions"] = self.robot.get_joint_positions()
            obs["joint_velocities"] = self.robot.get_joint_velocities()
            obs["time"] = self.robot.time

            for sensor in self.robot.sensors:
                try:
                    data = sensor.read()
                    obs[sensor.name] = data
                except Exception:
                    pass

        self._sensor_cache = obs
        return obs

    def act(self, observation: Optional[Dict[str, Any]] = None) -> Any:
        """Get action from the active policy model.

        Args:
            observation: sensor data (if None, calls observe())

        Returns:
            Action to execute (typically joint torques or positions)
        """
        if observation is None:
            observation = self.observe()

        policy = self._active_models.get("policy")
        if policy:
            return policy.predict(observation)

        return None

    def perceive(self, image=None) -> Dict[str, Any]:
        """Run all perception models on the latest sensor data.

        Returns combined perception results from vision, depth,
        scene understanding, etc.
        """
        results = {}

        vision = self._active_models.get("vision")
        if vision and image is not None:
            results["detections"] = vision.predict(image)

        depth = self._active_models.get("depth_estimation")
        if depth and image is not None:
            results["depth"] = depth.predict(image)

        scene = self._active_models.get("scene")
        if scene and image is not None:
            results["scene"] = scene.predict(image)

        embedding = self._active_models.get("embedding")
        if embedding and image is not None:
            results["embedding"] = embedding.predict(image)

        return results

    def speak(self, text: str) -> Optional[Any]:
        """Text-to-speech via NIL."""
        tts = self._active_models.get("tts")
        if tts:
            return tts.predict(text)
        return None

    def listen(self) -> Optional[str]:
        """Speech-to-text via NIL."""
        stt = self._active_models.get("stt")
        if stt:
            return stt.predict(None)  # listens from microphone
        return None

    def generate_image(self, prompt: str, **kwargs) -> Any:
        """Generate image from text prompt via NIL."""
        gen = self._active_models.get("image_gen")
        if gen:
            return gen.predict(prompt, **kwargs)
        return None

    def generate_motion(self, task_description: str, **kwargs) -> Any:
        """Generate motion trajectory from description via NIL."""
        gen = self._active_models.get("motion_gen")
        if gen:
            return gen.predict(task_description, **kwargs)
        return None

    def adapt(
        self, demonstrations: List[Any], capability: str = "policy"
    ) -> Optional[BaseNILModel]:
        """Few-shot adapt a model using demonstrations.

        This is NIL's core capability — learn from <10 examples.
        """
        model = self._active_models.get(capability)
        if model and hasattr(model, "adapt"):
            adapted = model.adapt(demonstrations)
            self._active_models[capability] = adapted
            return adapted
        return None

    @property
    def active_models(self) -> List[str]:
        return list(self._active_models.keys())

    def status(self) -> Dict[str, Any]:
        return {
            "robot": self.robot.name if self.robot else None,
            "active_models": {
                cap: model.name for cap, model in self._active_models.items()
            },
            "registry_size": len(self.registry.list_all()),
            "available_capabilities": self.registry.list_capabilities(),
        }

    def _fallback_model(self, capability: str) -> Optional[BaseNILModel]:
        """Create a fallback model from stacc's built-in AI."""

        if capability == "vision":
            return StaccVisionAdapter()
        elif capability == "slam":
            return StaccSLAMAdapter()
        elif capability == "nlp":
            return StaccNLPAdapter()
        elif capability == "policy":
            return StaccPolicyAdapter()

        return None


# ══════════════════════════════════════════
# STACC FALLBACK ADAPTERS
# ══════════════════════════════════════════
# These wrap stacc's built-in AI modules as NILModel interface
# so they can be used when NIL package isn't installed.


class StaccVisionAdapter(BaseNILModel):
    """Wraps stacc.ai.vision as a NIL model."""

    def __init__(self):
        super().__init__("stacc_vision", "0.1.0", "vision")
        self._detector = None

    def predict(self, image: Any, **kwargs) -> Any:
        if self._detector is None:
            try:
                from stacc.ai.vision import ObjectDetector

                self._detector = ObjectDetector(backend="opencv")
            except ImportError:
                return []
        return self._detector.detect(image)


class StaccSLAMAdapter(BaseNILModel):
    """Wraps stacc.ai.slam as a NIL model."""

    def __init__(self):
        super().__init__("stacc_slam", "0.1.0", "slam")
        self._slam = None

    def predict(self, inputs: Any, **kwargs) -> Any:
        if self._slam is None:
            try:
                from stacc.ai.slam import EKFSLAM

                self._slam = EKFSLAM()
            except ImportError:
                return None

        if isinstance(inputs, dict):
            v = inputs.get("velocity", 0)
            w = inputs.get("angular_velocity", 0)
            dt = inputs.get("dt", 0.01)
            self._slam.predict(v, w, dt)

            obs = inputs.get("observations", [])
            if obs:
                self._slam.update(obs)

        return {"pose": self._slam.pose, "landmarks": self._slam.landmarks}


class StaccNLPAdapter(BaseNILModel):
    """Wraps stacc.ai.nlp as a NIL model."""

    def __init__(self):
        super().__init__("stacc_nlp", "0.1.0", "nlp")
        self._parser = None

    def predict(self, text: str, **kwargs) -> Any:
        if self._parser is None:
            try:
                from stacc.ai.nlp import CommandParser

                self._parser = CommandParser()
            except ImportError:
                return None
        return self._parser.parse(text).to_dict()


class StaccPolicyAdapter(BaseNILModel):
    """Simple PID-based policy as fallback."""

    def __init__(self):
        super().__init__("stacc_pid_policy", "0.1.0", "policy")

    def predict(self, inputs: Any, **kwargs) -> Any:
        if isinstance(inputs, dict):
            positions = inputs.get("joint_positions", [])
            targets = inputs.get("targets", [0.0] * len(positions))
            kp = kwargs.get("kp", 100.0)
            return [kp * (t - p) for t, p in zip(targets, positions)]
        return []


# ══════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════


class TrainingPipeline:
    """Train NIL models inside stacc simulation.

    Usage:
        pipeline = TrainingPipeline(robot, model, env_config)
        metrics = pipeline.train(
            num_epochs=1000,
            tasks_per_batch=5,
            support_episodes=10,
        )
    """

    def __init__(self, robot, model: BaseNILModel, env_config: Optional[Dict] = None):
        self.robot = robot
        self.model = model
        self.env_config = env_config or {}
        self._env = None

    def _create_env(self):
        """Create stacc Gym environment for training."""
        try:
            from stacc.sim.gym_env import StaccEnv

            self._env = StaccEnv(
                robot=self.robot,
                render=self.env_config.get("render", False),
                max_steps=self.env_config.get("max_steps", 1000),
                dt=self.env_config.get("dt", 0.002),
            )
        except ImportError:
            raise RuntimeError("stacc.sim.gym_env required for training")

    def train(self, num_epochs: int = 100, **kwargs) -> Dict[str, float]:
        """Run training loop."""
        if self._env is None:
            self._create_env()

        if hasattr(self.model, "train"):
            return self.model.train(self._env, num_epochs=num_epochs, **kwargs)

        return {"error": "model does not support training"}

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """Evaluate trained model."""
        if self._env is None:
            self._create_env()

        total_reward = 0
        successes = 0

        for ep in range(num_episodes):
            obs, _ = self._env.reset()
            ep_reward = 0

            for step in range(self._env.max_steps):
                action = self.model.predict(obs)
                obs, reward, terminated, truncated, info = self._env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    if terminated:
                        successes += 1
                    break

            total_reward += ep_reward

        return {
            "mean_reward": total_reward / num_episodes,
            "success_rate": successes / num_episodes,
            "num_episodes": num_episodes,
        }


# ══════════════════════════════════════════
# DEPLOYMENT PIPELINE
# ══════════════════════════════════════════


class DeploymentPipeline:
    """Deploy NIL models to stacc hardware targets.

    Handles model export, optimization, and transfer to edge devices.

    Usage:
        deployer = DeploymentPipeline(model, target="raspberry_pi")
        deployer.optimize()       # quantize, prune
        deployer.export("onnx")   # export to edge format
        deployer.deploy("192.168.1.100")  # push to robot
    """

    def __init__(self, model: BaseNILModel, target: str = "cpu"):
        self.model = model
        self.target = target
        self._export_path: Optional[str] = None

    def optimize(self, method: str = "quantize") -> Dict[str, Any]:
        """Optimize model for edge deployment."""
        original_size = self.model.metadata.get("params", 0)

        if method == "quantize":
            # INT8 quantization reduces size ~4x
            return {
                "method": "quantize",
                "original_params": original_size,
                "optimized_params": original_size // 4,
                "compression_ratio": 4.0,
                "target": self.target,
            }
        elif method == "prune":
            return {
                "method": "prune",
                "original_params": original_size,
                "optimized_params": int(original_size * 0.3),
                "compression_ratio": 3.3,
            }

        return {"method": "none"}

    def export(self, format: str = "onnx", output_path: Optional[str] = None) -> str:
        """Export model to deployment format.

        Formats: onnx, tflite, torchscript, coreml
        """
        path = output_path or f"/tmp/{self.model.name}.{format}"
        self._export_path = path

        if format == "onnx":
            try:
                import torch

                if hasattr(self.model, "_model") and isinstance(
                    self.model._model, torch.nn.Module
                ):
                    dummy = torch.randn(1, self.model.metadata.get("input_dim", 10))
                    torch.onnx.export(self.model._model, dummy, path)
            except ImportError:
                self.model.save(path)
        else:
            self.model.save(path)

        return path

    def deploy(
        self, host: str, user: str = "pi", remote_path: str = "/home/pi/nil_model/"
    ):
        """Deploy exported model to remote robot via SSH."""
        if self._export_path is None:
            self.export("onnx")

        try:
            from stacc.hal.deploy import Deployer

            deployer = Deployer(host=host, user=user)
            if deployer.connect():
                deployer.upload(self._export_path, remote_path)
                deployer.run(f"python3 -c \"import nil; nil.load('{remote_path}')\"")
                deployer.close()
                return True
        except ImportError:
            pass

        return False
