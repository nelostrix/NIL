"""NIL-STACC Product System — licensing, tiers, and model access.

NIL and stacc are independent products that work together.

NIL (standalone):
    - AI platform — train, host, and serve models
    - Anyone can use NIL models for any purpose (NLP, vision, generation)
    - Models available via API, Python SDK, or download
    - Like OpenAI but open-core and robotics-first

stacc (standalone):
    - Robotics platform — define, simulate, deploy robots
    - Works without NIL using built-in AI (OpenCV, regex NLP, PID control)
    - Like Unreal Engine but for robotics

NIL + stacc (bundled in Pro/Enterprise):
    - NIL models auto-integrate with stacc robots
    - Train NIL models inside stacc simulation
    - Deploy NIL models to stacc hardware
    - Few-shot adaptation for any robot task

Product Tiers:
    ┌────────────────────────────────────────────────────────────┐
    │ stacc Community (FREE, MIT)                               │
    │ - Full robotics platform                                  │
    │ - MuJoCo simulation                                       │
    │ - Built-in AI (OpenCV vision, regex NLP, PID, A*)         │
    │ - 17 microcontroller targets                              │
    │ - Community support                                       │
    ├────────────────────────────────────────────────────────────┤
    │ NIL Free (FREE, Apache 2.0)                               │
    │ - Core meta-learning framework                            │
    │ - 3 pre-trained models (reach, push, navigate)            │
    │ - Python SDK for inference                                 │
    │ - Community support                                       │
    ├────────────────────────────────────────────────────────────┤
    │ stacc Pro ($49/mo or $499/yr)                             │
    │ - Everything in stacc Community                           │
    │ - NIL model bundle (10+ pre-trained models)               │
    │ - Vision: YOLO-Robotics, depth estimation, segmentation   │
    │ - Control: pre-trained policies for arm, quadruped, drone  │
    │ - SLAM: neural SLAM, visual odometry                      │
    │ - NLP: LLM-powered command parsing                        │
    │ - Generation: motion trajectory synthesis                  │
    │ - Priority support + Slack access                         │
    │ - Commercial license                                      │
    ├────────────────────────────────────────────────────────────┤
    │ stacc Enterprise (Custom pricing)                         │
    │ - Everything in Pro                                       │
    │ - Custom model training on your data                      │
    │ - On-premise deployment (air-gapped)                      │
    │ - Dedicated support engineer                              │
    │ - SLA guarantees                                          │
    │ - White-label option                                      │
    └────────────────────────────────────────────────────────────┘

NIL Model Marketplace:
    - Any developer can publish models to the NIL marketplace
    - Models are versioned, tested, and reviewed
    - Free models + paid premium models
    - Revenue share: 70% creator / 30% Nelo Robotics
    - stacc Pro users get all marketplace models included
"""

from __future__ import annotations

import json
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional

# ══════════════════════════════════════════
# PRODUCT TIERS
# ══════════════════════════════════════════


class Tier(Enum):
    COMMUNITY = "community"  # stacc free, no NIL models
    NIL_FREE = "nil_free"  # NIL free, 3 models
    PRO = "pro"  # stacc + NIL bundle
    ENTERPRISE = "enterprise"  # custom everything


class License:
    """License and entitlement for a stacc/NIL installation."""

    def __init__(self, tier: Tier = Tier.COMMUNITY, key: Optional[str] = None):
        self.tier = tier
        self.key = key
        self._valid = False
        self._expires: Optional[float] = None
        self._features: Dict[str, bool] = {}

        self._set_features()

    def _set_features(self):
        """Set feature flags based on tier."""
        base = {
            "simulation": True,
            "urdf_parser": True,
            "sensors": True,
            "path_planning": True,
            "behavior_trees": True,
            "hal": True,
            "deploy": True,
            "gym_env": True,
            "recording": True,
            "world_builder": True,
            "gaits": True,
            "grippers": True,
            "builtin_vision": True,
            "builtin_slam": True,
            "builtin_nlp": True,
            # NIL model access
            "nil_models": False,
            "nil_vision_pro": False,
            "nil_policy_pro": False,
            "nil_slam_pro": False,
            "nil_llm": False,
            "nil_generation": False,
            "nil_marketplace": False,
            "nil_custom_training": False,
            # Support
            "priority_support": False,
            "slack_access": False,
            "dedicated_engineer": False,
            "sla": False,
            "commercial_license": False,
            "white_label": False,
        }

        if self.tier == Tier.NIL_FREE:
            base["nil_models"] = True  # 3 free models only

        elif self.tier == Tier.PRO:
            base["nil_models"] = True
            base["nil_vision_pro"] = True
            base["nil_policy_pro"] = True
            base["nil_slam_pro"] = True
            base["nil_llm"] = True
            base["nil_generation"] = True
            base["nil_marketplace"] = True
            base["priority_support"] = True
            base["slack_access"] = True
            base["commercial_license"] = True

        elif self.tier == Tier.ENTERPRISE:
            for k in base:
                base[k] = True

        self._features = base

    def has_feature(self, feature: str) -> bool:
        return self._features.get(feature, False)

    def can_use_model(self, model_tier: str) -> bool:
        """Check if this license can use a model of given tier."""
        if model_tier == "free":
            return True
        if model_tier == "pro" and self.tier in (Tier.PRO, Tier.ENTERPRISE):
            return True
        if model_tier == "enterprise" and self.tier == Tier.ENTERPRISE:
            return True
        return False

    @property
    def is_pro(self) -> bool:
        return self.tier in (Tier.PRO, Tier.ENTERPRISE)

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "valid": self._valid,
            "features_enabled": sum(1 for v in self._features.values() if v),
            "features_total": len(self._features),
            "nil_models": self._features.get("nil_models", False),
            "commercial": self._features.get("commercial_license", False),
        }


# ══════════════════════════════════════════
# NIL MODEL CATALOG
# ══════════════════════════════════════════


class ModelInfo:
    """Metadata for a model in the NIL catalog."""

    def __init__(
        self,
        name: str,
        capability: str,
        version: str = "1.0.0",
        tier: str = "free",
        description: str = "",
        params: int = 0,
        size_mb: float = 0,
        latency_ms: float = 0,
        accuracy: float = 0,
        author: str = "Nelo Robotics",
        tags: Optional[List[str]] = None,
        requires: Optional[List[str]] = None,
        runtime: str = "python",
    ):
        self.name = name
        self.capability = capability
        self.version = version
        self.tier = tier  # "free", "pro", "enterprise"
        self.description = description
        self.params = params
        self.size_mb = size_mb
        self.latency_ms = latency_ms
        self.accuracy = accuracy
        self.author = author
        self.tags = tags or []
        self.requires = requires or []
        self.runtime = runtime  # "python", "rust", "onnx", "tflite"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capability": self.capability,
            "version": self.version,
            "tier": self.tier,
            "description": self.description,
            "params": self.params,
            "size_mb": self.size_mb,
            "latency_ms": self.latency_ms,
            "accuracy": self.accuracy,
            "author": self.author,
            "tags": self.tags,
            "runtime": self.runtime,
        }


# Official NIL model catalog
NIL_CATALOG: List[ModelInfo] = [
    # ── FREE TIER (3 models) ──
    ModelInfo(
        name="nil-reach-v1",
        capability="policy",
        tier="free",
        description="Meta-learned reaching policy. Adapts to new targets in <10 demos.",
        params=12_000,
        size_mb=0.5,
        latency_ms=0.3,
        accuracy=0.92,
        tags=["manipulation", "meta-learning", "few-shot"],
    ),
    ModelInfo(
        name="nil-push-v1",
        capability="policy",
        tier="free",
        description="Object pushing policy with few-shot adaptation.",
        params=15_000,
        size_mb=0.6,
        latency_ms=0.4,
        accuracy=0.85,
        tags=["manipulation", "push", "contact"],
    ),
    ModelInfo(
        name="nil-navigate-v1",
        capability="navigation",
        tier="free",
        description="Basic 2D navigation with obstacle avoidance.",
        params=20_000,
        size_mb=0.8,
        latency_ms=0.5,
        accuracy=0.88,
        tags=["navigation", "obstacle-avoidance"],
    ),
    # ── PRO TIER (bundled with stacc Pro) ──
    ModelInfo(
        name="nil-vision-v2",
        capability="vision",
        tier="pro",
        description="Robotics-optimized object detector. 80 classes, real-time on edge.",
        params=3_500_000,
        size_mb=14,
        latency_ms=8,
        accuracy=0.91,
        tags=["detection", "real-time", "edge"],
        runtime="onnx",
    ),
    ModelInfo(
        name="nil-depth-v1",
        capability="depth_estimation",
        tier="pro",
        description="Monocular depth estimation for any camera.",
        params=5_000_000,
        size_mb=20,
        latency_ms=15,
        accuracy=0.89,
        tags=["depth", "monocular", "3D"],
        runtime="onnx",
    ),
    ModelInfo(
        name="nil-segment-v1",
        capability="vision",
        tier="pro",
        description="Instance segmentation for robotic manipulation scenes.",
        params=8_000_000,
        size_mb=32,
        latency_ms=20,
        accuracy=0.87,
        tags=["segmentation", "instance", "manipulation"],
        runtime="onnx",
    ),
    ModelInfo(
        name="nil-embedding-v1",
        capability="embedding",
        tier="pro",
        description="Visual-semantic embedding. Maps images and text to shared space.",
        params=12_000_000,
        size_mb=48,
        latency_ms=5,
        tags=["embedding", "multimodal", "CLIP-style"],
        runtime="onnx",
    ),
    ModelInfo(
        name="nil-grasp-v1",
        capability="grasp_planner",
        tier="pro",
        description="6-DOF grasp pose estimation from point clouds.",
        params=4_000_000,
        size_mb=16,
        latency_ms=25,
        accuracy=0.83,
        tags=["grasping", "6dof", "point-cloud"],
    ),
    ModelInfo(
        name="nil-slam-neural-v1",
        capability="slam",
        tier="pro",
        description="Neural SLAM with learned feature extraction. Better than ORB-SLAM in indoor.",
        params=6_000_000,
        size_mb=24,
        latency_ms=12,
        accuracy=0.94,
        tags=["slam", "neural", "indoor"],
    ),
    ModelInfo(
        name="nil-policy-arm-v2",
        capability="policy",
        tier="pro",
        description="Pre-trained manipulation policy for 6/7-DOF arms. Few-shot adaptable.",
        params=500_000,
        size_mb=2,
        latency_ms=0.5,
        accuracy=0.90,
        tags=["manipulation", "arm", "few-shot"],
    ),
    ModelInfo(
        name="nil-policy-quadruped-v1",
        capability="policy",
        tier="pro",
        description="Locomotion policy for quadruped robots. Terrain-adaptive.",
        params=300_000,
        size_mb=1.2,
        latency_ms=0.3,
        accuracy=0.88,
        tags=["locomotion", "quadruped", "terrain"],
    ),
    ModelInfo(
        name="nil-policy-drone-v1",
        capability="policy",
        tier="pro",
        description="Flight control policy for multirotor drones. Wind-resistant.",
        params=200_000,
        size_mb=0.8,
        latency_ms=0.2,
        accuracy=0.93,
        tags=["flight", "drone", "multirotor"],
    ),
    ModelInfo(
        name="nil-command-llm-v1",
        capability="llm",
        tier="pro",
        description="Small LLM fine-tuned for robot commands. Runs on edge.",
        params=50_000_000,
        size_mb=200,
        latency_ms=50,
        tags=["llm", "commands", "edge", "small"],
        runtime="onnx",
    ),
    ModelInfo(
        name="nil-motion-gen-v1",
        capability="motion_gen",
        tier="pro",
        description="Generate smooth motion trajectories from text descriptions.",
        params=10_000_000,
        size_mb=40,
        latency_ms=30,
        tags=["generation", "motion", "trajectory"],
    ),
    ModelInfo(
        name="nil-vo-v1",
        capability="slam",
        tier="pro",
        description="Visual odometry with learned features. Works in textureless environments.",
        params=3_000_000,
        size_mb=12,
        latency_ms=10,
        accuracy=0.91,
        tags=["odometry", "visual", "learned"],
        runtime="onnx",
    ),
    # ── ENTERPRISE TIER ──
    ModelInfo(
        name="nil-foundation-v1",
        capability="policy",
        tier="enterprise",
        description="Foundation model for robotics. Pre-trained on 1M+ trajectories. Fine-tune for any task.",
        params=100_000_000,
        size_mb=400,
        latency_ms=5,
        tags=["foundation", "pre-trained", "universal"],
    ),
    ModelInfo(
        name="nil-world-model-v1",
        capability="world_model",
        tier="enterprise",
        description="Predictive world model. Plans in imagination. Reduces real-world trials 10x.",
        params=30_000_000,
        size_mb=120,
        latency_ms=8,
        tags=["world-model", "planning", "imagination"],
    ),
    ModelInfo(
        name="nil-sim2real-v1",
        capability="sim2real",
        tier="enterprise",
        description="Domain randomization + adaptation for sim-to-real transfer.",
        params=15_000_000,
        size_mb=60,
        tags=["sim2real", "transfer", "domain-adaptation"],
    ),
]


def get_catalog(tier: Optional[str] = None) -> List[ModelInfo]:
    """Get models from the catalog, optionally filtered by tier."""
    if tier:
        return [m for m in NIL_CATALOG if m.tier == tier]
    return list(NIL_CATALOG)


def get_model_info(name: str) -> Optional[ModelInfo]:
    """Look up a model by name."""
    for m in NIL_CATALOG:
        if m.name == name:
            return m
    return None


def catalog_summary() -> str:
    """Print catalog summary."""
    lines = ["NIL Model Catalog"]
    lines.append(f"  Total models: {len(NIL_CATALOG)}")

    for tier in ["free", "pro", "enterprise"]:
        models = get_catalog(tier)
        lines.append(f"\n  {tier.upper()} ({len(models)} models):")
        for m in models:
            size = f"{m.size_mb}MB" if m.size_mb > 0 else ""
            runtime = f"[{m.runtime}]" if m.runtime != "python" else ""
            lines.append(f"    {m.name} — {m.description[:60]}  {size} {runtime}")

    return "\n".join(lines)


# ══════════════════════════════════════════
# MODEL MARKETPLACE
# ══════════════════════════════════════════


class MarketplaceModel:
    """A model published to the NIL marketplace by a third-party developer."""

    def __init__(
        self,
        name: str,
        author: str,
        capability: str,
        description: str,
        price: float = 0.0,  # 0 = free
        currency: str = "USD",
        downloads: int = 0,
        rating: float = 0.0,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.author = author
        self.capability = capability
        self.description = description
        self.price = price
        self.currency = currency
        self.downloads = downloads
        self.rating = rating
        self.version = version
        self.tags = tags or []
        self.published_at: Optional[float] = None

    @property
    def is_free(self) -> bool:
        return self.price <= 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "author": self.author,
            "capability": self.capability,
            "description": self.description,
            "price": self.price,
            "currency": self.currency,
            "downloads": self.downloads,
            "rating": self.rating,
            "version": self.version,
            "tags": self.tags,
            "free": self.is_free,
        }


class Marketplace:
    """NIL Model Marketplace — publish, discover, and install models.

    Revenue share: 70% creator / 30% Nelo Robotics

    Usage:
        mp = Marketplace()

        # Discover models
        vision_models = mp.search(capability="vision")
        top_models = mp.trending()

        # Install a model
        mp.install("community/awesome-detector-v2")

        # Publish your model
        mp.publish(my_model, price=9.99)
    """

    REVENUE_SHARE_CREATOR = 0.70
    REVENUE_SHARE_PLATFORM = 0.30

    def __init__(self):
        self._models: List[MarketplaceModel] = []
        self._installed: List[str] = []

    def publish(self, model: MarketplaceModel) -> bool:
        """Publish a model to the marketplace."""
        model.published_at = time.time()
        self._models.append(model)
        return True

    def search(
        self,
        query: Optional[str] = None,
        capability: Optional[str] = None,
        free_only: bool = False,
        sort_by: str = "downloads",
    ) -> List[MarketplaceModel]:
        """Search the marketplace."""
        results = list(self._models)

        if capability:
            results = [m for m in results if m.capability == capability]

        if free_only:
            results = [m for m in results if m.is_free]

        if query:
            q = query.lower()
            results = [
                m
                for m in results
                if q in m.name.lower()
                or q in m.description.lower()
                or any(q in t for t in m.tags)
            ]

        if sort_by == "downloads":
            results.sort(key=lambda m: m.downloads, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda m: m.rating, reverse=True)
        elif sort_by == "price":
            results.sort(key=lambda m: m.price)

        return results

    def trending(self, limit: int = 10) -> List[MarketplaceModel]:
        return sorted(self._models, key=lambda m: m.downloads, reverse=True)[:limit]

    def install(self, model_name: str) -> bool:
        """Install a marketplace model."""
        self._installed.append(model_name)
        return True

    @property
    def installed(self) -> List[str]:
        return list(self._installed)

    def calculate_payout(
        self, model_name: str, total_revenue: float
    ) -> Dict[str, float]:
        """Calculate revenue split for a model."""
        return {
            "total": total_revenue,
            "creator": total_revenue * self.REVENUE_SHARE_CREATOR,
            "platform": total_revenue * self.REVENUE_SHARE_PLATFORM,
        }
