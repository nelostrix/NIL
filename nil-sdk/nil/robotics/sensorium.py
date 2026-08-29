"""Embodied Robotics Sensorium & Multimodal Sensory Ingest.

Buffers, synchronizes, and cleans high-frequency sensory observations:
- Joint State Telemetry: q, q_dot, tau, temperature
- IMU: 1000Hz linear accelerations, angular velocities
- 3D Vision & Depth: Point clouds, RGB-D frames, bounding detections
- Force / Torque / Contact: 6D wrist wrenches, foot contact switches
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time


class SensorFrame:
    """A synchronized multimodal snapshot of robot sensory observations."""

    def __init__(
        self,
        timestamp: Optional[float] = None,
        joint_positions: Optional[List[float]] = None,
        joint_velocities: Optional[List[float]] = None,
        joint_torques: Optional[List[float]] = None,
        imu_accel: Optional[List[float]] = None,
        imu_gyro: Optional[List[float]] = None,
        contact_states: Optional[List[bool]] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        raw_metadata: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.joint_positions = list(joint_positions or [])
        self.joint_velocities = list(joint_velocities or [])
        self.joint_torques = list(joint_torques or [])
        self.imu_accel = list(imu_accel or [0.0, 0.0, 9.81])
        self.imu_gyro = list(imu_gyro or [0.0, 0.0, 0.0])
        self.contact_states = list(contact_states or [])
        self.detected_objects = list(detected_objects or [])
        self.raw_metadata = dict(raw_metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "joint_torques": self.joint_torques,
            "imu_accel": self.imu_accel,
            "imu_gyro": self.imu_gyro,
            "contact_states": self.contact_states,
            "detected_objects": self.detected_objects,
            "raw_metadata": self.raw_metadata
        }


class EmbodiedSensorium:
    """Sliding-Window Temporal Sensorium Buffer for Embodied AI."""

    def __init__(self, buffer_size: int = 200):
        self.buffer_size = int(buffer_size)
        self.frames: List[SensorFrame] = []

    def ingest(self, frame: SensorFrame) -> None:
        """Push a synchronized sensory frame into the rolling buffer."""
        self.frames.append(frame)
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)

    def get_latest(self) -> Optional[SensorFrame]:
        """Return the most recent sensor observation."""
        return self.frames[-1] if self.frames else None

    def get_joint_trajectory(self) -> List[List[float]]:
        """Retrieve recent history of joint angles for kinematic analysis."""
        return [f.joint_positions for f in self.frames if f.joint_positions]

    def has_contact(self) -> bool:
        """Check if any contact switches are active in the latest frame."""
        latest = self.get_latest()
        if not latest or not latest.contact_states:
            return False
        return any(latest.contact_states)
