"""Robotics assembly cell — a full Nil SDK domain-specific application.

Simulates a dual-arm robotic workcell with force feedback, visual servoing,
task planning, and continuous self-improvement via evolution + multi-agent
coordination.

Run:
    ANTHROPIC_API_KEY=sk-... python examples/robotics_assembly.py
    # or without keys (stub fallback):
    python examples/robotics_assembly.py
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from nil import Agent, Domain, EvolutionConfig, PersistentMemory, tool

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("robotics")


# ── Simulated robot state (in-process, no real hardware) ──────────────

@dataclass
class JointState:
    q: list[float]  # joint angles (radians)
    dq: list[float]  # joint velocities

@dataclass
class Pose:
    x: float; y: float; z: float
    roll: float = 0.0; pitch: float = 0.0; yaw: float = 0.0

@dataclass
class GripperState:
    position: float = 0.0    # 0 = closed, 1 = fully open
    force: float = 0.0       # N
    object_held: bool = False

@dataclass
class Part:
    part_id: str
    type: str                # "gear", "shaft", "bearing", "housing"
    pose: Pose
    weight: float = 0.0

class WeldState(Enum):
    IDLE = "idle"
    WELDING = "welding"
    COOLDOWN = "cooldown"

class ConveyorSpeed(Enum):
    STOP = 0
    SLOW = 1
    FAST = 2

@dataclass
class CellState:
    arm1_joints: JointState = field(default_factory=lambda: JointState([0.0]*6, [0.0]*6))
    arm2_joints: JointState = field(default_factory=lambda: JointState([0.0]*6, [0.0]*6))
    arm1_pose: Pose = field(default_factory=lambda: Pose(0,0,0))
    arm2_pose: Pose = field(default_factory=lambda: Pose(0,0,0))
    gripper1: GripperState = field(default_factory=GripperState)
    gripper2: GripperState = field(default_factory=GripperState)
    conveyor_speed: ConveyorSpeed = ConveyorSpeed.STOP
    parts_on_conveyor: list[Part] = field(default_factory=list)
    weld_gun: WeldState = WeldState.IDLE
    completed_assemblies: int = 0
    cycle_time: float = 0.0
    safety_stop: bool = False
    error_log: list[str] = field(default_factory=list)
    collision_warning: bool = False


_cell_state = CellState()


def _init_cell():
    global _cell_state
    _cell_state = CellState()
    _cell_state.arm1_pose = Pose(0.3, 0.0, 0.5)
    _cell_state.arm2_pose = Pose(-0.3, 0.0, 0.5)
    _cell_state.gripper1 = GripperState(position=1.0)
    _cell_state.gripper2 = GripperState(position=1.0)
    _cell_state.parts_on_conveyor = [
        Part("gear-001", "gear", Pose(0.0, -0.8, 0.0), 0.5),
        Part("shaft-001", "shaft", Pose(0.2, -0.8, 0.0), 1.2),
        Part("bearing-001", "bearing", Pose(-0.2, -0.8, 0.0), 0.3),
    ]
    _cell_state.weld_gun = WeldState.IDLE


# ── Tool definitions ─────────────────────────────────────────────────

@tool
def move_to(arm: str, x: float, y: float, z: float,
            speed: float = 0.5, blend_radius: float = 0.0) -> str:
    """Move a robot arm to an absolute Cartesian position.

    Args:
        arm: "arm1" or "arm2"
        x, y, z: target position in meters
        speed: fraction of max speed (0.0-1.0)
        blend_radius: radius in m for trajectory blending (smooth transitions)
    """
    global _cell_state
    if _cell_state.safety_stop:
        return "SAFETY STOP ACTIVE — move rejected"

    if arm == "arm1":
        _cell_state.arm1_pose = Pose(x, y, z)
    elif arm == "arm2":
        _cell_state.arm2_pose = Pose(x, y, z)
    else:
        return f"Unknown arm: {arm}"

    time.sleep(0.05 * (1.0 / max(speed, 0.1)))
    return f"{arm} moved to ({x:.3f}, {y:.3f}, {z:.3f}) at speed {speed}"


@tool
def move_relative(arm: str, dx: float, dy: float, dz: float,
                  speed: float = 0.5) -> str:
    """Move a robot arm relative to its current position.

    Args:
        arm: "arm1" or "arm2"
        dx, dy, dz: displacement in meters
        speed: fraction of max speed (0.0-1.0)
    """
    global _cell_state
    current = _cell_state.arm1_pose if arm == "arm1" else _cell_state.arm2_pose
    nx = current.x + dx
    ny = current.y + dy
    nz = current.z + dz
    return move_to(arm, nx, ny, nz, speed)


@tool
def gripper_control(arm: str, action: str, force: float = 5.0) -> str:
    """Open or close a gripper.

    Args:
        arm: "arm1" or "arm2"
        action: "open" or "close"
        force: grip force in Newtons (only for "close")
    """
    global _cell_state
    gripper = _cell_state.gripper1 if arm == "arm1" else _cell_state.gripper2

    if action == "open":
        gripper.position = 1.0
        gripper.force = 0.0
        gripper.object_held = False
        return f"{arm} gripper opened"
    elif action == "close":
        gripper.position = 0.0
        gripper.force = force
        gripper.object_held = True
        return f"{arm} gripper closed with {force}N"
    return f"Unknown gripper action: {action}"


@tool
def read_sensor(sensor: str) -> str:
    """Read a sensor value from the workcell.

    Sensors:
      "force_1", "force_2"    — gripper force sensors (N)
      "vision_pose"           — estimated part pose from vision
      "torque_1", "torque_2"  — joint torque sensors (Nm)
      "proximity"             — object detection in pickup zone
      "conveyor_position"     — belt encoder position (m)
      "weld_temp"             — weld gun tip temperature (C)
      "cycle_count"           — total completed assemblies
    """
    global _cell_state
    sim = {
        "force_1": json.dumps({"sensor": "force_1", "value": _cell_state.gripper1.force, "unit": "N"}),
        "force_2": json.dumps({"sensor": "force_2", "value": _cell_state.gripper2.force, "unit": "N"}),
        "vision_pose": json.dumps({"sensor": "vision_pose", "parts_detected": len(_cell_state.parts_on_conveyor), "target_exists": len(_cell_state.parts_on_conveyor) > 0}),
        "torque_1": json.dumps({"sensor": "torque_1", "value": 2.3 + 0.1 * math.sin(time.time()), "unit": "Nm"}),
        "torque_2": json.dumps({"sensor": "torque_2", "value": 1.8 + 0.1 * math.cos(time.time()), "unit": "Nm"}),
        "proximity": json.dumps({"sensor": "proximity", "object_detected": len(_cell_state.parts_on_conveyor) > 0, "distance_m": 0.15}),
        "conveyor_position": json.dumps({"sensor": "conveyor_position", "position_m": 12.4, "speed": _cell_state.conveyor_speed.value}),
        "weld_temp": json.dumps({"sensor": "weld_temp", "temperature_C": 180 if _cell_state.weld_gun == WeldState.WELDING else 30, "unit": "C"}),
        "cycle_count": json.dumps({"sensor": "cycle_count", "value": _cell_state.completed_assemblies}),
    }
    return sim.get(sensor, json.dumps({"error": f"Unknown sensor: {sensor}"}))


@tool
def conveyor_control(action: str) -> str:
    """Start, stop, or reverse the conveyor belt.

    Args:
        action: "start", "stop", "reverse", or "fast"
    """
    global _cell_state
    mapping = {"start": ConveyorSpeed.SLOW, "fast": ConveyorSpeed.FAST,
               "stop": ConveyorSpeed.STOP, "reverse": ConveyorSpeed.SLOW}
    if action not in mapping:
        return f"Unknown conveyor action: {action}"

    _cell_state.conveyor_speed = mapping[action]
    direction = "reverse" if action == "reverse" else "forward"
    speed_str = "fast" if action == "fast" else "slow" if action != "stop" else ""
    return f"Conveyor {direction} {speed_str}".strip()


@tool
def weld_joint(joint_id: str, duration_s: float = 2.0) -> str:
    """Execute a weld at the specified joint.

    Args:
        joint_id: identifier for the weld joint
        duration_s: weld duration in seconds
    """
    global _cell_state
    if _cell_state.safety_stop:
        return "SAFETY STOP ACTIVE — weld rejected"
    if _cell_state.weld_gun == WeldState.WELDING:
        return "Weld gun already in use"

    _cell_state.weld_gun = WeldState.WELDING
    time.sleep(min(duration_s, 0.5))  # simulated
    _cell_state.weld_gun = WeldState.COOLDOWN
    return f"Joint {joint_id} welded ({duration_s}s) — quality: OK"


@tool
def check_collision_risk() -> str:
    """Check if the two arms are at risk of collision."""
    global _cell_state
    dx = abs(_cell_state.arm1_pose.x - _cell_state.arm2_pose.x)
    dy = abs(_cell_state.arm1_pose.y - _cell_state.arm2_pose.y)
    dz = abs(_cell_state.arm1_pose.z - _cell_state.arm2_pose.z)
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    risk = distance < 0.15
    _cell_state.collision_warning = risk
    return json.dumps({
        "distance_m": round(distance, 3),
        "risk": risk,
        "warning": "Arm separation below safety threshold" if risk else "Safe",
    })


@tool
def emergency_stop() -> str:
    """Immediately stop all workcell motion."""
    global _cell_state
    _cell_state.safety_stop = True
    _cell_state.conveyor_speed = ConveyorSpeed.STOP
    return "EMERGENCY STOP engaged — all motion halted. Call reset_safety to resume."


@tool
def reset_safety() -> str:
    """Reset the safety stop after an emergency stop."""
    global _cell_state
    _cell_state.safety_stop = False
    _cell_state.collision_warning = False
    _cell_state.weld_gun = WeldState.IDLE
    return "Safety reset — workcell operational"


@tool
def report_cell_status() -> str:
    """Get a full status report of the workcell."""
    global _cell_state
    status = {
        "arm1_pose": {"x": round(_cell_state.arm1_pose.x,3), "y": round(_cell_state.arm1_pose.y,3), "z": round(_cell_state.arm1_pose.z,3)},
        "arm2_pose": {"x": round(_cell_state.arm2_pose.x,3), "y": round(_cell_state.arm2_pose.y,3), "z": round(_cell_state.arm2_pose.z,3)},
        "gripper1": {"object_held": _cell_state.gripper1.object_held, "force": _cell_state.gripper1.force},
        "gripper2": {"object_held": _cell_state.gripper2.object_held, "force": _cell_state.gripper2.force},
        "conveyor": _cell_state.conveyor_speed.value,
        "parts_on_conveyor": len(_cell_state.parts_on_conveyor),
        "completed_assemblies": _cell_state.completed_assemblies,
        "weld_state": _cell_state.weld_gun.value,
        "safety_stop": _cell_state.safety_stop,
        "collision_warning": _cell_state.collision_warning,
    }
    return json.dumps(status, indent=2)


@tool
def compute_grasp_pose(part_type: str, approach_angle: str = "top") -> str:
    """Compute an optimal grasp pose for a given part type.

    Args:
        part_type: "gear", "shaft", "bearing", or "housing"
        approach_angle: "top", "side", or "angled"
    """
    poses = {
        "gear":    {"x": 0.0, "y": 0.0, "z": 0.12, "roll": 0, "pitch": 0, "yaw": 0},
        "shaft":   {"x": 0.0, "y": 0.0, "z": 0.15, "roll": 1.57, "pitch": 0, "yaw": 0},
        "bearing": {"x": 0.0, "y": 0.0, "z": 0.08, "roll": 0, "pitch": 0, "yaw": 0},
        "housing": {"x": 0.0, "y": 0.0, "z": 0.20, "roll": 0, "pitch": 0, "yaw": 0},
    }
    base = poses.get(part_type, poses["gear"])
    if approach_angle == "side":
        base["roll"] += 1.57
    elif approach_angle == "angled":
        base["pitch"] += 0.785
    return json.dumps({"grasp_pose": base, "approach_vector": [0, 0, -1]})


def _assembly_success(outcome) -> bool:
    """Success signal: assembly must increment cycle count."""
    return outcome.success


# ── Main application ─────────────────────────────────────────────────

def main():
    _init_cell()

    robotics_tools = [
        move_to, move_relative, gripper_control, read_sensor,
        conveyor_control, weld_joint, check_collision_risk,
        emergency_stop, reset_safety, report_cell_status,
        compute_grasp_pose,
    ]

    domain = Domain(
        name="robotic-assembly-cell",
        tools=robotics_tools,
        objectives=[
            "Perform precision pick-and-place of gears, shafts, and bearings",
            "Execute weld joints with force feedback and collision avoidance",
            "Coordinate dual arms for complex assembly sequences",
            "Detect and recover from faults (collision risk, grip loss)",
            "Optimize cycle time through learned task sequencing",
        ],
        success_signal=_assembly_success,
    )

    evo_config = EvolutionConfig(
        enabled=True,
        strategy="continuous",
        skill_extraction=True,
        cost_optimization=True,
        drift_detection=True,
    )

    agent = Agent(
        domain=domain,
        memory=PersistentMemory("./robotics_store"),
        evolution=evo_config,
        enable_economy=True,          # cost-aware cognition
        enable_curriculum=True,       # adaptive task difficulty
        enable_multi_agent=True,      # parallel task decomposition
        provider="openai",            # set OPENAI_API_KEY or OPENAI_BASE_URL for local
        base_model="gpt-4o",
    )

    tasks = [
        "Gear assembly: pick gear-001 from conveyor at (0.0, -0.8, 0.0), "
        "place at assembly station (0.4, 0.2, 0.15), return to home",

        "Shaft insertion: pick shaft-001 from (0.2, -0.8, 0.0), "
        "insert into gear hub at (0.4, 0.2, 0.15), secure with 10N grip",

        "Bearing press: pick bearing-001 from (-0.2, -0.8, 0.0), "
        "press onto shaft at assembly station, verify with force reading",

        "Dual-arm transfer: use arm1 to hold housing at (0.4, 0.0, 0.3), "
        "arm2 to place sub-assembly into housing, check collision risk",

        "Weld seam: after confirming housing is seated, weld joint 'housing-a' "
        "for 3 seconds, verify weld temp, report cell status",

        "Full cycle: start conveyor, pick next gear, insert shaft, "
        "press bearing, weld housing, increment cycle. Check sensors after each step.",
    ]

    logger.info("=== Robotics Assembly Cell — Training Loop ===")
    for i, task in enumerate(tasks, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i}/{len(tasks)}: {task[:80]}...")

        try:
            result = agent.run(task)

            logger.info(
                "  Result: success=%s  |  steps=%d  |  duration=%.2fs  |  tokens=%d+%d",
                result.success,
                len(result.steps),
                result.duration,
                result.total_tokens_in,
                result.total_tokens_out,
            )

            if result.error:
                logger.warning("  Error: %s", result.error)

        except Exception as e:
            logger.error("  Task failed with exception: %s", e)

    logger.info(f"\n{'='*60}")
    logger.info("Training complete. Agent stats:")
    stats = agent.stats()
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    skills = agent.learned_skills()
    if skills:
        logger.info(f"\nLearned skills ({len(skills)}):")
        for s in skills:
            logger.info("  - %s", s)

    logger.info(f"\nSuccess rate (last 10): {agent.success_rate(last_n=10):.0%}")
    logger.info("Store: ./robotics_store/nil.db")
    return agent


if __name__ == "__main__":
    agent = main()
