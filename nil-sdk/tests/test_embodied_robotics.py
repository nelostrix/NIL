"""Comprehensive validation suite for NIL Embodied Robotics AI Kernel."""

import math
import numpy as np
import pytest

from nil.robotics import (
    EmbodiedSensorium,
    SensorFrame,
    PhysicalWorldModel,
    PhysicalObject,
    AffordanceQuery,
    DynamicMovementPrimitive,
    EmbodiedSkillMemory,
    EmbodiedSkill,
    PhysicalSafetyGatekeeper,
    AutonomousManipulationDomain,
)


def test_embodied_sensorium():
    """Verify multimodal sensorium sliding-window buffer."""
    sensorium = EmbodiedSensorium(buffer_size=5)

    for i in range(10):
        frame = SensorFrame(
            joint_positions=[0.1 * i, -0.2 * i],
            joint_velocities=[0.01, -0.01],
            contact_states=[i > 7]
        )
        sensorium.ingest(frame)

    assert len(sensorium.frames) == 5
    latest = sensorium.get_latest()
    assert latest is not None
    assert latest.joint_positions[0] == pytest.approx(0.9)
    assert sensorium.has_contact()


def test_physical_world_model():
    """Verify 3D object-centric semantic memory and spatial queries."""
    world = PhysicalWorldModel()
    gear = PhysicalObject(
        name="bronze_gear_01",
        category="gear",
        position=(0.45, 0.10, 0.05),
        dimensions=(0.04, 0.04, 0.02),
        mass_kg=0.35,
        friction_coeff=0.6,
        is_fragile=False
    )
    bracket = PhysicalObject(
        name="bracket_01",
        category="bracket",
        position=(0.80, -0.20, 0.05),
        dimensions=(0.10, 0.08, 0.04),
        mass_kg=1.2
    )

    world.upsert_object(gear)
    world.upsert_object(bracket)

    assert len(world.get_all_objects()) == 2
    nearest = world.find_nearest_object(query_pos=(0.40, 0.05, 0.0), category="gear")
    assert nearest is not None
    assert nearest.name == "bronze_gear_01"


def test_spatial_affordance_engine():
    """Verify grasp and traversability affordance evaluations."""
    small_obj = PhysicalObject(name="screw", dimensions=(0.01, 0.01, 0.02), mass_kg=0.02)
    aff_small = AffordanceQuery.evaluate_grasp_affordance(small_obj, max_gripper_stroke_m=0.085)
    assert aff_small["feasible"] is True
    assert aff_small["score"] > 0.8

    heavy_obj = PhysicalObject(name="anvil", dimensions=(0.25, 0.25, 0.30), mass_kg=45.0)
    aff_heavy = AffordanceQuery.evaluate_grasp_affordance(heavy_obj, max_gripper_stroke_m=0.085)
    assert aff_heavy["feasible"] is False


def test_dynamic_movement_primitive():
    """Verify DMP trajectory imitation fitting and generalization to new target goals."""
    dmp = DynamicMovementPrimitive(num_dof=2, num_basis=15, K=120.0)

    # Synthetic demonstrated sine trajectory (from [0,0] to [1,1])
    t = np.linspace(0, 1, 50)
    demo_traj = np.column_stack([t, np.sin(t * np.pi / 2.0)])

    dmp.fit_demonstration(demo_traj)

    # Rollout to a new goal [2.0, 1.5] from start [0.0, 0.0]
    rollout = dmp.generate_rollout(x0=np.array([0.0, 0.0]), g=np.array([2.0, 1.5]), tau=1.0)
    assert len(rollout) > 0
    assert np.allclose(rollout[0], [0.0, 0.0], atol=1e-2)
    assert np.allclose(rollout[-1], [2.0, 1.5], atol=0.1)


def test_embodied_skill_memory_and_evolution():
    """Verify skill registration and automatic compliance tuning."""
    mem = EmbodiedSkillMemory()
    skill = EmbodiedSkill(name="peg_insertion", category="assembly", impedance_stiffness=[300.0, 300.0, 300.0])
    mem.register_skill(skill)

    # Record 1 failure (impact over-torque) -> should soften stiffness
    initial_stiffness = skill.stiffness[0]
    skill.record_execution(success=False)
    assert skill.stiffness[0] < initial_stiffness
    assert skill.confidence < 0.80

    # Record 3 successes -> should increase confidence
    for _ in range(3):
        skill.record_execution(success=True)
    assert skill.confidence > 0.80


def test_safety_gatekeeper():
    """Verify joint angle clamping and velocity rate-limiting."""
    safety = PhysicalSafetyGatekeeper(
        joint_limits_rad=[(-1.57, 1.57), (-2.0, 2.0)],
        max_joint_velocities_rad_s=1.0
    )

    # Test position clamp exceeding limit
    q_safe, modified, reason = safety.sanitize_joint_commands(
        q_current=np.array([0.0, 0.0]),
        q_target=np.array([2.5, -1.0]),
        dt=0.1
    )
    assert q_safe[0] <= 1.57
    assert modified is True

    # Test rate limiter
    q_safe_rate, mod_rate, _ = safety.sanitize_joint_commands(
        q_current=np.array([0.0, 0.0]),
        q_target=np.array([0.5, 0.0]),
        dt=0.01  # dt * max_vel = 0.01 rad max step
    )
    assert q_safe_rate[0] <= 0.02
    assert mod_rate is True


def test_autonomous_manipulation_domain():
    """Verify embodied manipulation domain tools."""
    dom = AutonomousManipulationDomain(name="test_manip_domain")
    assert len(dom.tools) == 3
    assert len(dom.objectives) >= 2
