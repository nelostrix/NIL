#!/usr/bin/env python3
"""Example 07: NIL Embodied AI Agent with Autonomous Motor Skill Evolution.

Demonstrates:
- Embodied multimodal sensory observation (Joint telemetry + IMU + Contact)
- 3D semantic physical world modeling
- Spatial grasp affordance evaluation
- Continuous Dynamic Movement Primitive (DMP) imitation and spatial adaptation
- Real-time physical safety gatekeeping
- Self-evolving impedance and compliance refinement
"""

import numpy as np
from nil import (
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


def main():
    print("=" * 65)
    print("NIL EMBODIED AI: Autonomous Physical Skill Evolution Kernel")
    print("=" * 65)

    # 1. Initialize 3D Semantic World Model
    world = PhysicalWorldModel()
    gear = PhysicalObject(
        name="bronze_gear_01",
        category="gear",
        position=(0.45, 0.12, 0.04),
        dimensions=(0.05, 0.05, 0.03),
        mass_kg=0.42,
        friction_coeff=0.65,
        is_fragile=False
    )
    world.upsert_object(gear)
    print(f"Perceived Object: '{gear.name}' (Mass: {gear.mass_kg}kg, Pos: {gear.position})")

    # 2. Evaluate Spatial Grasp Affordance
    aff = AffordanceQuery.evaluate_grasp_affordance(gear, max_gripper_stroke_m=0.085)
    print(f"Grasp Affordance: Feasible={aff['feasible']}, Score={aff['score']}, Force={aff['recommended_force_n']:.1f}N")

    # 3. Dynamic Movement Primitive (DMP) Motor Skill
    dmp = DynamicMovementPrimitive(num_dof=3, num_basis=20, K=150.0)
    # Demonstrated trajectory (e.g. human teleoperation reach)
    t = np.linspace(0, 1, 50)
    demo_reach = np.column_stack([
        0.2 + 0.25 * t,
        0.0 + 0.12 * t,
        0.2 - 0.16 * np.sin(t * np.pi / 2.0)
    ])
    dmp.fit_demonstration(demo_reach, tau=1.0)

    # Generalize motor skill to target gear position
    target_pos = np.array(gear.position)
    start_pos = np.array([0.15, 0.0, 0.25])
    adapted_traj = dmp.generate_rollout(x0=start_pos, g=target_pos, tau=1.0)
    print(f"Generated Adapted DMP Trajectory ({len(adapted_traj)} waypoints) to {target_pos}")

    # 4. Physical Safety Gatekeeper
    safety = PhysicalSafetyGatekeeper(
        joint_limits_rad=[(-2.8, 2.8)] * 6,
        max_joint_velocities_rad_s=2.0
    )
    q_curr = np.array([0.0, 0.2, -0.4, 0.0, 0.1, 0.0])
    q_unsafe = np.array([3.5, 0.2, -0.4, 0.0, 0.1, 0.0])  # Exceeds limit
    q_safe, modified, reason = safety.sanitize_joint_commands(q_curr, q_unsafe)
    print(f"Safety Supervisor: Target joint commanded -> {reason} (Safe Q1 = {q_safe[0]:.2f} rad)")

    # 5. Continuous Skill Memory & Impedance Refinement
    skill_store = EmbodiedSkillMemory()
    insertion_skill = EmbodiedSkill(
        name="gearbox_insertion",
        category="assembly",
        impedance_stiffness=[250.0, 250.0, 100.0],
        confidence=0.82
    )
    skill_store.register_skill(insertion_skill)

    # Simulate physical execution outcome (soft contact success)
    insertion_skill.record_execution(success=True)
    print(f"Evolved Skill: '{insertion_skill.name}' Confidence: {insertion_skill.confidence * 100:.1f}%, Stiffness: {insertion_skill.stiffness}")

    print("\n🎉 NIL Embodied Robotics AI Agent verified operational!")


if __name__ == "__main__":
    main()
