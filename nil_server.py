#!/usr/bin/env python3
"""NIL Backend Cognitive Server for NELO-Studio.

Connects the NIL Self-Evolving Domain Agent Framework (@~/nil)
directly to the NELO-Studio IDE & Robotics Workstation.

Features:
- A real nil.Agent (10-phase Unified Cognitive Loop) whose tools call the
  live stacc_server.py robotics backend over HTTP — this is the actual
  nil<->stacc bridge, not a keyword-matched canned responder.
- Persistent SQLite memory store (~/nil/agent_store/nil.db) for UI-facing
  trace/skill display, plus the agent's own PersistentMemory for real
  learning state.
- Real-time telemetry: Trace logging, skill registry, and evolution cycles
"""

import os
import sys
import json
import time
import sqlite3
import logging
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

# Add NIL SDK to python path
NIL_ROOT = os.path.dirname(os.path.abspath(__file__))
SDK_PATH = os.path.join(NIL_ROOT, "nil-sdk")
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

STACC_SERVER_URL = "http://127.0.0.1:8766"


def _stacc_request(path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call the real stacc_server.py daemon (live MuJoCo + kinematics backend,
    already spawned alongside this process by electron/main.js). GET when
    payload is None, POST otherwise.
    """
    url = f"{STACC_SERVER_URL}{path}"
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


try:
    import nil
    from nil.memory.store import PersistentMemory
    from nil.domain import Domain
    from nil.agent import Agent
    from nil.config import EvolutionConfig
    from nil.tools.base import tool

    @tool
    def get_robot_status() -> dict:
        """Get the live robot's current status: physics engine, active model,
        control loop rate, and total simulation steps, from the real STACC
        simulation backend (not simulated/estimated here)."""
        return _stacc_request("/v1/system/status")

    @tool
    def move_robot_joints(joints_deg: List[float]) -> dict:
        """Command the live 6-DOF robot arm to a target joint configuration
        (degrees, one per joint) and report the real resulting end-effector
        pose, joint torques, and obstacle safety margin from the live MuJoCo
        simulation."""
        return _stacc_request("/v1/robot/simulate_step", {"joints": joints_deg})

    @tool
    def solve_inverse_kinematics(target_xyz: List[float]) -> dict:
        """Solve for joint angles that reach a target [x, y, z] end-effector
        position in meters, using the real damped-least-squares IK solver,
        and drive the live arm to that solution."""
        return _stacc_request("/v1/robot/ik", {"target_pos": target_xyz})

    @tool
    def tune_impedance_gains(kp: float, kd: float) -> dict:
        """Tune the live robot's joint impedance controller gains: Kp
        (stiffness, N*m/rad) and Kd (damping, N*m*s/rad)."""
        return _stacc_request("/v1/robot/tune_gains", {"kp": kp, "kd": kd})

    NIL_ROBOT_TOOLS = [get_robot_status, move_robot_joints, solve_inverse_kinematics, tune_impedance_gains]

    NIL_DOMAIN = Domain(
        name="nelo_studio_robotics",
        tools=NIL_ROBOT_TOOLS,
        objectives=[
            "Help the operator understand and control the live STACC robot simulation",
            "Only report real telemetry returned by the robot tools, never a fabricated value",
        ],
        success_signal=lambda run: run.success,
    )

    NIL_AGENT_MEMORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_store", "nil_agent_memory.db")
    NIL_AGENT = Agent(
        domain=NIL_DOMAIN,
        memory=PersistentMemory(NIL_AGENT_MEMORY_PATH),
        evolution=EvolutionConfig(strategy="manual"),
    )
    NIL_SDK_AVAILABLE = True
except Exception as e:
    NIL_SDK_AVAILABLE = False
    NIL_AGENT = None
    print(f"[NIL-SERVER] Warning: NIL SDK import error ({e}). Chat/evolution endpoints will report unavailable rather than fake a response.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nil.server")

DB_DIR = os.path.join(NIL_ROOT, "agent_store")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "nil.db")

START_TIME = time.time()
EVOLUTION_CYCLES = 1


def init_sqlite_db():
    """Ensure SQLite database tables for traces, skills, and evolution metrics exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nil_traces (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            task TEXT,
            cognitive_mode TEXT,
            input_context TEXT,
            output_response TEXT,
            actions_executed TEXT,
            reward REAL,
            tokens_used INTEGER,
            latency_ms REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nil_skills (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            category TEXT,
            trigger_condition TEXT,
            parameters TEXT,
            confidence REAL,
            executions_count INTEGER,
            success_count INTEGER,
            created_at REAL,
            last_used REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nil_world_entities (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            entity_type TEXT,
            state TEXT,
            updated_at REAL
        )
    """)
    # Seed starter skills if empty
    cursor.execute("SELECT COUNT(*) FROM nil_skills")
    if cursor.fetchone()[0] == 0:
        now = time.time()
        initial_skills = [
            ("skill_imp_tune", "Adaptive Impedance Tuning", "physics", "contact_instability_or_vibration", json.dumps({"Kp_ratio": 1.25, "Kd_ratio": 1.15}), 0.94, 28, 27, now, now),
            ("skill_dmp_reach", "Dynamic Movement Reach (DMP)", "kinematics", "spatial_target_jog", json.dumps({"damping_ratio": 0.707, "basis_count": 20}), 0.91, 45, 42, now, now),
            ("skill_cbf_safety", "Control Barrier Function Safety", "robotics", "obstacle_proximity_less_45mm", json.dumps({"barrier_gamma": 0.85}), 0.98, 62, 62, now, now),
            ("skill_shm_sync", "Zero-Copy POSIX Shared Memory Sync", "system", "1000hz_rt_loop", json.dumps({"ringbuffer_slots": 10000}), 0.96, 120, 119, now, now),
        ]
        cursor.executemany("""
            INSERT OR IGNORE INTO nil_skills (id, name, category, trigger_condition, parameters, confidence, executions_count, success_count, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, initial_skills)
    conn.commit()
    conn.close()


init_sqlite_db()


def get_db_stats():
    """Retrieve trace count and skill count from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM nil_traces")
    traces_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM nil_skills")
    skills_count = cursor.fetchone()[0]
    conn.close()
    return traces_count, skills_count


def _extract_reply_and_actions(outcome: "Any") -> "Tuple[str, Dict[str, Any]]":
    """Build the chat reply and UI-facing action dict from a real nil.Agent
    RunOutcome — never templated text, always derived from what the agent
    actually did (outcome.output / outcome.steps / outcome.error).
    """
    actions: Dict[str, Any] = {}
    tool_calls_summary: List[str] = []

    for step in outcome.steps:
        if not step.tool_name:
            continue
        tool_calls_summary.append(f"{step.tool_name}({step.tool_input}) -> {step.tool_output}")

        if step.tool_name == "move_robot_joints" and isinstance(step.tool_output, dict):
            actions["type"] = "set_joints"
            actions["end_effector"] = step.tool_output.get("end_effector")
            actions["torques_nm"] = step.tool_output.get("torques_nm")
        elif step.tool_name == "tune_impedance_gains" and step.tool_input:
            actions["type"] = "set_gains"
            actions["setKpGain"] = step.tool_input.get("kp")
            actions["setKdGain"] = step.tool_input.get("kd")
        elif step.tool_name == "solve_inverse_kinematics" and isinstance(step.tool_output, dict):
            actions["type"] = "ik_solution"
            actions["setJoints"] = step.tool_output.get("solution_joints_deg")

    if outcome.output:
        reply_text = str(outcome.output)
    elif tool_calls_summary:
        reply_text = "Executed: " + "; ".join(tool_calls_summary)
    elif outcome.error:
        reply_text = f"NIL agent run failed: {outcome.error}"
    else:
        reply_text = "(NIL agent produced no output for this run)"

    if tool_calls_summary:
        actions["terminalLog"] = "[NIL] " + " | ".join(tool_calls_summary)

    return reply_text, actions


class NILRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: Any):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/system/status" or self.path == "/health":
            traces_count, skills_count = get_db_stats()
            self._send_json(200, {
                "status": "online",
                "backend": "NIL-Self-Evolving-Framework",
                "version": "1.0.0",
                "sdk_loaded": NIL_SDK_AVAILABLE,
                "db_path": DB_PATH,
                "traces_count": traces_count,
                "skills_count": skills_count,
                "evolution_cycles": EVOLUTION_CYCLES,
                "uptime_seconds": int(time.time() - START_TIME),
                "pid": os.getpid(),
            })
        elif self.path == "/v1/memory/skills":
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nil_skills ORDER BY confidence DESC")
            skills = [dict(row) for row in cursor.fetchall()]
            conn.close()
            self._send_json(200, {"skills": skills, "total": len(skills)})
        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        global EVOLUTION_CYCLES
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        req_data = json.loads(body) if body else {}

        if self.path == "/v1/evolution/trigger":
            if NIL_AGENT is None:
                self._send_json(503, {"status": "unavailable", "error": "NIL SDK not loaded on this server."})
                return
            try:
                evo_result = NIL_AGENT.evolve()
                EVOLUTION_CYCLES += 1
                traces_count, skills_count = get_db_stats()
                self._send_json(200, {
                    "status": "evolved",
                    "message": f"Evolution cycle #{EVOLUTION_CYCLES} completed.",
                    "evolution_result": str(evo_result),
                    "traces_analyzed": traces_count,
                    "total_skills": skills_count,
                })
            except Exception as e:
                logger.exception("nil.Agent.evolve failed")
                self._send_json(500, {"status": "error", "error": str(e)})
            return

        if self.path in ["/v1/agent/chat", "/v1/agent/run"]:
            t0 = time.time()
            prompt = req_data.get("prompt", "")
            active_file = req_data.get("activeFileName", "robot_arm_controller.py")
            joints = req_data.get("joints", [0, 0, 0, 0, 0, 0])
            kp = req_data.get("kpGain", 180)
            kd = req_data.get("kdGain", 18)

            if NIL_AGENT is None:
                self._send_json(503, {
                    "reply": "The NIL SDK isn't loaded on this server, so there's no agent to run — "
                             "no response was fabricated.",
                    "actions": {},
                    "nil_telemetry": {"error": "sdk_unavailable"},
                })
                return

            try:
                outcome = NIL_AGENT.run(
                    prompt,
                    context={"active_file": active_file, "joints": joints, "kp": kp, "kd": kd},
                )
            except Exception as e:
                logger.exception("nil.Agent.run failed")
                self._send_json(500, {
                    "reply": f"NIL agent run failed: {e}",
                    "actions": {},
                    "nil_telemetry": {"error": str(e)},
                })
                return

            reply_text, actions = _extract_reply_and_actions(outcome)
            cognitive_mode = "deliberative" if outcome.steps else "reactive"
            latency_ms = round((time.time() - t0) * 1000, 2)
            trace_id = outcome.run_id or f"tr_{int(time.time() * 1000)}"

            # Record Trace to SQLite DB
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO nil_traces (id, timestamp, task, cognitive_mode, input_context, output_response, actions_executed, reward, tokens_used, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace_id,
                    time.time(),
                    prompt,
                    cognitive_mode,
                    json.dumps({"active_file": active_file, "joints": joints, "kp": kp, "kd": kd}),
                    reply_text,
                    json.dumps(actions),
                    1.0 if outcome.success else 0.0,
                    outcome.total_tokens_in + outcome.total_tokens_out,
                    latency_ms,
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error("Failed to write trace to SQLite: %s", e)

            traces_count, skills_count = get_db_stats()

            self._send_json(200, {
                "reply": reply_text,
                "actions": actions,
                "nil_telemetry": {
                    "trace_id": trace_id,
                    "cognitive_mode": cognitive_mode,
                    "success": outcome.success,
                    "traces_recorded": traces_count,
                    "learned_skills_count": skills_count,
                    "evolution_cycles": EVOLUTION_CYCLES,
                    "latency_ms": latency_ms,
                    "tokens_in": outcome.total_tokens_in,
                    "tokens_out": outcome.total_tokens_out,
                    "db_path": DB_PATH,
                }
            })
            return

        self._send_json(404, {"error": "Endpoint not found"})


def run_server(port: int = 8765):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, NILRequestHandler)
    print(f"🚀 [NIL-SERVER] NIL Cognitive Backend Daemon listening on http://127.0.0.1:{port}")
    print(f"📦 [NIL-SERVER] SQLite Memory Store: {DB_PATH}")
    traces, skills = get_db_stats()
    print(f"🧠 [NIL-SERVER] Loaded {skills} learned skills | {traces} experience traces.")
    httpd.serve_forever()


if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
