import json
import tempfile
import unittest
from pathlib import Path

from go2_foundation import (
    ACTION_DIM,
    OBSERVATION_DIM,
    CommandLimiter,
    CampusMissionPipeline,
    audit_policy_interface,
    command_from_mission_output,
    Go2ControlCycle,
    EpisodeMetrics,
    Go2ObservationConfig,
    IsaacLabStateAdapter,
    MetricsAccumulator,
    ObservationBuilder,
    PerformanceGate,
    PerformanceThresholds,
    PatrolMissionScheduler,
    PolicyRunner,
    RobotState,
    RouteProgressEvaluator,
    RouteSample,
    RouteWaypoint,
    SafetySupervisor,
    VelocityCommand,
    ZeroPolicy,
)


class FoundationTests(unittest.TestCase):
    def make_state(self):
        return RobotState(
            base_ang_vel=(1.0, 2.0, 3.0),
            projected_gravity=(0.0, 0.0, -1.0),
            dof_pos=(0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5),
            dof_vel=(1.0,) * ACTION_DIM,
            previous_action=(0.2,) * ACTION_DIM,
            command=VelocityCommand(0.5, -0.25, 0.4),
        )

    def test_observation_is_exact_45_dimensional_and_scaled(self):
        obs = ObservationBuilder().build(self.make_state())
        self.assertEqual(len(obs), OBSERVATION_DIM)
        self.assertEqual(obs[0:3], (0.25, 0.5, 0.75))
        self.assertEqual(obs[3:6], (0.0, 0.0, -1.0))
        self.assertEqual(obs[6:9], (1.0, -0.5, 0.1))
        self.assertEqual(obs[9:21], (0.0,) * ACTION_DIM)
        self.assertEqual(obs[21:33], (0.05,) * ACTION_DIM)
        self.assertEqual(obs[33:45], (0.2,) * ACTION_DIM)

    def test_policy_profiles_keep_command_scales_explicit(self):
        self.assertEqual(Go2ObservationConfig.mujoco().command_scale, (2.0, 2.0, 0.25))
        self.assertEqual(Go2ObservationConfig.real_deployment().command_scale, (3.0, 2.0, 0.5))
        rl_lab = Go2ObservationConfig.unitree_rl_lab_relative_joint()
        self.assertEqual(rl_lab.ang_vel_scale, 0.2)
        self.assertEqual(rl_lab.command_scale, (1.0, 1.0, 1.0))
        self.assertEqual(rl_lab.default_joint_angles, (0.0,) * ACTION_DIM)

    def test_policy_runner_validates_io(self):
        action = PolicyRunner(ZeroPolicy()).infer((0.0,) * OBSERVATION_DIM)
        self.assertEqual(action, (0.0,) * ACTION_DIM)

    def test_command_limiter(self):
        command = CommandLimiter().apply(VelocityCommand(9.0, -9.0, 9.0))
        self.assertEqual(command, VelocityCommand(0.8, -0.4, 1.2))

    def test_mission_bridge_accepts_existing_controller_names(self):
        command = command_from_mission_output(
            {"forward_mps": 0.25, "lateral_mps": 0.0, "yaw_rate_rps": -0.1}
        )
        self.assertEqual(command, VelocityCommand(0.25, 0.0, -0.1))

    def test_isaac_adapter_extracts_batched_env(self):
        raw = {
            "base_ang_vel": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            "projected_gravity": [[0.0, 0.0, -1.0], [0.0, 0.1, -0.99]],
            "dof_pos": [[0.0] * ACTION_DIM, [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]],
            "dof_vel": [[0.0] * ACTION_DIM, [0.2] * ACTION_DIM],
            "previous_action": [[0.0] * ACTION_DIM, [0.1] * ACTION_DIM],
            "commands": [[0.0, 0.0, 0.0], [0.25, 0.0, 0.1]],
        }
        state = IsaacLabStateAdapter(env_index=1).from_mapping(raw)
        self.assertEqual(state.command, VelocityCommand(0.25, 0.0, 0.1))
        self.assertEqual(state.base_ang_vel, (1.0, 2.0, 3.0))

    def test_safety_stops_on_fall_and_clips_action(self):
        supervisor = SafetySupervisor()
        decision = supervisor.evaluate((2.0,) * ACTION_DIM, observation_age_s=0.01, roll_rad=0.0, pitch_rad=0.0)
        self.assertTrue(decision.allow_motion)
        self.assertTrue(decision.clipped)
        self.assertEqual(decision.action, (1.0,) * ACTION_DIM)
        stopped = supervisor.evaluate((0.0,) * ACTION_DIM, observation_age_s=0.01, roll_rad=0.0, pitch_rad=0.0, fallen=True)
        self.assertFalse(stopped.allow_motion)
        self.assertEqual(stopped.reason, "fall_detected")

    def test_control_cycle_composes_all_layers(self):
        output = Go2ControlCycle(PolicyRunner(ZeroPolicy())).step(
            self.make_state(), observation_age_s=0.01, roll_rad=0.0, pitch_rad=0.0
        )
        self.assertEqual(len(output.observation), OBSERVATION_DIM)
        self.assertEqual(output.policy_action, (0.0,) * ACTION_DIM)
        self.assertTrue(output.safety.allow_motion)

    def test_metrics_summary(self):
        metrics = MetricsAccumulator()
        metrics.add(EpisodeMetrics("a", True, completion_time_s=10.0, falls=0))
        metrics.add(EpisodeMetrics("b", False, completion_time_s=20.0, falls=1))
        summary = metrics.summary()
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["total_falls"], 1)

    def test_route_evaluator_keeps_preview_and_rl_evidence_distinct(self):
        evaluator = RouteProgressEvaluator(
            (RouteWaypoint("P0", 0.0, 0.0), RouteWaypoint("P1", 1.0, 0.0)),
            reach_radius_m=0.1,
        )
        evaluator.update(RouteSample(0.0, 0.0, 0.0))
        evaluator.update(RouteSample(1.0, 1.0, 0.0, roll_rad=0.1))
        preview = evaluator.evaluate("kinematic_preview")
        self.assertTrue(preview.route_complete)
        self.assertFalse(preview.claimable_as_rl_result)
        self.assertIn("运动学预览", preview.notes[0])
        physics = evaluator.evaluate("rl_physics")
        self.assertTrue(physics.claimable_as_rl_result)
        self.assertEqual(physics.waypoint_reached, 2)

    def test_performance_gate_requires_rl_evidence_and_multiple_seeds(self):
        thresholds = PerformanceThresholds(minimum_episodes_per_seed=2, minimum_distinct_seeds=3)
        good = [
            EpisodeMetrics("a", True, velocity_error_mps=0.1, max_tilt_rad=0.2),
            EpisodeMetrics("b", True, velocity_error_mps=0.1, max_tilt_rad=0.2),
        ]
        gate = PerformanceGate(thresholds)
        preview = gate.assess({1: good, 2: good, 3: good}, "kinematic_preview")
        self.assertFalse(preview.promoted)
        physics = gate.assess({1: good, 2: good, 3: good}, "rl_physics")
        self.assertTrue(physics.promoted)
        missing_seed = gate.assess({1: good, 2: good}, "rl_physics")
        self.assertFalse(missing_seed.promoted)

    def test_campus_override_preserves_policy_interface(self):
        source = Path("campus_rl/campus_go2_velocity.py").read_text(encoding="utf-8")
        compile(source, "campus_go2_velocity.py", "exec")
        self.assertIn("class CampusGo2VelocityEnvCfg(RobotEnvCfg)", source)
        self.assertIn("command.resampling_time_range = (2.0, 6.0)", source)
        self.assertNotIn("self.actions.", source)

    def test_patrol_scheduler_inserts_dwell_and_safe_completion(self):
        class Command:
            def __init__(self, reached=False, complete=False, target_index=2):
                self.forward_mps = 0.3
                self.lateral_mps = 0.0
                self.yaw_rate_rps = 0.1
                self.target_index = target_index
                self.reached = reached
                self.mission_complete = complete

        scheduler = PatrolMissionScheduler({1: 2.0})
        moving = scheduler.update(Command(), 0.0)
        self.assertEqual(moving.phase, "moving")
        arrived = scheduler.update(Command(reached=True), 1.0)
        self.assertEqual(arrived.phase, "inspecting")
        self.assertEqual(arrived.command, VelocityCommand())
        self.assertEqual(scheduler.update(Command(), 2.0).phase, "inspecting")
        self.assertEqual(scheduler.update(Command(), 3.1).phase, "moving")
        complete = scheduler.update(Command(complete=True, target_index=8), 4.0)
        self.assertTrue(complete.mission_complete)
        self.assertEqual(complete.command, VelocityCommand())

    def test_patrol_scheduler_honors_final_waypoint_dwell_before_completion(self):
        class Command:
            forward_mps = 0.0
            lateral_mps = 0.0
            yaw_rate_rps = 0.0
            target_index = 2
            reached = True
            mission_complete = True

        scheduler = PatrolMissionScheduler({1: 2.0})
        arrival = scheduler.update(Command(), 0.0)
        self.assertEqual(arrival.phase, "inspecting")
        self.assertFalse(arrival.mission_complete)
        self.assertEqual(scheduler.update(Command(), 1.0).phase, "inspecting")
        completion = scheduler.update(Command(), 2.0)
        self.assertEqual(completion.phase, "complete")
        self.assertTrue(completion.mission_complete)

    def test_campus_pipeline_limits_controller_command(self):
        class Command:
            forward_mps = 5.0
            lateral_mps = -5.0
            yaw_rate_rps = 5.0
            target_index = 1
            reached = False
            mission_complete = False

        class Controller:
            def update(self, x, y, yaw):
                return Command()

        pipeline = CampusMissionPipeline(Controller(), PatrolMissionScheduler({}), CommandLimiter())
        result = pipeline.step(time_s=0.0, x=0.0, y=0.0, yaw=0.0)
        self.assertEqual(result.phase, "moving")
        self.assertEqual(result.command, VelocityCommand(0.8, -0.4, 1.2))
        pipeline.reset()

    def test_experiment_json_assessment_rejects_unknown_fields(self):
        from go2_foundation import assess_file

        payload = {
            "evidence_mode": "rl_physics",
            "results_by_seed": {"1": [{"episode_id": "x", "success": True, "unknown": 1}]},
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                assess_file(path)
        finally:
            Path(path).unlink()

    def test_module_cli_smoke_and_profiles(self):
        from go2_foundation.__main__ import _profiles, _smoke

        self.assertEqual(_smoke()["observation_dim"], OBSERVATION_DIM)
        self.assertIn("unitree_rl_lab_relative_joint", _profiles())

    def test_policy_interface_audit_fails_closed_on_profile_mismatch(self):
        valid = {
            "policy_name": "go2-policy",
            "profile": "unitree_rl_lab_relative_joint",
            "observation_dim": 45,
            "action_dim": 12,
            "action_scale": 0.25,
            "control_dt_s": 0.02,
            "command_scale": [1.0, 1.0, 1.0],
            "ang_vel_scale": 0.2,
        }
        self.assertTrue(audit_policy_interface(valid).compatible)
        invalid = dict(valid, command_scale=[2.0, 2.0, 0.25])
        result = audit_policy_interface(invalid)
        self.assertFalse(result.compatible)
        self.assertIn("command_scale", result.reasons[0])

    def test_policy_manifest_binds_model_hash_and_interface(self):
        from go2_foundation import sha256_file, verify_policy_manifest

        metadata = {
            "policy_name": "go2-policy",
            "profile": "unitree_rl_lab_relative_joint",
            "observation_dim": 45,
            "action_dim": 12,
            "action_scale": 0.25,
            "control_dt_s": 0.02,
            "command_scale": [1.0, 1.0, 1.0],
            "ang_vel_scale": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "policy.pt"
            model.write_bytes(b"test policy artifact")
            manifest = root / "policy_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_path": model.name,
                        "model_format": "torchscript",
                        "sha256": sha256_file(model),
                        "policy_interface": metadata,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(verify_policy_manifest(manifest).verified)
            model.write_bytes(b"different artifact")
            failed = verify_policy_manifest(manifest)
            self.assertFalse(failed.verified)
            self.assertIn("model_sha256_mismatch", failed.reasons)

    def test_manifest_creator_infers_only_supported_export_formats(self):
        from scripts.create_policy_manifest import infer_format, main
        from unittest.mock import patch

        self.assertEqual(infer_format(Path("policy.pt")), "torchscript")
        self.assertEqual(infer_format(Path("policy.onnx")), "onnx")
        with self.assertRaises(ValueError):
            infer_format(Path("policy.pth"))
        metadata = {
            "policy_name": "go2-policy",
            "profile": "unitree_rl_lab_relative_joint",
            "observation_dim": 45,
            "action_dim": 12,
            "action_scale": 0.25,
            "control_dt_s": 0.02,
            "command_scale": [1.0, 1.0, 1.0],
            "ang_vel_scale": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "policy.pt"
            metadata_path = root / "metadata.json"
            model.write_bytes(b"policy")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with patch("sys.argv", ["create_policy_manifest", str(model), str(metadata_path)]):
                self.assertEqual(main(), 0)
            self.assertTrue((root / "policy_manifest.json").is_file())

    def test_training_launcher_distinguishes_baseline_and_campus(self):
        from go2_foundation import TrainingRunSpec, build_train_command, verify_unitree_rl_lab_checkout

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "scripts" / "rsl_rl" / "train.py"
            task = root / "source" / "unitree_rl_lab" / "unitree_rl_lab" / "tasks" / "locomotion" / "robots" / "go2" / "__init__.py"
            train.parent.mkdir(parents=True)
            task.parent.mkdir(parents=True)
            train.write_text("# upstream train", encoding="utf-8")
            task.write_text("# upstream Go2 task", encoding="utf-8")
            self.assertTrue(verify_unitree_rl_lab_checkout(root)["ready"])
            baseline = build_train_command(root, TrainingRunSpec("baseline", seed=1, max_iterations=10))
            campus = build_train_command(root, TrainingRunSpec("campus", seed=1, max_iterations=10))
            self.assertIn("Unitree-Go2-Velocity", baseline)
            self.assertNotIn("campus_rl_overrides", " ".join(baseline))
            self.assertIn("Unitree-Go2-Campus-Velocity", campus)
            self.assertIn("import campus_rl", campus[2])

    def test_route_velocity_controller_can_be_reused_for_delivery_or_patrol(self):
        from go2_foundation import MissionWaypoint, RouteVelocityController, load_route_file

        route = (
            MissionWaypoint("start", 0.0, 0.0),
            MissionWaypoint("delivery_point", 2.0, 0.0, dwell_s=15.0, task="delivery_handoff"),
        )
        controller = RouteVelocityController(route, max_speed_mps=0.8)
        moving = controller.update(0.0, 0.0, 0.0)
        self.assertGreater(moving.forward_mps, 0.0)
        self.assertFalse(moving.mission_complete)
        arrived = controller.update(2.0, 0.0, 0.0)
        self.assertTrue(arrived.reached)
        self.assertTrue(arrived.mission_complete)
        campus_route = load_route_file("examples/campus_patrol_route.json")
        self.assertEqual(campus_route[0].name, "P0_主入口起点")
        self.assertEqual(len(campus_route), 8)


if __name__ == "__main__":
    unittest.main()
