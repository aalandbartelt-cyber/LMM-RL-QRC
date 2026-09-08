import tempfile
import unittest
from pathlib import Path

from go2_foundation.training_launcher import (
    CAMPUS_TASK,
    TrainingRunSpec,
    build_train_command,
    verify_unitree_rl_lab_checkout,
)


class TrainingIntegrationTests(unittest.TestCase):
    def make_checkout(self, root: Path, commit: str = "1330cb0d4236c4abd6bf1efec05a4fce7ad79678") -> Path:
        train = root / "scripts" / "rsl_rl" / "train.py"
        task = (
            root
            / "source"
            / "unitree_rl_lab"
            / "unitree_rl_lab"
            / "tasks"
            / "locomotion"
            / "robots"
            / "go2"
            / "__init__.py"
        )
        train.parent.mkdir(parents=True)
        task.parent.mkdir(parents=True)
        train.write_text("# upstream train", encoding="utf-8")
        task.write_text("# upstream Go2 task", encoding="utf-8")
        (root / ".qrc-upstream-commit").write_text(commit + "\n", encoding="utf-8")
        return root

    def test_campus_task_is_discoverable_by_unitree_filter(self):
        self.assertEqual(CAMPUS_TASK, "Unitree-Go2-Campus-Velocity")
        self.assertIn("Unitree", CAMPUS_TASK)
        self.assertNotIn("Isaac", CAMPUS_TASK)

    def test_checkout_rejects_wrong_pinned_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_checkout(Path(directory), commit="wrong")
            report = verify_unitree_rl_lab_checkout(root)
            self.assertFalse(report["ready"])
            self.assertIn("commit_mismatch", report["problems"])

    def test_campus_resume_command_registers_task_and_forwards_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_checkout(Path(directory))
            spec = TrainingRunSpec(
                "campus",
                seed=7,
                max_iterations=1500,
                num_envs=4096,
                device="cuda:0",
                run_name="campus_s1",
                resume=True,
                load_run="baseline_run",
                checkpoint="model_5000.pt",
            )
            command = build_train_command(root, spec, python_executable="python")
            joined = " ".join(command)
            self.assertIn("import campus_rl", command[2])
            self.assertIn("sys.path.insert", command[2])
            self.assertIn(repr(str((root / "scripts" / "rsl_rl").resolve())), command[2])
            self.assertIn("--task Unitree-Go2-Campus-Velocity", joined)
            self.assertIn("--resume", command)
            self.assertIn("--load_run baseline_run", joined)
            self.assertIn("--checkpoint model_5000.pt", joined)
            self.assertIn("--run_name campus_s1", joined)

    def test_headless_training_rejects_video_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_checkout(Path(directory))
            spec = TrainingRunSpec("campus", seed=1, max_iterations=10)
            with self.assertRaisesRegex(ValueError, "video"):
                build_train_command(root, spec, extra_args=("--video",))

    def test_campus_config_declares_all_required_terrain_families(self):
        source = Path("campus_rl/campus_go2_velocity.py").read_text(encoding="utf-8")
        compile(source, "campus_go2_velocity.py", "exec")
        for name in (
            '"flat"',
            '"random_rough"',
            '"pyramid_slope"',
            '"pyramid_slope_inv"',
            '"boxes"',
            '"pyramid_stairs"',
            '"pyramid_stairs_inv"',
        ):
            self.assertIn(name, source)
        self.assertIn('id="Unitree-Go2-Campus-Velocity"', source)

    def test_route_evaluator_is_policy_driven_without_root_teleportation(self):
        root = Path(__file__).parents[2]
        config_source = Path("campus_rl/campus_route_eval.py").read_text(encoding="utf-8")
        evaluator_source = (
            root / "simulation" / "scenes" / "campus_security" / "evaluate_physical_route.py"
        ).read_text(encoding="utf-8")
        compile(config_source, "campus_route_eval.py", "exec")
        compile(evaluator_source, "evaluate_physical_route.py", "exec")
        self.assertIn('id="Unitree-Go2-Campus-Route-Eval"', config_source)
        for building in ("office", "warehouse", "power_room", "gate_house"):
            self.assertIn(building, config_source)
        self.assertIn("torch.jit.load", evaluator_source)
        self.assertIn("vel_command_b", evaluator_source)
        self.assertIn("gym.wrappers.RecordVideo", evaluator_source)
        self.assertIn("args_cli.enable_cameras = True", evaluator_source)
        self.assertNotIn("write_root_pose_to_sim", evaluator_source)


if __name__ == "__main__":
    unittest.main()
