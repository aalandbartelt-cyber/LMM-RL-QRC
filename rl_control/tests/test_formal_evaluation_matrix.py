import importlib.util
import json
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
AGGREGATOR = REPOSITORY_ROOT / "evaluation" / "scripts" / "aggregate_formal_matrix.py"
LAUNCHER = (
    REPOSITORY_ROOT
    / "rl_control"
    / "legacy_isaacgym"
    / "run_formal_evaluation_matrix.sh"
)
STATUS_SCRIPT = (
    REPOSITORY_ROOT
    / "rl_control"
    / "legacy_isaacgym"
    / "show_formal_evaluation_status.sh"
)
GITATTRIBUTES = REPOSITORY_ROOT / ".gitattributes"
LAUNCHER_RELATIVE = LAUNCHER.relative_to(REPOSITORY_ROOT).as_posix()
STATUS_RELATIVE = STATUS_SCRIPT.relative_to(REPOSITORY_ROOT).as_posix()


def load_aggregator():
    spec = importlib.util.spec_from_file_location("aggregate_formal_matrix", AGGREGATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_bash(script: str, *, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
        exports = "; ".join(
            f"export {name}={shlex.quote(value)}" for name, value in env.items()
        )
        script = f"{exports}; {script}"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env=merged_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def source_launcher(command: str, *, env=None):
    script = (
        f'export QRC_SOURCE_ONLY=1; source "{LAUNCHER_RELATIVE}"; '
        f"{command}"
    )
    return run_bash(script, env=env)


class FormalMatrixAggregationTests(unittest.TestCase):
    def write_complete_matrix(self, root: Path, module) -> None:
        policy_values = {
            "baseline5001": (0.99, 0, 0.01, 0.10, 20.0, 0.10),
            "curriculum_seed1_5601": (0.95, 1, 0.02, 0.08, 22.0, 0.09),
            "curriculum_seed2_5601": (0.90, 2, 0.03, 0.07, 24.0, 0.08),
        }
        for policy in module.POLICIES:
            success, falls, collision, rmse, power, tilt = policy_values[policy]
            for seed_offset, seed in enumerate(module.EVALUATION_SEEDS):
                directory = root / policy / f"eval_seed_{seed}"
                directory.mkdir(parents=True)
                for terrain in module.TERRAINS:
                    payload = {
                        "terrain": terrain,
                        "seed": seed,
                        "success_rate": success - seed_offset * 0.001,
                        "fall_count": falls,
                        "collision_sample_rate": collision,
                        "tracking_rmse_mps": rmse,
                        "mean_mechanical_power_w": power,
                        "mean_tilt_rad": tilt,
                    }
                    (directory / f"{terrain}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

    def test_complete_matrix_is_ranked_and_written(self):
        module = load_aggregator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_complete_matrix(root, module)

            rows = module.collect_results(root)
            self.assertEqual(len(rows), 63)
            summary = module.summarize(rows)
            self.assertEqual(
                [row["policy"] for row in summary],
                [
                    "baseline5001",
                    "curriculum_seed1_5601",
                    "curriculum_seed2_5601",
                ],
            )
            self.assertEqual(summary[0]["rank"], 1)
            self.assertEqual(summary[0]["evaluation_seed_count"], 3)
            self.assertEqual(summary[0]["terrain_evaluation_count"], 21)
            self.assertEqual(summary[1]["total_falls"], 21)

            module.write_outputs(root, summary)
            manifest = module.write_sha256_manifest(root)
            self.assertTrue((root / "formal_policy_summary.csv").is_file())
            self.assertTrue((root / "formal_policy_summary.json").is_file())
            self.assertIn("formal_policy_summary.csv", manifest.read_text(encoding="utf-8"))

    def test_missing_terrain_result_is_rejected(self):
        module = load_aggregator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_complete_matrix(root, module)
            missing = (
                root
                / "baseline5001"
                / f"eval_seed_{module.EVALUATION_SEEDS[0]}"
                / f"{module.TERRAINS[0]}.json"
            )
            missing.unlink()

            with self.assertRaisesRegex(ValueError, "Expected 63 terrain results"):
                module.collect_results(root)


class FormalLauncherContractTests(unittest.TestCase):
    def test_gpu_selection_defaults_to_two_physical_gpus(self):
        result = run_bash(
            f'unset QRC_EVAL_GPUS; export QRC_SOURCE_ONLY=1; '
            f'source "{LAUNCHER_RELATIVE}"; parse_gpu_selection; '
            "declare -p PHYSICAL_GPUS",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('([0]="0" [1]="1")', result.stdout)

    def test_gpu_selection_accepts_one_gpu(self):
        result = source_launcher(
            "parse_gpu_selection; declare -p PHYSICAL_GPUS",
            env={"QRC_EVAL_GPUS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('([0]="1")', result.stdout)

    def test_gpu_selection_rejects_invalid_values(self):
        for value in ("", "0 0", "gpu0", "-1", "0 1 2"):
            with self.subTest(value=value):
                result = source_launcher(
                    "parse_gpu_selection",
                    env={"QRC_EVAL_GPUS": value},
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_gpu_selection_rejects_an_unavailable_index(self):
        result = source_launcher(
            "parse_gpu_selection; validate_gpu_selection_availability 1",
            env={"QRC_EVAL_GPUS": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unavailable", result.stderr.lower())

    def test_gpu_selection_rejects_an_overflowing_index(self):
        result = source_launcher(
            "parse_gpu_selection; validate_gpu_selection_availability 1",
            env={"QRC_EVAL_GPUS": "18446744073709551616"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unavailable", result.stderr.lower())

    def test_one_gpu_queue_contains_all_nine_jobs_once(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
            root = Path(directory)
            relative_root = root.relative_to(REPOSITORY_ROOT).as_posix()
            (root / "workers").mkdir()
            result = source_launcher(
                'PHYSICAL_GPUS=(1); BASE_RUN=base; '
                'CURRICULUM1_RUN=curriculum1; CURRICULUM2_RUN=curriculum2; '
                f"write_queues {shlex.quote(relative_root)}",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            queue = root / "workers" / "gpu1.queue"
            records = queue.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 9)
            self.assertEqual(len(set(records)), 9)
            self.assertFalse((root / "workers" / "gpu0.queue").exists())

    def test_launcher_encodes_fixed_protocol_and_isolated_workers(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("EVALUATION_SEEDS=(20260911 20260912 20260913)", source)
        self.assertIn(
            'TERRAINS="flat,slope,rough_slope,stairs_up,stairs_down,obstacles,wave"',
            source,
        )
        self.assertIn('NUM_ENVS="256"', source)
        self.assertIn('DURATION_SECONDS="20"', source)
        self.assertIn('local GPU="$1"', source)
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPU}"', source)
        self.assertIn("Setting seed: 20260911", source)
        self.assertIn("GO2 MULTI-TERRAIN BENCHMARK PASS", source)
        self.assertIn("WORKER_${GPU}_PASS", source)
        self.assertIn("MATRIX_FAILED", source)
        self.assertIn("model_5001.pt", source)
        self.assertIn("model_5601.pt", source)
        self.assertIn("aggregate_formal_matrix.py", source)
        self.assertIn("latest_formal_eval_dir.txt", source)
        self.assertIn('QRC_EVAL_GPUS="${QRC_EVAL_GPUS-0 1}"', source)
        self.assertIn('echo "physical_gpus=${PHYSICAL_GPUS[*]}"', source)
        self.assertIn('seed_smoke_gate "${output_root}" "${PHYSICAL_GPUS[0]}"', source)
        self.assertIn('for gpu in "${PHYSICAL_GPUS[@]}"; do', source)
        self.assertIn("read_manifest_gpu_list", source)
        self.assertNotIn("WORKER_0_FAILED", source)
        self.assertNotIn("WORKER_1_FAILED", source)
        self.assertNotIn("WORKER_0_PASS", source)
        self.assertNotIn("WORKER_1_PASS", source)
        self.assertNotIn("watch ", source)


class FormalStatusContractTests(unittest.TestCase):
    def test_status_is_read_only_and_reports_acceptance_markers(self):
        source = STATUS_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("latest_formal_eval_dir.txt", source)
        self.assertIn('${1:-', source)
        self.assertIn("COMPLETE", source)
        self.assertIn("FAILED", source)
        self.assertIn("physical_gpus", source)
        self.assertIn("PHYSICAL_GPUS=(0 1)", source)
        self.assertIn('for gpu in "${PHYSICAL_GPUS[@]}"; do', source)
        self.assertIn('WORKER_${gpu}_PASS', source)
        self.assertIn('WORKER_${gpu}_FAILED', source)
        self.assertIn("FORMAL_MATRIX_PASS", source)
        self.assertIn("kill -0", source)
        self.assertNotIn("watch ", source)
        for invocation in re.findall(r"\bkill\s+[^\n]+", source):
            self.assertTrue(invocation.startswith("kill -0"), invocation)

    def test_status_uses_manifest_gpu_selection(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
            root = Path(directory)
            (root / "workers").mkdir()
            (root / "manifest.txt").write_text(
                "physical_gpus=1\n", encoding="utf-8"
            )
            relative_root = root.relative_to(REPOSITORY_ROOT).as_posix()
            result = run_bash(
                f'bash "{STATUS_RELATIVE}" {shlex.quote(relative_root)}',
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("gpu1_pid=missing", result.stdout)
            self.assertNotIn("gpu0_pid=", result.stdout)
            self.assertIn("marker=workers/WORKER_1_PASS absent", result.stdout)
            self.assertNotIn("marker=workers/WORKER_0_PASS", result.stdout)

    def test_status_falls_back_to_two_gpus_for_legacy_manifest(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
            root = Path(directory)
            (root / "workers").mkdir()
            (root / "manifest.txt").write_text(
                "repository_commit=legacy\n", encoding="utf-8"
            )
            relative_root = root.relative_to(REPOSITORY_ROOT).as_posix()
            result = run_bash(
                f'bash "{STATUS_RELATIVE}" {shlex.quote(relative_root)}',
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("gpu0_pid=missing", result.stdout)
            self.assertIn("gpu1_pid=missing", result.stdout)
            self.assertIn("marker=workers/WORKER_0_PASS absent", result.stdout)
            self.assertIn("marker=workers/WORKER_1_PASS absent", result.stdout)


class ShellPortabilityTests(unittest.TestCase):
    def test_shell_scripts_are_pinned_to_lf(self):
        attributes = GITATTRIBUTES.read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attributes)
        for path in (LAUNCHER, STATUS_SCRIPT):
            self.assertNotIn(b"\r\n", path.read_bytes(), str(path))


if __name__ == "__main__":
    unittest.main()
