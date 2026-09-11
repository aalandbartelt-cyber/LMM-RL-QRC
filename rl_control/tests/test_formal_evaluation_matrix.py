import importlib.util
import json
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


def load_aggregator():
    spec = importlib.util.spec_from_file_location("aggregate_formal_matrix", AGGREGATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        self.assertNotIn("watch ", source)


if __name__ == "__main__":
    unittest.main()
