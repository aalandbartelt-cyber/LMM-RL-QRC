import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ArtifactPackagingTests(unittest.TestCase):
    def test_archive_contains_both_roots_and_self_verifies(self):
        packaging = load_script("package_training_run.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            checkpoint_dir = root / "checkpoint"
            run_dir.mkdir()
            checkpoint_dir.mkdir()
            (run_dir / "train.log").write_text("Learning iteration 3/10\n", encoding="utf-8")
            (checkpoint_dir / "model_3.pt").write_bytes(b"checkpoint")
            output = root / "go2_result.tar.gz"

            report = packaging.create_training_archive(run_dir, checkpoint_dir, output)

            self.assertTrue(report["verified"])
            self.assertEqual(report["archive_sha256"], packaging.sha256_file(output))
            checksum = output.with_name("go2_result.sha256")
            expected = checksum.read_text(encoding="utf-8").split()[0]
            self.assertEqual(expected, hashlib.sha256(output.read_bytes()).hexdigest())
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
                self.assertIn("training_run/train.log", names)
                self.assertIn("checkpoint/model_3.pt", names)
                manifest = json.load(archive.extractfile("MANIFEST.json"))
            self.assertEqual(len(manifest["files"]), 2)

    def test_symlinks_are_rejected(self):
        packaging = load_script("package_training_run.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            target = root / "target.txt"
            target.write_text("data", encoding="utf-8")
            link = source / "link.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks are unavailable on this Windows host")
            with self.assertRaisesRegex(ValueError, "symlink"):
                packaging.collect_files(source, "training_run")


class TrainingLogPlotTests(unittest.TestCase):
    def test_rsl_training_blocks_become_numeric_rows(self):
        plotting = load_script("plot_training_metrics.py")
        text = """
Learning iteration 10/20
Value function loss: 0.0056
Surrogate loss: -0.0062
Mean reward: 21.07
Mean episode length: 1225.02
Iteration time: 1.70s
Learning iteration 11/20
Value function loss: 0.0040
Mean reward: 22.50
Mean episode length: 1230.00
"""
        rows = plotting.parse_training_log(text)
        self.assertEqual([row["iteration"] for row in rows], [10, 11])
        self.assertAlmostEqual(rows[0]["mean_reward"], 21.07)
        self.assertAlmostEqual(rows[0]["iteration_time_s"], 1.70)
        self.assertAlmostEqual(rows[1]["value_function_loss"], 0.004)

    def test_empty_log_is_rejected(self):
        plotting = load_script("plot_training_metrics.py")
        with self.assertRaisesRegex(ValueError, "Learning iteration"):
            plotting.parse_training_log("simulator startup only")


if __name__ == "__main__":
    unittest.main()
