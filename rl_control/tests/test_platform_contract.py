import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "platform" / "preflight_self5000.py"
SPEC = importlib.util.spec_from_file_location("qrc_self5000_preflight", MODULE_PATH)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


class PlatformContractTests(unittest.TestCase):
    def test_version_lock_contains_exact_upstream_revisions(self):
        lock_path = MODULE_PATH.with_name("versions.json")
        locks = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(locks["isaac_sim"], "5.1.0")
        self.assertEqual(locks["isaac_lab_commit"], "3c6e67bb5c7ada942a6d1884ab69338f57596f77")
        self.assertEqual(locks["unitree_rl_lab_commit"], "1330cb0d4236c4abd6bf1efec05a4fce7ad79678")
        self.assertEqual(locks["rsl_rl"], "2.3.1")

    def test_old_driver_and_implicit_a100_are_blocked(self):
        snapshot = {
            "system": "Linux",
            "glibc": "2.35",
            "free_disk_gib": 100.0,
            "gpus": [
                {"name": "NVIDIA A100-PCIE-40GB", "driver": "535.129.03"},
                {"name": "NVIDIA A100-PCIE-40GB", "driver": "535.129.03"},
            ],
        }
        report = PREFLIGHT.assess_host(snapshot, requested_gpus=2, allow_unsupported_a100=False)
        self.assertFalse(report["ready"])
        self.assertIn("driver_too_old", report["errors"])
        self.assertIn("a100_not_supported_by_isaac_sim", report["errors"])

    def test_a100_can_only_reach_experimental_gate_with_current_driver(self):
        snapshot = {
            "system": "Linux",
            "glibc": "2.35",
            "free_disk_gib": 100.0,
            "gpus": [
                {"name": "NVIDIA A100-PCIE-40GB", "driver": "580.65.06"},
                {"name": "NVIDIA A100-PCIE-40GB", "driver": "580.65.06"},
            ],
        }
        report = PREFLIGHT.assess_host(snapshot, requested_gpus=2, allow_unsupported_a100=True)
        self.assertTrue(report["ready"])
        self.assertIn("a100_experimental_physics_only", report["warnings"])

    def test_missing_second_gpu_is_blocked(self):
        snapshot = {
            "system": "Linux",
            "glibc": "2.35",
            "free_disk_gib": 100.0,
            "gpus": [{"name": "NVIDIA RTX 6000 Ada", "driver": "580.65.06"}],
        }
        report = PREFLIGHT.assess_host(snapshot, requested_gpus=2, allow_unsupported_a100=False)
        self.assertFalse(report["ready"])
        self.assertIn("insufficient_gpus", report["errors"])

    def test_unitree_model_patch_requires_exact_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unitree.py"
            source.write_text("header\nUNITREE_MODEL_DIR = MISSING\nfooter\n", encoding="utf-8")
            changed = PREFLIGHT.configure_unitree_model_dir(source, Path("/work/unitree_model"))
            self.assertTrue(changed)
            text = source.read_text(encoding="utf-8")
            expected = json.dumps(str(Path("/work/unitree_model").resolve()))
            self.assertIn(f"UNITREE_MODEL_DIR = {expected}", text)
            self.assertFalse(PREFLIGHT.configure_unitree_model_dir(source, Path("/work/unitree_model")))

    def test_install_script_is_pinned_and_non_destructive(self):
        script = MODULE_PATH.with_name("bootstrap_self5000.sh").read_text(encoding="utf-8")
        self.assertNotIn("rm -rf", script)
        self.assertIn('preflight_self5000.py" --phase host', script)
        self.assertIn("isaacsim[all,extscache]==${ISAAC_SIM_VERSION}", script)
        self.assertIn("rsl-rl-lib==${RSL_RL_VERSION}", script)
        self.assertNotIn("--video", script)

    def test_paid_training_scripts_isolate_each_seed_to_one_gpu(self):
        single = MODULE_PATH.with_name("start_training.sh").read_text(encoding="utf-8")
        dual = MODULE_PATH.with_name("start_two_seed_stage.sh").read_text(encoding="utf-8")
        progress = MODULE_PATH.with_name("show_progress.sh").read_text(encoding="utf-8")
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPU}"', single)
        self.assertIn("--device cuda:0", single)
        self.assertIn("nohup", single)
        self.assertNotIn("--video", single)
        self.assertIn('start_training.sh" "${STAGE}" 0 1', dual)
        self.assertIn('start_training.sh" "${STAGE}" 1 2', dual)
        self.assertIn("Learning iteration", progress)
        self.assertIn("nvidia-smi", progress)

    def test_result_wrapper_captures_provenance_before_packaging(self):
        wrapper = MODULE_PATH.with_name("package_results.sh").read_text(encoding="utf-8")
        self.assertIn("nvidia-smi", wrapper)
        self.assertIn("pip freeze", wrapper)
        self.assertIn("package_training_run.py", wrapper)
        self.assertIn("PACKAGE VERIFIED", wrapper)

    def test_export_wrapper_uses_explicit_checkpoint_without_camera(self):
        wrapper = MODULE_PATH.with_name("export_policy.sh").read_text(encoding="utf-8")
        self.assertIn("--checkpoint", wrapper)
        self.assertIn("policy.pt", wrapper)
        self.assertIn("policy.onnx", wrapper)
        self.assertIn("CUDA_VISIBLE_DEVICES", wrapper)
        self.assertNotIn("--video", wrapper)

    def test_runbook_keeps_host_gate_before_install_and_training(self):
        runbook = (
            MODULE_PATH.parents[2] / "docs" / "training" / "SELF5000_GO2_CAMPUS.md"
        ).read_text(encoding="utf-8")
        gate = runbook.index("--phase host")
        install = runbook.index("bootstrap_self5000.sh")
        train = runbook.index("start_two_seed_stage.sh")
        self.assertLess(gate, install)
        self.assertLess(install, train)
        self.assertIn("go2-campus-2a100-s1s2-0908", runbook)
        self.assertIn("driver_too_old", runbook)


if __name__ == "__main__":
    unittest.main()
