import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
TRAINER = ROOT / "legacy_isaacgym" / "train_go2_curriculum_v2.py"
LAUNCHER = ROOT / "legacy_isaacgym" / "start_two_seed_curriculum_v2.sh"


def assigned_literal(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment not found: {name}")


class LegacyIsaacGymCurriculumTests(unittest.TestCase):
    def test_curriculum_is_balanced_and_starts_easier(self):
        source = TRAINER.read_text(encoding="utf-8")
        compile(source, str(TRAINER), "exec")
        proportions = assigned_literal(source, "TERRAIN_PROPORTIONS")
        self.assertAlmostEqual(sum(proportions), 1.0)
        self.assertEqual(len(proportions), 9)
        self.assertLessEqual(proportions[3], 0.25)
        self.assertGreaterEqual(proportions[8], 0.15)
        self.assertIn("max_init_terrain_level = 2", source)
        self.assertIn('QRC_LEARNING_RATE", "0.0003"', source)
        self.assertIn('QRC_EXPECT_SOURCE_ITERATION", "5001"', source)
        self.assertNotIn("Jul28_", source)

    def test_launcher_uses_two_isolated_physical_gpus(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPU}"', source)
        self.assertIn('--seed="${SEED}"', source)
        self.assertIn("model_${CHECKPOINT}.pt", source)
        self.assertIn('"${ENV_PREFIX}/bin/python" -u', source)
        self.assertIn("nohup", source)
        self.assertIn("TRAINING PROCESS CHECK PASS", source)
        self.assertNotIn("torchrun", source)


if __name__ == "__main__":
    unittest.main()
