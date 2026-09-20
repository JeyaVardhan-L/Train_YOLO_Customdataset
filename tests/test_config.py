"""Tests for dataset and training YAML configuration files."""

import unittest
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestConfigurations(unittest.TestCase):
    """Test YAML configurations for structure, classes, and path validity."""

    def test_configs_data_yaml_structure(self):
        config_path = PROJECT_ROOT / "configs" / "data.yaml"
        self.assertTrue(config_path.exists(), f"File {config_path} must exist")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("path", cfg)
        self.assertIn("train", cfg)
        self.assertIn("val", cfg)
        self.assertIn("names", cfg)

        names = cfg["names"]
        self.assertEqual(len(names), 3)
        self.assertEqual(names[0], "Aircrafts")
        self.assertEqual(names[1], "Bird")
        self.assertEqual(names[2], "Drone")

        # Verify path resolution from configs/ location
        resolved_path = (config_path.parent / cfg["path"]).resolve()
        self.assertTrue(resolved_path.exists(), f"Path '{cfg['path']}' must resolve to existing directory")

        train_dir = resolved_path / cfg["train"]
        val_dir = resolved_path / cfg["val"]
        self.assertTrue(train_dir.exists(), f"Train directory {train_dir} must exist")
        self.assertTrue(val_dir.exists(), f"Val directory {val_dir} must exist")

    def test_root_data_yaml_structure(self):
        config_path = PROJECT_ROOT / "data.yaml"
        self.assertTrue(config_path.exists(), f"File {config_path} must exist")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("path", cfg)
        self.assertIn("train", cfg)
        self.assertIn("val", cfg)
        self.assertIn("names", cfg)

        resolved_path = (PROJECT_ROOT / cfg["path"]).resolve()
        self.assertTrue(resolved_path.exists(), f"Root path '{cfg['path']}' must resolve")

    def test_train_config_yaml_structure(self):
        train_cfg_path = PROJECT_ROOT / "configs" / "train_config.yaml"
        self.assertTrue(train_cfg_path.exists(), f"File {train_cfg_path} must exist")

        with open(train_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg.get("task"), "detect")
        self.assertEqual(cfg.get("mode"), "train")
        self.assertIn("epochs", cfg)
        self.assertIn("batch", cfg)
        self.assertIn("imgsz", cfg)


if __name__ == "__main__":
    unittest.main()
