"""Config resolution, run directory and stage-state tests (A01, A02, B08 lifecycle)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from tests import helpers

from uavdet import artifacts
from uavdet.config import (
    ConfigError,
    apply_overrides,
    config_digest,
    load_config,
    protocol_digest,
    training_protocol,
    validate_run_id,
)
from uavdet.types import RunError


class ConfigLoadingTests(helpers.TempDirTestCase):
    def test_paths_resolve_against_config_file_not_cwd(self):
        project = self.project()
        config = load_config(project.config_path)
        expected_root = Path(config["project"]["root"])
        self.assertEqual(expected_root, project.root.resolve())
        self.assertEqual(Path(config["data"]["root"]), project.data_root.resolve())
        self.assertEqual(Path(config["project"]["output_root"]), project.root.resolve() / "outputs")
        self.assertEqual(Path(config["train"]["init_weights"]),
                         project.root.resolve() / "weights" / "yolo11s.pt")

        # Same config from a completely different working directory.
        with tempfile.TemporaryDirectory() as other:
            previous = os.getcwd()
            try:
                os.chdir(other)
                again = load_config(project.config_path)
            finally:
                os.chdir(previous)
        self.assertEqual(again["data"]["root"], config["data"]["root"])
        self.assertEqual(again["project"]["root"], config["project"]["root"])

    def test_command_line_override_wins(self):
        project = self.project()
        config = load_config(project.config_path, {"train.epochs": 50, "data.root": "alt_data"})
        self.assertEqual(config["train"]["epochs"], 50)
        self.assertEqual(Path(config["data"]["root"]), project.root.resolve() / "alt_data")

    def test_unknown_keys_are_rejected(self):
        project = self.project()
        raw = yaml.safe_load(project.config_path.read_text())
        raw["train"]["epoch_count"] = 10
        project.config_path.write_text(yaml.safe_dump(raw))
        with self.assertRaises(ConfigError) as ctx:
            load_config(project.config_path)
        self.assertIn("epoch_count", str(ctx.exception))

        with self.assertRaises(ConfigError):
            load_config(project.config_path, {"train.nope": 1})
        with self.assertRaises(ConfigError):
            load_config(project.config_path, {"nope.key": 1})

    def test_value_validation(self):
        project = self.project()
        for override in ({"train.epochs": "five"}, {"train.imgsz": 650},
                         {"eval.conf": 1.5}, {"train.device": "gpu0"},
                         {"figures.formats": ["png", "docx"]}):
            with self.assertRaises(ConfigError, msg=str(override)):
                load_config(project.config_path, override)

    def test_class_order_is_fixed(self):
        project = self.project()
        raw = yaml.safe_load(project.config_path.read_text())
        raw["data"]["class_names"][0], raw["data"]["class_names"][1] = (
            raw["data"]["class_names"][1], raw["data"]["class_names"][0])
        project.config_path.write_text(yaml.safe_dump(raw))
        with self.assertRaises(ConfigError):
            load_config(project.config_path)

    def test_extends_chain(self):
        project = self.project()
        base = yaml.safe_load(project.config_path.read_text())
        base["scope"] = "smoke"
        base["train"]["epochs"] = 2
        base["train"]["batch"] = 2
        base["train"]["close_mosaic"] = 0
        project.config_path.write_text(yaml.safe_dump(base, sort_keys=False))
        (project.config_path.parent / "smoke.yaml").write_text(
            yaml.safe_dump({"extends": "project.yaml", "train": {"epochs": 3}}, sort_keys=False)
        )
        smoke_config = load_config(project.config_path.parent / "smoke.yaml")
        self.assertEqual(smoke_config["scope"], "smoke")
        self.assertEqual(smoke_config["train"]["epochs"], 3)
        self.assertEqual(Path(smoke_config["data"]["root"]), project.data_root.resolve())

    def test_circular_extends_is_rejected(self):
        project = self.project()
        (project.config_path.parent / "a.yaml").write_text("extends: b.yaml\n")
        (project.config_path.parent / "b.yaml").write_text("extends: a.yaml\n")
        with self.assertRaises(ConfigError):
            load_config(project.config_path.parent / "a.yaml")

    def test_digests_are_stable_and_split_paths_from_protocol(self):
        project = self.project()
        config = load_config(project.config_path)
        self.assertEqual(config_digest(config), config_digest(load_config(project.config_path)))
        moved = apply_overrides(config, {"project.workspace_root": "other_ws",
                                         "train.device": "cpu", "train.workers": 0})
        self.assertNotEqual(config_digest(moved), config_digest(config))
        self.assertEqual(protocol_digest(moved), protocol_digest(config))
        changed = apply_overrides(config, {"train.batch": 4})
        self.assertNotEqual(protocol_digest(changed), protocol_digest(config))
        self.assertIn("train", training_protocol(config))


class RunIdTests(unittest.TestCase):
    def test_valid_and_invalid_ids(self):
        self.assertEqual(validate_run_id("formal_001-x"), "formal_001-x")
        for bad in ("../escape", "a/b", "..", "", "a" * 65, "run id", "run.id"):
            with self.assertRaises(ConfigError, msg=bad):
                validate_run_id(bad)


class RunLifecycleTests(helpers.TempDirTestCase):
    def _open(self, project, run_id="run_a", extra=None):
        return artifacts.open_run(project.config(extra), run_id)

    def test_open_run_creates_manifest_and_directories(self):
        project = self.project()
        run = self._open(project)
        self.assertTrue(run.manifest_path.exists())
        self.assertTrue(run.resolved_config_path.exists())
        for directory in (run.audit_dir, run.train_dir, run.evaluation_dir,
                          run.benchmark_dir, run.logs_dir):
            self.assertTrue(directory.is_dir(), directory)
        self.assertEqual(run.manifest["scope"], "formal")
        self.assertEqual(set(run.manifest["stages"]), {"audit", "train", "evaluation",
                                                      "benchmark", "figures"})
        for stage in run.manifest["stages"].values():
            self.assertEqual(stage["status"], "pending")

    def test_reopen_keeps_identity(self):
        project = self.project()
        run = self._open(project)
        created = run.manifest["created_at"]
        again = self._open(project)
        self.assertEqual(again.manifest["created_at"], created)
        self.assertEqual(again.run_dir, run.run_dir)

    def test_run_id_cannot_escape_output_root(self):
        project = self.project()
        with self.assertRaises(ConfigError):
            artifacts.open_run(project.config(), "../escape")
        self.assertFalse((project.root / "escape").exists())

    def test_incomplete_directory_without_manifest_is_not_adopted(self):
        project = self.project()
        stray = project.outputs_dir / "stray"
        stray.mkdir(parents=True)
        (stray / "notes.txt").write_text("someone else's data")
        with self.assertRaises(RunError):
            artifacts.open_run(project.config(), "stray")

    def test_completed_stage_refuses_rerun(self):
        project = self.project()
        run = self._open(project)
        run.begin_stage("audit")
        run.complete_stage("audit", status="ok")
        self.assertEqual(run.stage_status("audit"), "complete")
        with self.assertRaises(RunError):
            run.begin_stage("audit")
        run.begin_stage("audit", allow_repeat=True)
        self.assertEqual(run.stage_status("audit"), "running")
        self.assertEqual(run.stage_block("audit")["attempts"], 2)

    def test_failed_stage_is_not_complete(self):
        project = self.project()
        run = self._open(project)
        run.begin_stage("train")
        run.fail_stage("train", "synthetic boom")
        self.assertEqual(run.stage_status("train"), "failed")
        self.assertIn("synthetic boom", run.stage_block("train")["error"])
        with self.assertRaises(RunError):
            run.complete_stage("train")

    def test_skipped_is_not_complete(self):
        project = self.project()
        run = self._open(project)
        run.begin_stage("benchmark")
        artifacts.skip_stage(run, "benchmark", "not part of this run")
        self.assertEqual(run.stage_status("benchmark"), "skipped")
        self.assertFalse(artifacts.stage_products_ok(run, "benchmark"))

    def test_stale_running_state_becomes_interrupted(self):
        project = self.project()
        run = self._open(project)
        run.begin_stage("train")
        # Simulate a process that died without writing a transition.
        run.manifest["stages"]["train"]["status"] = "running"
        artifacts.write_json_atomic(run.manifest_path, run.manifest)
        reloaded = artifacts.open_run(project.config(), "run_a")
        reloaded.begin_stage("train")
        self.assertEqual(reloaded.stage_status("train"), "running")
        kinds = [event["kind"] for event in reloaded.manifest["events"]]
        self.assertIn("stage_interrupted", kinds)

    def test_protocol_change_forces_a_new_run(self):
        project = self.project()
        self._open(project)
        with self.assertRaises(RunError):
            artifacts.open_run(project.config({"train.batch": 8}), "run_a")
        # Path/device/worker moves are allowed and recorded.
        moved = artifacts.open_run(
            project.config({"train.device": "cpu", "train.workers": 0,
                            "project.workspace_root": "ws2"}), "run_a")
        kinds = [event["kind"] for event in moved.manifest["events"]]
        self.assertIn("config_rebase", kinds)

    def test_atomic_json_refuses_nan(self):
        target = self.tmp_path / "bad.json"
        with self.assertRaises(ValueError):
            artifacts.write_json_atomic(target, {"value": float("nan")})
        self.assertFalse(target.exists())

    def test_run_summary_reports_stages(self):
        project = self.project()
        run = self._open(project)
        run.begin_stage("audit")
        run.complete_stage("audit", status="ok")
        summary = artifacts.run_summary(run)
        self.assertEqual(summary["stages"]["audit"]["status"], "complete")
        self.assertEqual(summary["scope"], "formal")
        self.assertFalse(summary["is_formal_complete"])

    def test_record_environment_writes_packages(self):
        project = self.project()
        run = self._open(project)
        info = artifacts.record_environment(run, "cpu")
        self.assertTrue(run.environment_path.exists())
        self.assertIn("python", info)
        self.assertIn("packages", info)
        self.assertEqual(info["requested_device"], "cpu")


class FixtureSmokeTests(helpers.TempDirTestCase):
    """The shared fixtures themselves must be honest (not silently broken)."""

    def test_fake_run_writes_every_schema_file(self):
        project = self.project()
        run = helpers.make_fake_run(project)
        required = [
            "manifest.json", "config_resolved.yaml",
            "audit/data_manifest.json", "audit/dataset_stats.json", "audit/issues.csv",
            "train/results.csv", "train/args.yaml", "train/weights/best.pt",
            "evaluation/metrics.json", "evaluation/per_class.csv",
            "evaluation/curves.npz", "evaluation/curves_meta.json",
            "evaluation/confusion_matrix.npz", "evaluation/confusion_meta.json",
            "evaluation/predictions.jsonl", "evaluation/ground_truth.jsonl",
            "evaluation/cases.json", "benchmark/summary.json", "benchmark/timings.csv",
        ]
        for relative in required:
            self.assertTrue((run.run_dir / relative).exists(), relative)
        for stage in ("audit", "train", "evaluation", "benchmark"):
            self.assertEqual(run.stage_status(stage), "complete")

    def test_project_helper_counts_match_audit(self):
        project = self.project()
        audit = __import__("uavdet.data", fromlist=["audit_dataset"]).audit_dataset(project.config())
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["splits"]["train"]["num_images"], project.dataset["counts"]["train"])
        self.assertEqual(audit["splits"]["val"]["num_images"], project.dataset["counts"]["val"])


if __name__ == "__main__":
    unittest.main()
