"""End-to-end wiring tests: CLI -> stages -> artefacts -> figures.

The heavy engines are replaced by injected fakes at the documented seams
(``training.default_yolo_factory`` and the evaluation/benchmark command
adapters). Everything else is the real code: config resolution, audit, dataset
view copies, run manifest/stage transitions, figure rendering and the CLI.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from tests import helpers

from uavdet import artifacts, cli, report
from uavdet.types import RunError


def _write_evaluation_fixture(run, project, data_yaml: Path, weights: Path) -> dict:
    """A schema-correct evaluation directory (stands in for the native pass)."""
    config = run.config
    view_root = Path(data_yaml).parent
    images = sorted((view_root / "images" / "val").glob("*.jpg"))
    eval_dir = run.evaluation_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = [
        [(0, (4.0, 4.0, 18.0, 18.0)), (3, (28.0, 18.0, 46.0, 34.0))] for _ in images
    ]
    helpers.make_predictions_jsonl(eval_dir / "predictions.jsonl", images, gt_rows)
    dataset_digest = (run.manifest.get("dataset_view") or {}).get("digest", "0" * 64)
    helpers.make_metrics_json(
        eval_dir / "metrics.json",
        scope=run.scope,
        num_images=len(images),
        num_instances=sum(len(rows) for rows in gt_rows),
        weights_sha256=artifacts.sha256_file(weights),
        dataset_manifest_sha256=dataset_digest,
        data_view_root=str(view_root),
    )
    # one class without GT -> must be N/A, never zero
    helpers.make_per_class_csv(eval_dir / "per_class.csv", valid=[0, 1, 3, 4])
    helpers.make_curves_npz(eval_dir / "curves.npz")
    meta = {
        "source": {"evaluator": "fake", "version": "test"},
        "class_order": list(__import__("uavdet.types", fromlist=["CLASS_NAMES"]).CLASS_NAMES),
        "mean_row_index": 10,
        "smoothing": "none",
        "arrays": {},
        "valid_classes": [0, 1, 3, 4],
    }
    artifacts.write_json_atomic(eval_dir / "curves_meta.json", meta)
    matrix = __import__("numpy").zeros((11, 11), dtype="int64")
    for index in range(10):
        matrix[index, index] = 10 + index
    matrix[10, 10] = 3
    matrix[0, 1] = 2
    matrix[10, 0] = 1
    __import__("numpy").savez_compressed(eval_dir / "confusion_matrix.npz", matrix=matrix)
    artifacts.write_json_atomic(eval_dir / "confusion_meta.json", {
        "true_axis": "row", "pred_axis": "column", "background_index": 10,
        "class_order": meta["class_order"], "conf": config["eval"]["conf"],
        "iou": config["eval"]["iou"], "source_version": "test",
    })
    helpers.make_cases_json(eval_dir / "cases.json", [image.stem for image in images],
                            [images[0].stem] if images else [])
    return {
        "map50": 0.42992, "map50_95": 0.25111, "precision": 0.78123, "recall": 0.65234,
        "num_images": len(images), "num_instances": sum(len(rows) for rows in gt_rows),
        "status": "ok", "warnings": [], "weights_sha256": artifacts.sha256_file(weights),
    }


class PipelineEndToEndTests(helpers.TempDirTestCase):
    def _project(self):
        project = self.project(n_train=4, n_val=2)
        weights = project.root / "weights" / "yolo11s.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"synthetic-pretrained-weights")
        return project

    def _patch_engines(self, project):
        """Context manager patching the three injected seams."""
        calls = {"train": 0, "eval": 0, "bench": 0}
        stack = mock.patch.object(
            __import__("uavdet.training", fromlist=["default_yolo_factory"]),
            "default_yolo_factory",
            side_effect=lambda weights: helpers.RecordingYOLO(weights),
        )
        stack.start()

        def fake_evaluate(config, run, weights, split="val", **kwargs):
            calls["eval"] += 1
            data_yaml = kwargs.get("data_yaml") or run.manifest["dataset_view"]["data_yaml"]
            return _write_evaluation_fixture(run, project, Path(data_yaml), Path(weights))

        def fake_benchmark(config, run, weights, **kwargs):
            calls["bench"] += 1
            helpers.make_benchmark(run, num_images=3, repeats=2)
            return {"status": "ok", "mean_ms": 9.4, "p50_ms": 9.2, "p95_ms": 10.1,
                    "fps": 106.3, "num_images": 3, "device": "cpu"}

        from uavdet import benchmark as benchmark_module
        from uavdet import evaluation as evaluation_module

        stack2 = mock.patch.object(evaluation_module, "evaluate_model", fake_evaluate)
        stack3 = mock.patch.object(benchmark_module, "run_benchmark", fake_benchmark)
        stack2.start()
        stack3.start()

        def stop():
            for patcher in (stack3, stack2, stack):
                patcher.stop()

        return calls, stop

    def _cli(self, project, *argv) -> int:
        return cli.main(["--config", str(project.config_path), *argv])

    def test_full_pipeline_produces_every_stage_and_figure(self):
        project = self._project()
        before = helpers.sha_tree(project.data_root)
        calls, stop = self._patch_engines(project)
        try:
            self.assertEqual(self._cli(project, "check", "--run-id", "e2e"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "e2e"), 0)
            self.assertEqual(self._cli(project, "val", "--run-id", "e2e"), 0)
            self.assertEqual(self._cli(project, "benchmark", "--run-id", "e2e"), 0)
            self.assertEqual(self._cli(project, "plot", "--run-id", "e2e"), 0)
        finally:
            stop()

        run_dir = project.outputs_dir / "e2e"
        manifest = artifacts.read_manifest(run_dir)
        for stage in ("audit", "train", "evaluation", "benchmark", "figures"):
            self.assertEqual(manifest["stages"][stage]["status"], "complete", stage)

        self.assertTrue((run_dir / "audit" / "dataset_stats.json").exists())
        for name in ("best.pt", "last.pt"):
            self.assertTrue((run_dir / "train" / "weights" / name).exists())
        self.assertTrue((run_dir / "evaluation" / "metrics.json").exists())
        self.assertTrue((run_dir / "benchmark" / "summary.json").exists())

        figure_manifest = artifacts.read_json(
            run_dir / "figures" / "v1" / "figure_manifest.json")
        self.assertEqual(figure_manifest["status"], "ok", figure_manifest["missing"])
        self.assertEqual(sorted(figure_manifest["produced"]),
                         ["F01", "F02", "F03", "F04", "F05", "F06",
                          "F07", "F08", "F09", "F10", "F11", "F12"])
        self.assertTrue((run_dir / "figures" / "v1" / "figure_index.md").exists())
        # one png/pdf/svg per figure
        pngs = sorted((run_dir / "figures" / "v1").glob("*.png"))
        self.assertEqual(len(pngs), 12)

        self.assertEqual(calls["eval"], 1)
        self.assertEqual(calls["bench"], 1)
        self.assertEqual(helpers.sha_tree(project.data_root), before,
                         "the full pipeline must not modify the source dataset")

    def test_resume_trains_from_the_run_checkpoint(self):
        project = self._project()
        calls, stop = self._patch_engines(project)
        try:
            self.assertEqual(self._cli(project, "check", "--run-id", "resume_run"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "resume_run"), 0)
            checkpoint = project.outputs_dir / "resume_run" / "train" / "weights" / "last.pt"
            self.assertTrue(checkpoint.exists())
            self.assertEqual(self._cli(project, "train", "--run-id", "resume_run",
                                       "--resume", str(checkpoint)), 0)
        finally:
            instances = list(helpers.RecordingYOLO.instances)
            stop()
        resumed = [item for item in instances if item.train_kwargs.get("resume")]
        self.assertTrue(resumed, "resume training must reach the YOLO factory with resume=True")

    def test_pipeline_continue_does_not_retrain(self):
        project = self._project()
        calls, stop = self._patch_engines(project)
        try:
            self.assertEqual(self._cli(project, "pipeline", "--run-id", "pipe_run"), 0)
            training_before = len(helpers.RecordingYOLO.instances)
            self.assertEqual(self._cli(project, "pipeline", "--run-id", "pipe_run",
                                       "--continue"), 0)
            self.assertEqual(len(helpers.RecordingYOLO.instances), training_before,
                             "a complete train stage must not be re-executed")
            self.assertEqual(calls["eval"], 1)
        finally:
            stop()

    def test_figure_tag_cannot_be_overwritten(self):
        project = self._project()
        calls, stop = self._patch_engines(project)
        try:
            self.assertEqual(self._cli(project, "check", "--run-id", "tag_run"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "tag_run"), 0)
            self.assertEqual(self._cli(project, "val", "--run-id", "tag_run"), 0)
            self.assertEqual(self._cli(project, "plot", "--run-id", "tag_run"), 0)
            code = self._cli(project, "plot", "--run-id", "tag_run")
            self.assertNotEqual(code, 0)
        finally:
            stop()

    def test_scope_mismatch_refuses_a_report(self):
        """D04: smoke/historical artefacts cannot be mixed into a formal report."""
        project = self._project()
        calls, stop = self._patch_engines(project)
        try:
            self.assertEqual(self._cli(project, "check", "--run-id", "scope_run"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "scope_run"), 0)
            self.assertEqual(self._cli(project, "val", "--run-id", "scope_run"), 0)
        finally:
            stop()
        metrics_path = (project.outputs_dir / "scope_run" / "evaluation" / "metrics.json")
        payload = artifacts.read_json(metrics_path)
        payload["scope"] = "smoke"
        artifacts.write_json_atomic(metrics_path, payload)

        run = artifacts.open_run(project.config(), "scope_run")
        with self.assertRaises(RunError):
            report.build_figures(run, "mixed")

    def test_revalidation_run_reports_training_figures_as_not_applicable(self):
        project = self._project()
        calls, stop = self._patch_engines(project)
        try:
            self.assertEqual(self._cli(project, "check", "--run-id", "reval"), 0)
            weights = project.root / "weights" / "reval.pt"
            weights.write_bytes(b"synthetic-10-class-weights")
            self.assertEqual(self._cli(project, "val", "--run-id", "reval",
                                       "--weights", str(weights)), 0)
            self.assertEqual(self._cli(project, "plot", "--run-id", "reval"), 0)
        finally:
            stop()
        figure_manifest = artifacts.read_json(
            project.outputs_dir / "reval" / "figures" / "v1" / "figure_manifest.json")
        self.assertEqual(figure_manifest["status"], "ok")
        self.assertEqual(sorted(item["figure_id"] for item in figure_manifest["not_applicable"]),
                         ["F04", "F05", "F06"])
        self.assertIn("F01", figure_manifest["produced"])
        self.assertIn("F07", figure_manifest["produced"])

    def test_check_reports_audit_failure_with_nonzero_exit(self):
        project = self.project(n_train=3, n_val=2, anomalies=("corrupt_image",))
        (project.root / "weights").mkdir(exist_ok=True)
        (project.root / "weights" / "yolo11s.pt").write_bytes(b"weights")
        code = self._cli(project, "check", "--run-id", "bad_data")
        self.assertNotEqual(code, 0)
        manifest = artifacts.read_manifest(project.outputs_dir / "bad_data")
        self.assertEqual(manifest["stages"]["audit"]["status"], "failed")
        self.assertTrue((project.outputs_dir / "bad_data" / "audit" / "issues.csv").exists())



class CliContractTests(helpers.TempDirTestCase):
    """Declared flags must be honoured; finished stages must not be overwritten."""

    def _project(self):
        project = self.project(n_train=4, n_val=2)
        weights = project.root / "weights" / "yolo11s.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"synthetic-pretrained-weights")
        return project

    def _cli(self, project, *argv):
        return cli.main(["--config", str(project.config_path), *argv])

    def test_benchmark_images_flag_reaches_the_config(self):
        project = self._project()
        seen = {}

        def fake_benchmark(config, run, weights, **kwargs):
            seen["num_images"] = config["benchmark"]["num_images"]
            helpers.make_benchmark(run, num_images=2, repeats=1)
            return {"status": "ok", "mean_ms": 9.0, "p50_ms": 9.0, "p95_ms": 9.0,
                    "fps": 111.0, "num_images": 2, "requested_images": 2,
                    "available_images": 2, "device": "cpu"}

        from uavdet import benchmark as benchmark_module
        from uavdet import training as training_module

        with mock.patch.object(training_module, "default_yolo_factory",
                               side_effect=lambda w: helpers.RecordingYOLO(w)), \
             mock.patch.object(benchmark_module, "run_benchmark", fake_benchmark):
            self.assertEqual(self._cli(project, "check", "--run-id", "flags"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "flags"), 0)
            self.assertEqual(self._cli(project, "benchmark", "--run-id", "flags",
                                       "--images", "7"), 0)
        self.assertEqual(seen["num_images"], 7, "--images must override benchmark.num_images")

    def test_completed_validation_and_benchmark_refuse_a_rerun(self):
        from uavdet import benchmark as benchmark_module
        from uavdet import evaluation as evaluation_module
        from uavdet import training as training_module

        def fake_evaluate(config, run, weights, split="val", **kwargs):
            data_yaml = kwargs.get("data_yaml") or run.manifest["dataset_view"]["data_yaml"]
            return _write_evaluation_fixture(run, project, Path(data_yaml), Path(weights))

        project = self._project()
        with mock.patch.object(training_module, "default_yolo_factory",
                               side_effect=lambda w: helpers.RecordingYOLO(w)), \
             mock.patch.object(evaluation_module, "evaluate_model", fake_evaluate):
            self.assertEqual(self._cli(project, "check", "--run-id", "replay"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "replay"), 0)
            self.assertEqual(self._cli(project, "val", "--run-id", "replay"), 0)
            # second val in the same run is refused (plan: re-validate in a new run)
            self.assertEqual(self._cli(project, "val", "--run-id", "replay"), 4)
            # ... unless the operator explicitly forces a recorded re-run
            self.assertEqual(self._cli(project, "val", "--run-id", "replay", "--force"), 0)

        with mock.patch.object(benchmark_module, "run_benchmark", lambda *a, **k: (
                helpers.make_benchmark(a[1], num_images=2, repeats=1) or
                {"status": "ok", "mean_ms": 9.0, "p50_ms": 9.0, "p95_ms": 9.0, "fps": 111.0,
                 "num_images": 2, "requested_images": 2, "available_images": 2,
                 "device": "cpu"})):
            self.assertEqual(self._cli(project, "benchmark", "--run-id", "replay"), 0)
            self.assertEqual(self._cli(project, "benchmark", "--run-id", "replay"), 4)

    def test_parent_run_id_is_recorded_for_a_revalidation_run(self):
        from uavdet import evaluation as evaluation_module
        from uavdet import training as training_module

        def fake_evaluate(config, run, weights, split="val", **kwargs):
            data_yaml = kwargs.get("data_yaml") or run.manifest["dataset_view"]["data_yaml"]
            return _write_evaluation_fixture(run, project, Path(data_yaml), Path(weights))

        project = self._project()
        with mock.patch.object(training_module, "default_yolo_factory",
                               side_effect=lambda w: helpers.RecordingYOLO(w)), \
             mock.patch.object(evaluation_module, "evaluate_model", fake_evaluate):
            self.assertEqual(self._cli(project, "check", "--run-id", "parent_src"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "parent_src"), 0)
            weights = project.outputs_dir / "parent_src" / "train" / "weights" / "best.pt"
            self.assertEqual(self._cli(project, "val", "--run-id", "reval_02",
                                       "--weights", str(weights),
                                       "--parent-run-id", "parent_src"), 0)
        manifest = artifacts.read_manifest(project.outputs_dir / "reval_02")
        self.assertEqual(manifest["parent_run_id"], "parent_src")
        self.assertNotEqual(artifacts.read_manifest(project.outputs_dir / "parent_src")
                            ["parent_run_id"], "parent_src")

    def test_pipeline_formal_gate_refuses_an_incomplete_validation(self):
        from uavdet import benchmark as benchmark_module
        from uavdet import evaluation as evaluation_module
        from uavdet import training as training_module

        project = self._project()

        def short_evaluate(config, run, weights, split="val", **kwargs):
            data_yaml = kwargs.get("data_yaml") or run.manifest["dataset_view"]["data_yaml"]
            result = _write_evaluation_fixture(run, project, Path(data_yaml), Path(weights))
            result["num_images"] = 1          # pretend only one image was evaluated
            return result

        with mock.patch.object(training_module, "default_yolo_factory",
                               side_effect=lambda w: helpers.RecordingYOLO(w)), \
             mock.patch.object(evaluation_module, "evaluate_model", short_evaluate), \
             mock.patch.object(benchmark_module, "run_benchmark", lambda *a, **k: {}):
            code = self._cli(project, "pipeline", "--run-id", "gate")
        self.assertNotEqual(code, 0)
        manifest = artifacts.read_manifest(project.outputs_dir / "gate")
        self.assertEqual(manifest["stages"]["evaluation"]["status"], "failed")
        self.assertIn("complete validation", manifest["stages"]["evaluation"]["error"])

if __name__ == "__main__":
    unittest.main()


class AuditReuseTests(helpers.TempDirTestCase):
    """A saved audit may only be reused for the data it was made from (audit F1)."""

    def _project(self):
        project = self.project(n_train=4, n_val=2)
        (project.root / "weights").mkdir(exist_ok=True)
        (project.root / "weights" / "yolo11s.pt").write_bytes(b"weights")
        return project

    def test_unchanged_data_reuses_the_audit(self):
        project = self._project()
        self.assertEqual(self._cli(project, "check", "--run-id", "reuse"), 0)
        first = artifacts.read_manifest(project.outputs_dir / "reuse")
        self.assertEqual(self._cli(project, "check", "--run-id", "reuse"), 0)
        second = artifacts.read_manifest(project.outputs_dir / "reuse")
        self.assertEqual(first["stages"]["audit"]["attempts"], 1)
        self.assertEqual(second["stages"]["audit"]["attempts"], 1)

    def test_changed_data_root_forces_a_real_reaudit(self):
        project = self._project()
        self.assertEqual(self._cli(project, "check", "--run-id", "reuse"), 0)
        other_root = self.tmp_path / "other_data"
        helpers.make_dataset(other_root, n_train=3, n_val=2)
        code = self._cli(project, "check", "--run-id", "reuse", "--data-root", str(other_root))
        self.assertNotEqual(code, 0, "a different data root must be audited, not inherited")
        manifest = artifacts.read_manifest(project.outputs_dir / "reuse")
        kinds = [event["kind"] for event in manifest.get("events") or []]
        self.assertIn("audit_reuse_rejected", kinds)
        self.assertEqual(manifest["stages"]["audit"]["status"], "failed")

    def _cli(self, project, *argv):
        return cli.main(["--config", str(project.config_path), *argv])


class StageRecordProtectionTests(helpers.TempDirTestCase):
    """A rejected command must not rewrite stage records (audit F2)."""

    def _project(self):
        project = self.project(n_train=4, n_val=2)
        (project.root / "weights").mkdir(exist_ok=True)
        (project.root / "weights" / "yolo11s.pt").write_bytes(b"weights")
        return project

    def _cli(self, project, *argv):
        return cli.main(["--config", str(project.config_path), *argv])

    def _patch(self, project):
        from uavdet import benchmark as benchmark_module
        from uavdet import evaluation as evaluation_module
        from uavdet import training as training_module

        def fake_evaluate(config, run, weights, split="val", **kwargs):
            data_yaml = kwargs.get("data_yaml") or run.manifest["dataset_view"]["data_yaml"]
            return _write_evaluation_fixture(run, project, Path(data_yaml), Path(weights))

        def fake_benchmark(config, run, weights, **kwargs):
            helpers.make_benchmark(run, num_images=2, repeats=1)
            return {"status": "ok", "mean_ms": 9.0, "p50_ms": 9.0, "p95_ms": 9.0,
                    "fps": 111.0, "num_images": 2, "requested_images": 2,
                    "available_images": 2, "device": "cpu"}

        return (mock.patch.object(training_module, "default_yolo_factory",
                                  side_effect=lambda w: helpers.RecordingYOLO(w)),
                mock.patch.object(evaluation_module, "evaluate_model", fake_evaluate),
                mock.patch.object(benchmark_module, "run_benchmark", fake_benchmark))

    def test_second_plot_same_tag_keeps_the_complete_stage(self):
        project = self._project()
        p1, p2, p3 = self._patch(project)
        with p1, p2, p3:
            self.assertEqual(self._cli(project, "check", "--run-id", "tags"), 0)
            self.assertEqual(self._cli(project, "train", "--run-id", "tags"), 0)
            self.assertEqual(self._cli(project, "val", "--run-id", "tags"), 0)
            self.assertEqual(self._cli(project, "plot", "--run-id", "tags"), 0)
        before = artifacts.read_manifest(project.outputs_dir / "tags")["stages"]["figures"]
        self.assertEqual(self._cli(project, "plot", "--run-id", "tags"), 4)
        after = artifacts.read_manifest(project.outputs_dir / "tags")["stages"]["figures"]
        self.assertEqual(after["status"], "complete", "a refused plot must not flip the stage")
        self.assertEqual(after["attempts"], before["attempts"])

    def test_unconfigured_test_split_fails_fast(self):
        project = self._project()
        p1, p2, p3 = self._patch(project)
        with p1, p2, p3:
            self.assertEqual(self._cli(project, "check", "--run-id", "no_test"), 0)
            code = self._cli(project, "val", "--run-id", "no_test", "--split", "test")
        self.assertEqual(code, 3)
        manifest = artifacts.read_manifest(project.outputs_dir / "no_test")
        self.assertEqual(manifest["stages"]["evaluation"]["status"], "pending")

    def test_continue_explains_a_completed_stage_with_a_missing_product(self):
        project = self._project()
        p1, p2, p3 = self._patch(project)
        with p1, p2, p3:
            self.assertEqual(self._cli(project, "pipeline", "--run-id", "contin"), 0)
        (project.outputs_dir / "contin" / "evaluation" / "metrics.json").unlink()
        p1, p2, p3 = self._patch(project)
        with p1, p2, p3:
            code = self._cli(project, "pipeline", "--run-id", "contin", "--continue")
        self.assertEqual(code, 4)
        manifest = artifacts.read_manifest(project.outputs_dir / "contin")
        error = manifest["stages"]["evaluation"].get("error") or ""
        self.assertEqual(manifest["stages"]["evaluation"]["status"], "complete",
                         "the stage record itself must stay as it was")
        self.assertTrue(error == "" or "metrics.json" in error)


class RealModelIntegrationTests(helpers.TempDirTestCase):
    """T07: real-model integration is opt-in via UAVDET_REAL_WEIGHTS."""

    @unittest.skipUnless(os.environ.get("UAVDET_REAL_WEIGHTS"),
                         "set UAVDET_REAL_WEIGHTS=<10-class .pt> to run the real-model check")
    def test_real_weights_predict_on_a_real_image(self):
        from uavdet.inference import Detector
        from uavdet.types import InferenceOptions

        weights = Path(os.environ["UAVDET_REAL_WEIGHTS"])
        image_path = Path(os.environ.get(
            "UAVDET_REAL_IMAGE",
            "dataset_visdrone/VisDrone2019-DET-val/images/0000271_01401_d_0000380.jpg"))
        if not image_path.is_file():
            self.skipTest(f"no real image at {image_path}")
        import cv2

        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image)
        with Detector(weights, "cpu") as detector:
            result = detector.predict(image, InferenceOptions(), image_id="real")
        self.assertEqual(len(detector.names), 10)
        self.assertGreaterEqual(result.count, 0)
        self.assertEqual((result.width, result.height), (image.shape[1], image.shape[0]))
