"""Tests for the report figure renderers (F01-F12).

Everything runs on synthetic fixtures from :mod:`tests.helpers` inside temporary
directories: these numbers are test logic only, never model-performance
evidence, and nothing is written outside the temp dir.

Coverage:

* every applicable figure renders (files really exist, PNG opens with PIL at
  ~300 dpi and is not blank, PDF/SVG non-empty);
* the plotted numbers match ``dataset_stats.json`` / ``results.csv`` /
  ``per_class.csv`` / ``confusion_matrix.npz`` exactly (planning 03 D03);
* percent conversion happens exactly once (D01: 0.42992 -> 42.992);
* a class without GT is an explicit N/A, never a 0 bar;
* missing inputs give ``missing`` records and no files at all (D02);
* records carry input hashes, style/software versions and a neutral caption;
* rendering never imports torch/ultralytics.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from uavdet.artifacts import read_json, read_jsonl, sha256_file
from uavdet.plotting import cases as cases_plots
from uavdet.plotting import dataset as dataset_plots
from uavdet.plotting import evaluation as evaluation_plots
from uavdet.plotting import training as training_plots
from uavdet.plotting.style import (
    CLASS_COLORS,
    LOSS_COLORS,
    METRIC_COLORS,
    STYLE_VERSION,
    to_percent,
)
from uavdet.report import FIGURE_META, MODULE_FIGURES
from uavdet.types import CLASS_NAMES, NUM_CLASSES, RunContext
from tests import helpers

FIGURE_MODULES = (dataset_plots, training_plots, evaluation_plots, cases_plots)
BASENAMES = {figure_id: meta[0] for figure_id, meta in FIGURE_META.items()}
VALID_CLASSES = (0, 1, 2, 3, 4, 5, 6, 8, 9)   # awning-tricycle (7) has no GT


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_project(tmp_path: Path, **kwargs: Any) -> helpers.TempProject:
    """Project fixture: ``configs/project.yaml`` + ``data/`` in a temp directory."""
    return helpers.make_project(tmp_path, **kwargs)


def render_all(run: RunContext, out_dir: Path) -> list[dict]:
    records: list[dict] = []
    for module in FIGURE_MODULES:
        records.extend(module.render(run, out_dir))
    return records


def by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(record["figure_id"]): record for record in records}


def source_npz(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    paths = [path for path in record.get("source_data_paths") or [] if path.endswith(".npz")]
    if not paths:
        raise AssertionError(f"{record['figure_id']} has no source_data npz")
    with np.load(paths[0]) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def source_csv(record: Mapping[str, Any]) -> list[dict[str, str]]:
    paths = [path for path in record.get("source_data_paths") or [] if path.endswith(".csv")]
    if not paths:
        raise AssertionError(f"{record['figure_id']} has no source_data csv")
    return helpers.read_csv_rows(paths[0])


def source_meta(record: Mapping[str, Any]) -> dict[str, Any]:
    paths = [path for path in record.get("source_data_paths") or [] if path.endswith(".json")]
    if not paths:
        raise AssertionError(f"{record['figure_id']} has no source_data meta json")
    return read_json(paths[0])


def png_size(path: str | Path) -> tuple[int, int, float, int]:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        dpi = float(image.info.get("dpi", (0, 0))[0])
        colours = len(image.convert("RGB").getcolors(maxcolors=100000) or [])
        return image.size[0], image.size[1], dpi, colours


def assert_no_blank_png(testcase: unittest.TestCase, path: str | Path) -> None:
    width, height, dpi, colours = png_size(path)
    testcase.assertGreaterEqual(width, 600, f"{path} is implausibly narrow")
    testcase.assertGreaterEqual(height, 300, f"{path} is implausibly short")
    testcase.assertAlmostEqual(dpi, 300.0, delta=1.0, msg=f"{path} is not 300 dpi")
    testcase.assertGreater(colours, 1, f"{path} looks blank (only one colour)")
    testcase.assertGreater(Path(path).stat().st_size, 4096, f"{path} is suspiciously small")


def write_results_csv(path: Path, rows: Iterable[Sequence[Any]]) -> Path:
    """Write a schema-correct results.csv with hand-chosen values."""
    target = Path(path)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(helpers.RESULTS_COLUMNS))
        writer.writerows(rows)
    return target


class RenderedRun(unittest.TestCase):
    """One fake run rendered once for the whole class (fast, read-only)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmp.name)
        cls.project = make_project(cls.tmp_path / "proj")
        cls.run_ctx = helpers.make_fake_run(cls.project, valid_classes=VALID_CLASSES)
        cls.out_dir = cls.tmp_path / "figures"
        cls.modules_before = set(sys.modules)
        cls.records = render_all(cls.run_ctx, cls.out_dir)
        cls.by_id = by_id(cls.records)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def record(self, figure_id: str) -> Mapping[str, Any]:
        self.assertIn(figure_id, self.by_id)
        return self.by_id[figure_id]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
class TestRenderAllFigures(RenderedRun):
    def test_one_record_per_figure(self) -> None:
        self.assertEqual(len(self.records), len(FIGURE_META))
        self.assertEqual(set(self.by_id), set(FIGURE_META))

    def test_every_figure_is_ok_and_files_exist(self) -> None:
        for figure_id, record in self.by_id.items():
            with self.subTest(figure_id=figure_id):
                self.assertEqual(record["data_status"], "ok",
                                 f"{figure_id}: {record.get('reason') or record.get('warnings')}")
                self.assertEqual(record["figure_tag"], "figures")
                self.assertEqual(record["title"], FIGURE_META[figure_id][1])
                self.assertEqual(record["manuscript_section"], FIGURE_META[figure_id][2])
                paths = record["output_paths"]
                self.assertEqual(len(paths), 3)
                self.assertTrue(any(path.endswith(".png") for path in paths))
                self.assertTrue(any(path.endswith(".pdf") for path in paths))
                self.assertTrue(any(path.endswith(".svg") for path in paths))
                for path in paths:
                    self.assertTrue(Path(path).is_file(), path)
                    self.assertGreater(Path(path).stat().st_size, 0, path)
                for path in paths:
                    if path.endswith(".png"):
                        self.assertEqual(Path(path).name, f"{BASENAMES[figure_id]}.png")
                        assert_no_blank_png(self, path)

    def test_module_basenames_match_report_meta(self) -> None:
        for module in FIGURE_MODULES:
            for figure_id, entry in module.FIGURE_META.items():
                with self.subTest(figure_id=figure_id):
                    self.assertEqual(entry, FIGURE_META[figure_id])
            self.assertEqual(
                tuple(module.FIGURE_META),
                tuple(MODULE_FIGURES[module.__name__.rsplit(".", 1)[-1]]),
            )

    def test_records_carry_full_provenance(self) -> None:
        for figure_id, record in self.by_id.items():
            with self.subTest(figure_id=figure_id):
                self.assertTrue(record["input_paths"])
                self.assertTrue(record["input_sha256"])
                for rel_path, digest in record["input_sha256"].items():
                    if rel_path.startswith("external/"):
                        # portable label: resolve by file name inside the dataset
                        name = rel_path.split("/", 1)[1]
                        matches = list(self.project.dataset["images"]["val"].glob(name))
                        if not matches:
                            matches = list(self.project.data_root.rglob(name))
                        self.assertTrue(matches, f"{figure_id}: {rel_path}")
                        absolute = matches[0]
                    else:
                        absolute = self.run_ctx.run_dir / rel_path
                        self.assertTrue(absolute.is_file(), f"{figure_id}: {rel_path}")
                    self.assertEqual(sha256_file(absolute), digest)
                for key in ("variables", "units", "filters", "transforms", "normalization"):
                    self.assertIsInstance(record[key], list if key != "units" else dict)
                self.assertTrue(record["variables"], figure_id)
                self.assertTrue(record["units"], figure_id)
                self.assertTrue(record["filters"], figure_id)
                self.assertTrue(record["normalization"], figure_id)
                self.assertEqual(record["class_order"], list(CLASS_NAMES))
                self.assertTrue(record["sample_count"], figure_id)
                self.assertTrue(all(value >= 0 for value in record["sample_count"].values()))
                self.assertEqual(record["style_version"], STYLE_VERSION)
                versions = record["software_versions"]
                for key in ("uavdet", "python", "matplotlib", "numpy", "style"):
                    self.assertIn(key, versions)
                self.assertEqual(record["scientific_review"], "pending")
                self.assertEqual(record["visual_review"], "pending")
                self.assertTrue(record["notes"].strip())

    def test_captions_stay_neutral(self) -> None:
        banned = ("significant", "significantly", "best", "outperform", "optimal",
                  "superior", "state-of-the-art")
        for figure_id, record in self.by_id.items():
            with self.subTest(figure_id=figure_id):
                text = str(record["notes"]).lower()
                for word in banned:
                    self.assertNotIn(word, text, f"{figure_id} caption claims '{word}'")

    def test_source_data_is_finite_and_parses(self) -> None:
        for figure_id, record in self.by_id.items():
            with self.subTest(figure_id=figure_id):
                self.assertTrue(record["source_data_paths"])
                for path in record["source_data_paths"]:
                    self.assertTrue(Path(path).is_file(), path)
                arrays = source_npz(record)
                self.assertTrue(arrays)
                for key, values in arrays.items():
                    if np.issubdtype(values.dtype, np.floating):
                        self.assertTrue(np.all(np.isfinite(values)),
                                        f"{figure_id}:{key} contains NaN/inf")
                meta = source_meta(record)
                self.assertIsInstance(meta, dict)
                self.assertIn("units", meta)

    def test_percent_conversion_happens_once(self) -> None:
        """D01: 0.42992 must become 42.992, never 0.42992 % or 4299.2 %."""
        frame = helpers.read_csv_rows(self.run_ctx.train_dir / "results.csv")
        stored = np.asarray([float(row["metrics/mAP50(B)"]) for row in frame])
        arrays = source_npz(self.record("F05"))
        self.assertEqual(arrays["map50_0to1"].shape, stored.shape)
        np.testing.assert_allclose(arrays["map50_0to1"], stored, rtol=0, atol=0)
        np.testing.assert_allclose(arrays["map50_percent"], to_percent(stored), rtol=0, atol=1e-9)
        self.assertAlmostEqual(float(arrays["map50_percent"][-1]), stored[-1] * 100.0, places=9)
        self.assertFalse(np.allclose(arrays["map50_percent"], stored))          # not 0-1
        self.assertFalse(np.allclose(arrays["map50_percent"], stored * 10000))  # not x10000
        meta = source_meta(self.record("F05"))
        self.assertIn("100", meta["display_transform"])
        csv_rows = source_csv(self.record("F05"))
        self.assertAlmostEqual(float(csv_rows[-1]["map50_percent"]), stored[-1] * 100.0, places=9)
        self.assertAlmostEqual(float(csv_rows[-1]["map50_0to1"]), stored[-1], places=12)

    def test_figure_colour_semantics(self) -> None:
        self.assertEqual(METRIC_COLORS["map50"], "#0072B2")
        self.assertEqual(METRIC_COLORS["map50_95"], "#009E73")
        self.assertEqual(METRIC_COLORS["precision"], "#E69F00")
        self.assertEqual(METRIC_COLORS["recall"], "#D55E00")
        self.assertEqual(LOSS_COLORS["box"], "#0072B2")
        self.assertEqual(LOSS_COLORS["cls"], "#009E73")
        self.assertEqual(LOSS_COLORS["dfl"], "#E69F00")
        self.assertEqual(len(CLASS_COLORS), NUM_CLASSES)
        meta = source_meta(self.record("F04"))
        self.assertEqual(meta["loss_colors"]["box"], LOSS_COLORS["box"])
        self.assertEqual(meta["reference_lines"], {"train": "solid", "val": "dashed"})
        m05 = source_meta(self.record("F05"))
        self.assertEqual(m05["metric_colors"]["map50"], METRIC_COLORS["map50"])

    def test_dataset_figures_match_audit(self) -> None:
        stats = read_json(self.run_ctx.audit_dir / "dataset_stats.json")
        arrays = source_npz(self.record("F01"))
        for split in ("train", "val"):
            expected = np.asarray(stats["splits"][split]["per_class_list"], dtype=np.int64)
            np.testing.assert_array_equal(arrays[f"{split}_gt_instances"], expected)
        # every instance lands in exactly one area bin, per split
        areas = source_npz(self.record("F02"))
        for split in ("train", "val"):
            histogram = stats["splits"][split]["area_histogram"]
            np.testing.assert_array_equal(areas[f"{split}_counts"],
                                          np.asarray(histogram["counts"], dtype=np.int64))
            np.testing.assert_allclose(areas["bin_edges"], np.asarray(histogram["edges"]))
        per_image = source_npz(self.record("F03"))
        np.testing.assert_array_equal(
            per_image["train_objects_per_image"],
            np.asarray(stats["splits"]["train"]["per_image_objects"], dtype=np.int64))
        self.assertEqual(int(per_image["train_objects_per_image"].size),
                         int(stats["splits"]["train"]["num_images"]))
        self.assertEqual(int(per_image["val_objects_per_image"].size),
                         int(stats["splits"]["val"]["num_images"]))
        highest = int(max(per_image["train_objects_per_image"].max(),
                          per_image["val_objects_per_image"].max()))
        self.assertEqual(len(per_image["bin_edges"]), min(highest, 25) + 1)

    def test_training_figures_match_results_csv(self) -> None:
        """D03: three epochs compared value-by-value with results.csv."""
        frame = {int(row["epoch"]): row for row in
                 helpers.read_csv_rows(self.run_ctx.train_dir / "results.csv")}
        epochs = (3, 7, 12)
        loss = source_npz(self.record("F04"))
        for epoch in epochs:
            index = int(np.where(loss["epoch"] == epoch)[0][0])
            row = frame[epoch]
            self.assertAlmostEqual(float(loss["train_box_loss"][index]),
                                   float(row["train/box_loss"]), places=9)
            self.assertAlmostEqual(float(loss["val_box_loss"][index]),
                                   float(row["val/box_loss"]), places=9)
            self.assertAlmostEqual(float(loss["val_dfl_loss"][index]),
                                   float(row["val/dfl_loss"]), places=9)
        map_arrays = source_npz(self.record("F05"))
        pr_arrays = source_npz(self.record("F06"))
        for epoch in epochs:
            index = int(np.where(map_arrays["epoch"] == epoch)[0][0])
            row = frame[epoch]
            self.assertAlmostEqual(float(map_arrays["map50_95_0to1"][index]),
                                   float(row["metrics/mAP50-95(B)"]), places=9)
            self.assertAlmostEqual(float(pr_arrays["precision_0to1"][index]),
                                   float(row["metrics/precision(B)"]), places=9)
            self.assertAlmostEqual(float(pr_arrays["recall_percent"][index]),
                                   float(row["metrics/recall(B)"]) * 100.0, places=6)

    def test_per_class_ap_matches_csv(self) -> None:
        rows = {int(row["class_id"]): row
                for row in helpers.read_csv_rows(self.run_ctx.evaluation_dir / "per_class.csv")}
        arrays = source_npz(self.record("F07"))
        for class_id in (0, 4, 9):
            row = rows[class_id]
            self.assertTrue(arrays["valid_mask"][class_id])
            self.assertAlmostEqual(float(arrays["ap50_0to1"][class_id]),
                                   float(row["ap50"]), places=12)
            self.assertAlmostEqual(float(arrays["ap50_95_percent"][class_id]),
                                   float(row["ap50_95"]) * 100.0, places=6)
            self.assertEqual(int(arrays["num_gt"][class_id]), int(row["num_gt"]))
        # the N/A class is not silently zeroed anywhere
        na_index = 7
        self.assertFalse(bool(arrays["valid_mask"][na_index]))
        self.assertEqual(rows[na_index]["ap50"], "N/A")
        csv_rows = source_csv(self.record("F07"))
        self.assertEqual(csv_rows[na_index]["ap50_0to1"], "N/A")
        self.assertEqual(csv_rows[na_index]["valid"], "False")

    def test_confusion_matrix_orientation_and_normalisation(self) -> None:
        with np.load(self.run_ctx.evaluation_dir / "confusion_matrix.npz") as loaded:
            raw = np.asarray(loaded["matrix"], dtype=np.int64)
        arrays = source_npz(self.record("F08"))
        np.testing.assert_array_equal(arrays["counts"], raw)
        # (row=true, column=pred): pedestrian predicted as people
        self.assertEqual(int(arrays["counts"][0, 1]), 3)
        # background row predicted as pedestrian
        self.assertEqual(int(arrays["counts"][NUM_CLASSES, 0]), 2)
        # bicycle missed into background
        self.assertEqual(int(arrays["counts"][2, NUM_CLASSES]), 1)
        np.testing.assert_array_equal(arrays["row_sums"], raw.sum(axis=1))
        expected = (raw[0] / raw[0].sum())
        np.testing.assert_allclose(arrays["normalised"][0], expected)
        self.assertAlmostEqual(float(arrays["normalised"][0, 1]), 3.0 / raw[0].sum(), places=12)
        meta = source_meta(self.record("F08"))
        self.assertEqual(meta["true_axis"], "row")
        self.assertEqual(meta["pred_axis"], "column")
        self.assertEqual(meta["background_index"], NUM_CLASSES)
        self.assertEqual(meta["labels"][NUM_CLASSES], "background")
        self.assertEqual(meta["labels"][:NUM_CLASSES], list(CLASS_NAMES))
        self.assertEqual(meta["labels"].count("background"), 1)
        # the caption must state the conf / IoU this matrix was built with
        notes = str(self.record("F08")["notes"])
        self.assertIn("conf=0.001", notes)
        self.assertIn("IoU=0.7", notes)

    def test_curves_match_npz_and_mask(self) -> None:
        with np.load(self.run_ctx.evaluation_dir / "curves.npz") as loaded:
            stored = {key: np.asarray(loaded[key]) for key in loaded.files}
        pr = source_npz(self.record("F09"))
        np.testing.assert_array_equal(pr["recall"], stored["recall"])
        np.testing.assert_array_equal(pr["per_class_precision"], stored["pr_precision"][:NUM_CLASSES])
        np.testing.assert_array_equal(pr["mean_precision"], stored["pr_precision"][NUM_CLASSES])
        f1 = source_npz(self.record("F10"))
        np.testing.assert_array_equal(f1["mean_f1"], stored["f1"][NUM_CLASSES])
        self.assertEqual(int(f1["mean_peak_index"][0]), int(np.argmax(stored["f1"][NUM_CLASSES])))
        meta = source_meta(self.record("F10"))
        self.assertIn("validation set only", meta["mean_peak"]["scope"])
        # PR values are never re-scaled to percent
        self.assertLessEqual(float(np.max(pr["per_class_precision"])), 1.0)
        self.assertEqual(pr["per_class_precision"].shape[1], stored["recall"].size)

    def test_qualitative_cases_use_recorded_predictions(self) -> None:
        cases_payload = read_json(self.run_ctx.evaluation_dir / "cases.json")
        expected = len(cases_payload["successful"])
        f11 = self.record("F11")
        self.assertEqual(f11["sample_count"]["cases_rendered"], expected)
        rows = source_csv(f11)
        self.assertEqual([row["image_id"] for row in rows],
                         [entry["image_id"] for entry in cases_payload["successful"]])
        meta = source_meta(f11)
        self.assertAlmostEqual(float(meta["display_threshold"]), 0.25)
        val_images = {image.name for image in self.project.dataset["images"]["val"].glob("*.jpg")}
        for row in rows:
            self.assertIn(Path(row["image_path"]).name, val_images)
            # boxes are original-image pixels inside the frame
            self.assertLessEqual(float(row["crop_x2"]), float(row["width"]))
            self.assertLessEqual(float(row["crop_y2"]), float(row["height"]))
        f12 = self.record("F12")
        failure_rows = source_csv(f12)
        self.assertEqual(len(failure_rows), len(cases_payload["failure"]))
        reasons = {entry["image_id"]: entry["reason"]
                   for entry in cases_payload["failure"]}
        for row in failure_rows:
            self.assertEqual(row["reason"], reasons[row["image_id"]])
        self.assertIn("possible", str(f12["notes"]).lower())
        for figure_id in ("F11", "F12"):
            for warning in self.record(figure_id)["warnings"]:
                self.assertNotIn("could not be resolved", warning)

    def test_rendering_imports_no_torch_or_ultralytics(self) -> None:
        for module_name in ("torch", "ultralytics", "cv2"):
            self.assertNotIn(module_name, set(sys.modules) - self.modules_before)

    def test_importing_renderers_is_light(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r);"
            "import uavdet.plotting.dataset, uavdet.plotting.training,"
            " uavdet.plotting.evaluation, uavdet.plotting.cases;"
            "bad=[m for m in ('torch','ultralytics','cv2','PyQt6') if m in sys.modules];"
            "print(','.join(bad))"
        ) % str(SRC_DIR)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                check=False, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "", "renderers imported heavy modules")


# ---------------------------------------------------------------------------
# D01 — explicit 0.42992 -> 42.992 check with a hand-written results.csv
# ---------------------------------------------------------------------------
class TestPercentConversion(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        project = make_project(self.tmp_path / "proj")
        self.run_ctx = helpers.make_fake_run(project, valid_classes=VALID_CLASSES)
        rows = [
            [1, 45.0, 2.0, 1.5, 1.3, 0.45, 0.35, 0.20000, 0.10000,
             1.9, 1.4, 1.2, 0.01, 0.01, 0.01],
            [2, 90.0, 1.9, 1.4, 1.2, 0.50, 0.40, 0.25000, 0.12000,
             1.8, 1.3, 1.1, 0.009, 0.009, 0.009],
            [3, 135.0, 1.8, 1.3, 1.1, 0.60, 0.50, 0.42992, 0.25111,
             1.7, 1.2, 1.0, 0.008, 0.008, 0.008],
        ]
        write_results_csv(self.run_ctx.train_dir / "results.csv", rows)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_map50_percent_is_multiplied_exactly_once(self) -> None:
        out_dir = self.tmp_path / "figures"
        records = by_id(training_plots.render(self.run_ctx, out_dir))
        record = records["F05"]
        self.assertEqual(record["data_status"], "ok")
        arrays = source_npz(record)
        self.assertEqual(float(arrays["map50_0to1"][-1]), 0.42992)
        self.assertAlmostEqual(float(arrays["map50_percent"][-1]), 42.992, places=9)
        self.assertAlmostEqual(float(arrays["map50_95_percent"][-1]), 25.111, places=9)
        drawn = np.concatenate([arrays["map50_percent"], arrays["map50_95_percent"]])
        for wrong in (0.42992, 4.2992, 429.92, 4299.2):
            self.assertFalse(np.any(np.isclose(drawn, wrong)), f"drawn {wrong}")

    def test_map50_is_drawn_in_percent(self) -> None:
        import matplotlib.axes

        captured: list[np.ndarray] = []
        original = matplotlib.axes.Axes.plot

        def spy(self, *args, **kwargs):
            for value in args[1:2]:
                try:
                    captured.append(np.asarray(value, dtype=float))
                except (TypeError, ValueError):
                    pass
            return original(self, *args, **kwargs)

        out_dir = self.tmp_path / "figures_plot"
        matplotlib.axes.Axes.plot = spy
        try:
            training_plots.render(self.run_ctx, out_dir)
        finally:
            matplotlib.axes.Axes.plot = original
        arrays = [array for array in captured if array.size == 3]
        flattened = np.array([value for array in arrays for value in array.tolist()])
        self.assertTrue(np.any(np.isclose(flattened, 42.992)),
                        f"42.992 was not drawn; saw {sorted(set(np.round(flattened, 4)))}")


# ---------------------------------------------------------------------------
# D01/planning-02: N/A class is never drawn as a zero bar
# ---------------------------------------------------------------------------
class TestMissingGroundTruthClass(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        project = make_project(self.tmp_path / "proj")
        self.run_ctx = helpers.make_fake_run(project, valid_classes=VALID_CLASSES)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_na_class_gets_marker_and_zero_bar_is_absent(self) -> None:
        import matplotlib.axes

        widths: list[float] = []
        texts: list[str] = []
        original_barh = matplotlib.axes.Axes.barh
        original_text = matplotlib.axes.Axes.text

        def spy_barh(self, y, width, *args, **kwargs):
            widths.append(float(width))
            return original_barh(self, y, width, *args, **kwargs)

        def spy_text(self, x, y, text, *args, **kwargs):
            texts.append(str(text))
            return original_text(self, x, y, text, *args, **kwargs)

        matplotlib.axes.Axes.barh = spy_barh
        matplotlib.axes.Axes.text = spy_text
        try:
            records = by_id(evaluation_plots.render(self.run_ctx, self.tmp_path / "figures"))
        finally:
            matplotlib.axes.Axes.barh = original_barh
            matplotlib.axes.Axes.text = original_text

        record = records["F07"]
        self.assertEqual(record["data_status"], "ok")
        self.assertTrue(widths)
        self.assertNotIn(0.0, widths, "a class without GT was drawn as a zero bar")
        self.assertIn("N/A", texts)
        self.assertEqual(record["sample_count"]["classes_without_gt"], 1)
        arrays = source_npz(record)
        self.assertEqual(int(arrays["valid_mask"].sum()), len(VALID_CLASSES))
        self.assertFalse(bool(arrays["valid_mask"][7]))
        # only the classes with GT contribute bars (2 bars each)
        self.assertEqual(len(widths), 2 * len(VALID_CLASSES))


# ---------------------------------------------------------------------------
# D02 — missing inputs produce missing records and no files
# ---------------------------------------------------------------------------
class TestMissingInputs(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        project = make_project(self.tmp_path / "proj")
        self.run_ctx = helpers.make_fake_run(project, valid_classes=VALID_CLASSES)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_missing(self, records: Sequence[Mapping[str, Any]], figure_ids: Sequence[str],
                        out_dir: Path) -> None:
        indexed = by_id(records)
        for figure_id in figure_ids:
            with self.subTest(figure_id=figure_id):
                record = indexed[figure_id]
                self.assertEqual(record["data_status"], "missing")
                self.assertTrue(record["reason"])
                self.assertTrue(record.get("missing_inputs"), record.get("missing_inputs"))
                self.assertFalse(list(out_dir.glob(f"{BASENAMES[figure_id]}.*")),
                                 f"{figure_id} left a file behind despite missing inputs")

    def test_missing_results_csv(self) -> None:
        (self.run_ctx.train_dir / "results.csv").unlink()
        out_dir = self.tmp_path / "figures"
        records = training_plots.render(self.run_ctx, out_dir)
        self._assert_missing(records, ("F04", "F05", "F06"), out_dir)
        for record in records:
            self.assertIn("results.csv", record["reason"])
        self.assertFalse(list((out_dir / "source_data").glob("loss_curves*"))
                         if (out_dir / "source_data").exists() else [])

    def test_missing_curves_npz(self) -> None:
        (self.run_ctx.evaluation_dir / "curves.npz").unlink()
        out_dir = self.tmp_path / "figures"
        records = evaluation_plots.render(self.run_ctx, out_dir)
        self._assert_missing(records, ("F09", "F10"), out_dir)
        indexed = by_id(records)
        self.assertEqual(indexed["F07"]["data_status"], "ok")
        self.assertEqual(indexed["F08"]["data_status"], "ok")

    def test_missing_confusion_matrix(self) -> None:
        (self.run_ctx.evaluation_dir / "confusion_matrix.npz").unlink()
        out_dir = self.tmp_path / "figures"
        records = evaluation_plots.render(self.run_ctx, out_dir)
        self._assert_missing(records, ("F08",), out_dir)

    def test_missing_predictions_jsonl(self) -> None:
        (self.run_ctx.evaluation_dir / "predictions.jsonl").unlink()
        out_dir = self.tmp_path / "figures"
        records = cases_plots.render(self.run_ctx, out_dir)
        self._assert_missing(records, ("F11", "F12"), out_dir)

    def test_missing_dataset_stats(self) -> None:
        (self.run_ctx.audit_dir / "dataset_stats.json").unlink()
        out_dir = self.tmp_path / "figures"
        records = dataset_plots.render(self.run_ctx, out_dir)
        self._assert_missing(records, ("F01", "F02", "F03"), out_dir)

    def test_no_successful_case_is_reported_not_invented(self) -> None:
        cases_path = self.run_ctx.evaluation_dir / "cases.json"
        payload = read_json(cases_path)
        payload["successful"] = []
        cases_path.write_text(json.dumps(payload), encoding="utf-8")
        out_dir = self.tmp_path / "figures"
        records = by_id(cases_plots.render(self.run_ctx, out_dir))
        self.assertEqual(records["F11"]["data_status"], "missing")
        self.assertIn("no image meets", records["F11"]["reason"])
        self.assertFalse(list(out_dir.glob("successful_cases.*")))
        self.assertEqual(records["F12"]["data_status"], "ok")


# ---------------------------------------------------------------------------
# image resolution: source -> data_view_root/source_rel fallback
# ---------------------------------------------------------------------------
class TestCaseImageResolution(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.project = make_project(self.tmp_path / "proj")
        self.run_ctx = helpers.make_fake_run(self.project, valid_classes=VALID_CLASSES)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_falls_back_to_data_view_root(self) -> None:
        import shutil

        view_root = self.tmp_path / "view"
        images = sorted(self.project.dataset["images"]["val"].glob("*.jpg"))
        for image in images:
            target = view_root / "images" / "val" / image.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
        metrics_path = self.run_ctx.evaluation_dir / "metrics.json"
        metrics = read_json(metrics_path)
        metrics["data_view_root"] = str(view_root)
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        predictions_path = self.run_ctx.evaluation_dir / "predictions.jsonl"
        rows = read_jsonl(predictions_path)
        for row in rows:
            row["source"] = f"/nonexistent/elsewhere/{row['image_id']}.jpg"
            row["source_rel"] = f"images/val/{Path(row['source_rel']).name}"
        predictions_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8")

        out_dir = self.tmp_path / "figures"
        records = by_id(cases_plots.render(self.run_ctx, out_dir))
        self.assertEqual(records["F11"]["data_status"], "ok",
                         records["F11"].get("reason") or records["F11"]["warnings"])
        for row in source_csv(records["F11"]):
            name = Path(row["image_path"]).name
            self.assertEqual(Path(row["image_path"]).as_posix(), name)
            self.assertTrue((view_root / "images" / "val" / name).is_file(), name)


# ---------------------------------------------------------------------------
# crowded scenes (real VisDrone has 40-90 GT objects per image)
# ---------------------------------------------------------------------------
class TestCrowdedCaseLayout(unittest.TestCase):
    """A dense scene must stay readable: cropped panels, no per-box labels."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.project = make_project(self.tmp_path / "proj")
        self.run_ctx = helpers.make_fake_run(self.project, valid_classes=VALID_CLASSES)
        self.image = sorted(self.project.dataset["images"]["val"].glob("*.jpg"))[0]
        from PIL import Image

        with Image.open(self.image) as handle:
            self.width, self.height = handle.size
        self.gt_objects = self._dense_objects(40)
        self._rewrite_products()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dense_objects(self, count: int) -> list[dict[str, Any]]:
        objects = []
        columns = 8
        for index in range(count):
            row, column = divmod(index, columns)
            x1 = 1.0 + column * 7.5
            y1 = 1.0 + row * 8.0
            objects.append({
                "class_id": index % 4,
                "class_name": CLASS_NAMES[index % 4],
                "xyxy": [x1, y1, x1 + 5.0, y1 + 6.0],
            })
        return objects

    def _rewrite_products(self) -> None:
        evaluation = self.run_ctx.evaluation_dir
        image_id = self.image.stem
        row_base = {
            "image_id": image_id, "source": str(self.image),
            "source_rel": f"images/val/{self.image.name}",
            "width": self.width, "height": self.height,
        }
        ground_truth = dict(row_base, num_objects=len(self.gt_objects),
                            objects=self.gt_objects)
        detections = [
            {"class_id": obj["class_id"], "class_name": obj["class_name"],
             "confidence": 0.9 - 0.01 * index,
             "xyxy": [value + 0.5 for value in obj["xyxy"]]}
            for index, obj in enumerate(self.gt_objects[:20])
        ]
        prediction = dict(row_base, num_detections=len(detections),
                          detections=detections, timing={"total_ms": 12.0})
        (evaluation / "ground_truth.jsonl").write_text(
            json.dumps(ground_truth) + "\n", encoding="utf-8")
        (evaluation / "predictions.jsonl").write_text(
            json.dumps(prediction) + "\n", encoding="utf-8")
        payload = read_json(evaluation / "cases.json")
        payload["failure"] = [{
            "image_id": image_id, "reason": "dense scene", "num_gt": len(self.gt_objects),
            "tp": 20, "fp": 0, "fn": len(self.gt_objects) - 20,
            "failure_kinds": ["false_negative"],
        }]
        payload["successful"] = []
        (evaluation / "cases.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_dense_case_is_cropped_and_unlabelled(self) -> None:
        import matplotlib.axes
        import matplotlib.figure

        texts: list[str] = []
        legends: list[int] = []
        original_text = matplotlib.axes.Axes.text
        original_legend = matplotlib.figure.Figure.legend

        def spy_text(self, x, y, text, *args, **kwargs):
            texts.append(str(text))
            return original_text(self, x, y, text, *args, **kwargs)

        def spy_legend(self, *args, **kwargs):
            legends.append(1)
            return original_legend(self, *args, **kwargs)

        matplotlib.axes.Axes.text = spy_text
        matplotlib.figure.Figure.legend = spy_legend
        try:
            records = by_id(cases_plots.render(self.run_ctx, self.tmp_path / "figures"))
        finally:
            matplotlib.axes.Axes.text = original_text
            matplotlib.figure.Figure.legend = original_legend

        record = records["F12"]
        self.assertEqual(record["data_status"], "ok",
                         record.get("reason") or record["warnings"])
        # no per-box class/confidence labels anywhere on the photo panels
        for text in texts:
            self.assertNotIn(text, CLASS_NAMES)
            self.assertFalse(text[:1].isdigit() and "." in text,
                             f"a numeric box label was drawn: {text!r}")
        # the class-colour key must exist instead
        self.assertTrue(legends, "the class-colour legend was not drawn")

        rows = source_csv(record)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        crop_width = float(row["crop_x2"]) - float(row["crop_x1"])
        crop_height = float(row["crop_y2"]) - float(row["crop_y1"])
        self.assertLessEqual(crop_width, 0.35 * self.width)
        self.assertLessEqual(crop_height, 0.35 * self.height)
        self.assertLess(crop_width * crop_height, 0.5 * self.width * self.height)
        # the crop must contain annotation so the panel is not empty
        x1, y1 = float(row["crop_x1"]), float(row["crop_y1"])
        x2, y2 = float(row["crop_x2"]), float(row["crop_y2"])
        inside = [obj for obj in self.gt_objects
                  if x1 <= (obj["xyxy"][0] + obj["xyxy"][2]) / 2 <= x2
                  and y1 <= (obj["xyxy"][1] + obj["xyxy"][3]) / 2 <= y2]
        self.assertTrue(inside, "the crop contains no ground-truth box")
        meta = source_meta(record)
        self.assertIn("densest window", meta["crop_rule"])
        self.assertIn("no per-box text", meta["labels"])


# ---------------------------------------------------------------------------
# real zeros versus N/A in F07
# ---------------------------------------------------------------------------
class TestPerClassZeroVersusNa(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        project = make_project(self.tmp_path / "proj")
        self.run_ctx = helpers.make_fake_run(project, valid_classes=VALID_CLASSES)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_zero_ap_is_drawn_and_zero_gt_class_is_na(self) -> None:
        import matplotlib.axes

        # class 3: real 0 % AP with real GT. class 5: num_gt=0 but numeric AP.
        path = self.run_ctx.evaluation_dir / "per_class.csv"
        rows = helpers.read_csv_rows(path)
        for row in rows:
            if int(row["class_id"]) == 3:
                row["ap50"] = "0.000000"
                row["ap50_95"] = "0.000000"
            if int(row["class_id"]) == 5:
                row["num_gt"] = "0"
                row["ap50"] = "0.500000"
                row["ap50_95"] = "0.400000"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        widths: list[float] = []
        texts: list[str] = []
        original_barh = matplotlib.axes.Axes.barh
        original_text = matplotlib.axes.Axes.text

        def spy_barh(self, y, width, *args, **kwargs):
            widths.append(float(width))
            return original_barh(self, y, width, *args, **kwargs)

        def spy_text(self, x, y, text, *args, **kwargs):
            texts.append(str(text))
            return original_text(self, x, y, text, *args, **kwargs)

        matplotlib.axes.Axes.barh = spy_barh
        matplotlib.axes.Axes.text = spy_text
        try:
            records = by_id(evaluation_plots.render(self.run_ctx, self.tmp_path / "figures"))
        finally:
            matplotlib.axes.Axes.barh = original_barh
            matplotlib.axes.Axes.text = original_text

        arrays = source_npz(records["F07"])
        self.assertTrue(bool(arrays["valid_mask"][3]), "a real 0 % AP was treated as N/A")
        self.assertEqual(float(arrays["ap50_0to1"][3]), 0.0)
        self.assertIn(0.0, widths, "the real 0 % bar was not drawn")
        self.assertFalse(bool(arrays["valid_mask"][5]), "a zero-GT class was treated as numeric AP")
        self.assertIn("N/A", texts)
        self.assertTrue(any("num_gt=0" in warning for warning in records["F07"]["warnings"]))
        csv_rows = source_csv(records["F07"])
        self.assertEqual(csv_rows[5]["ap50_0to1"], "N/A")
        self.assertEqual(float(csv_rows[3]["ap50_0to1"]), 0.0)


# ---------------------------------------------------------------------------
# custom figure config (formats/dpi/width) is honoured
# ---------------------------------------------------------------------------
class TestFigureConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        project = make_project(self.tmp_path / "proj",
                               overrides={"figures.dpi": 200, "figures.figure_width": 6.0})
        self.run_ctx = helpers.make_fake_run(project, valid_classes=VALID_CLASSES)
        self.run_ctx.config["figures"]["formats"] = ["png", "pdf"]
        self.run_ctx.config["figures"]["dpi"] = 200

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_config_formats_and_dpi_are_used(self) -> None:
        out_dir = self.tmp_path / "figures"
        records = by_id(dataset_plots.render(self.run_ctx, out_dir))
        record = records["F01"]
        self.assertEqual(record["data_status"], "ok")
        suffixes = sorted(Path(path).suffix for path in record["output_paths"])
        self.assertEqual(suffixes, [".pdf", ".png"])
        png = [path for path in record["output_paths"] if path.endswith(".png")][0]
        _width, _height, dpi, _colours = png_size(png)
        self.assertAlmostEqual(dpi, 200.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()


class EnglishOnlyFigureOutputTests(unittest.TestCase):
    """The plotting code and every figure-facing string must stay English."""

    HAN = __import__("re").compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

    def test_plotting_source_has_no_chinese(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        targets = sorted((root / "src" / "uavdet" / "plotting").glob("*.py")) + [
            root / "src" / "uavdet" / "report.py",
            root / "src" / "uavdet" / "plotting" / "assets" / "sci-report.mplstyle",
        ]
        for path in targets:
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertIsNone(self.HAN.search(text), f"non-English text in {path}")

    def test_rendered_records_and_index_have_no_chinese_even_in_a_cjk_project_path(self):
        import pathlib
        import re
        import tempfile

        from tests import helpers
        from uavdet import artifacts, report

        han = self.HAN
        with tempfile.TemporaryDirectory() as tmp:
            # The project lives under a directory whose name contains Chinese:
            # the index must still come out English-only.
            project = helpers.make_project(pathlib.Path(tmp) / "课程资料" / "proj")
            run = helpers.make_fake_run(project)
            result = report.build_figures(run, "english_only")
            self.assertEqual(result["status"], "ok")

            figure_dir = run.figures_dir("english_only")
            text = (figure_dir / "figure_index.md").read_text(encoding="utf-8")
            self.assertIsNone(han.search(text), "figure_index.md contains non-English text")

            manifest = artifacts.read_json(figure_dir / "figure_manifest.json")
            for record in manifest["figures"]:
                for field in ("title", "manuscript_section", "notes", "warnings"):
                    value = record.get(field)
                    text_value = " ".join(value) if isinstance(value, list) else str(value or "")
                    self.assertIsNone(han.search(text_value),
                                      f"{record['figure_id']}.{field} contains non-English text")
                for field in ("variables", "units", "filters", "transforms", "normalization",
                              "class_order"):
                    value = record.get(field)
                    text_value = " ".join(str(item) for item in (value or []))
                    self.assertIsNone(han.search(text_value),
                                      f"{record['figure_id']}.{field} contains non-English text")
