"""Dataset audit and dataset-view tests (A03-A08)."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests import helpers

from uavdet.artifacts import open_run
from uavdet.data import (
    audit_dataset,
    dataset_stats,
    label_row_to_xyxy,
    load_view_manifest,
    normalized_box_to_pixels,
    prepare_dataset_view,
    save_audit,
    split_dir_paths,
    view_split_entries,
)
from uavdet.types import CLASS_NAMES, DataError


class AuditHappyPathTests(helpers.TempDirTestCase):
    def test_clean_dataset_passes_and_counts_add_up(self):
        """A07: fixed 0-9 class order, totals equal the GT instance count."""
        project = self.project(n_train=5, n_val=3)
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["failures"], [])
        self.assertEqual(audit["splits"]["train"]["num_images"], 5)
        self.assertEqual(audit["splits"]["val"]["num_images"], 3)
        per_class = audit["splits"]["train"]["per_class"]
        self.assertEqual(list(per_class), list(CLASS_NAMES))
        self.assertEqual(sum(per_class.values()), audit["splits"]["train"]["num_instances"])
        self.assertEqual(audit["totals"]["num_images"], 8)
        self.assertEqual(sum(audit["totals"]["num_instances"] for _ in range(1)),
                         audit["splits"]["train"]["num_instances"]
                         + audit["splits"]["val"]["num_instances"])
        # the manifest lists every image once
        self.assertEqual(len(audit["images"]), 8)
        self.assertEqual(len({(e["split"], e["image_id"]) for e in audit["images"]}), 8)

    def test_manifest_digest_is_content_based(self):
        project = self.project()
        first = audit_dataset(project.config())
        second = audit_dataset(project.config())
        self.assertEqual(first["manifest_digest"], second["manifest_digest"])
        image = next(project.dataset["images"]["train"].glob("*.jpg"))
        helpers.write_png(image, project.dataset["width"], project.dataset["height"],
                          colour=(9, 9, 9))
        third = audit_dataset(project.config())
        self.assertNotEqual(first["manifest_digest"], third["manifest_digest"])

    def test_audit_never_writes_into_the_data_root(self):
        project = self.project()
        before = helpers.sha_tree(project.data_root)
        audit = audit_dataset(project.config())
        stats = dataset_stats(audit)
        self.assertEqual(stats["status"], "ok")
        after = helpers.sha_tree(project.data_root)
        self.assertEqual(before, after, "the source dataset must stay byte-identical")

    def test_empty_label_is_legal_and_counted(self):
        """A05: legal empty annotation is counted, not rejected."""
        project = self.project(anomalies=("missing_label",))
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "ok")
        kinds = {issue["kind"] for issue in audit["issues"]}
        self.assertIn("missing_label_file", kinds)
        self.assertGreaterEqual(audit["splits"]["train"]["empty_label_images"], 1)
        self.assertEqual(audit["splits"]["train"]["num_label_files"],
                         audit["splits"]["train"]["num_images"] - 1)

    def test_area_histogram_matches_instance_count(self):
        project = self.project()
        audit = audit_dataset(project.config())
        for split in ("train", "val"):
            block = audit["splits"][split]
            histogram = block["area_histogram"]
            self.assertEqual(len(histogram["edges"]), histogram["bin_count"] + 1)
            self.assertEqual(len(histogram["counts"]), histogram["bin_count"])
            self.assertEqual(sum(histogram["counts"]), block["num_instances"])
            self.assertLessEqual(histogram["counts"][0] + histogram["counts"][-1],
                                 block["num_instances"])
            quantiles = block["area_quantiles"]
            self.assertGreaterEqual(quantiles["max"], quantiles["p50"])
            self.assertGreaterEqual(quantiles["p50"], quantiles["min"])

    def test_per_image_objects_length_matches_images(self):
        project = self.project(n_train=3, n_val=2)
        audit = audit_dataset(project.config())
        for split in ("train", "val"):
            block = audit["splits"][split]
            self.assertEqual(len(block["per_image_objects"]), block["num_images"])
            self.assertEqual(len(block["per_image_ids"]), block["num_images"])
            self.assertEqual(sum(block["per_image_objects"]), block["num_instances"])


class AuditFailureTests(helpers.TempDirTestCase):
    def test_corrupt_image_fails_the_audit(self):
        """A04: an undecodable image fails the audit with its location."""
        project = self.project(anomalies=("corrupt_image",))
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "failed")
        kinds = {issue["kind"] for issue in audit["issues"]}
        self.assertIn("corrupt_image", kinds)
        self.assertTrue(any("val_broken" in failure for failure in audit["failures"]))

    def test_negative_box_is_rejected(self):
        """A04: a negative box size is an error, not silently clipped."""
        project = self.project(anomalies=("bad_box",))
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "failed")
        kinds = {issue["kind"] for issue in audit["issues"]}
        self.assertIn("invalid_label", kinds)

    def test_out_of_range_class_is_rejected(self):
        """A04: a class id outside 0-9 fails the audit."""
        project = self.project(anomalies=("out_of_range_cls",))
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "failed")
        details = " ".join(issue["detail"] for issue in audit["issues"]
                           if issue["kind"] == "invalid_label")
        self.assertIn("class id 14", details)

    def test_orphan_label_is_rejected(self):
        """A04: a label with no matching image fails the audit."""
        project = self.project(anomalies=("orphan_label",))
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "failed")
        self.assertIn("orphan_label", {issue["kind"] for issue in audit["issues"]})

    def test_image_count_mismatch_is_rejected(self):
        project = self.project()
        config = project.config({"data.expected_train_images": 999})
        audit = audit_dataset(config)
        self.assertEqual(audit["status"], "failed")
        self.assertIn("image_count_mismatch", {issue["kind"] for issue in audit["issues"]})

    def test_cross_split_duplicate_fails_and_same_split_duplicate_warns(self):
        project = self.project(duplicate_across_splits=True)
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "failed")
        kinds = {issue["kind"] for issue in audit["issues"]}
        self.assertIn("cross_split_duplicate_image", kinds)

        same = self.project(name="same_split", n_train=4, n_val=2)
        images = sorted(same.dataset["images"]["train"].glob("*.jpg"))
        import shutil

        # Same count, identical content: a duplicate inside one split.
        shutil.copy2(images[0], images[1])
        audit2 = audit_dataset(same.config())
        self.assertEqual(audit2["status"], "ok")
        self.assertIn("duplicate_image", {issue["kind"] for issue in audit2["issues"]})

    def test_duplicate_annotation_lines_are_reported(self):
        project = self.project()
        stembase = project.dataset["stems"]["train"][0]
        helpers.write_label(
            project.dataset["labels"]["train"] / f"{stembase}.txt",
            [(0, 0.5, 0.5, 0.2, 0.2), (0, 0.5, 0.5, 0.2, 0.2)],
        )
        audit = audit_dataset(project.config())
        self.assertIn("duplicate_annotation", {issue["kind"] for issue in audit["issues"]})


class CoordinateConversionTests(helpers.TempDirTestCase):
    def test_reference_case_from_acceptance_list(self):
        # A06: 100x80 image, label `0 .5 .5 .2 .25` -> xyxy [40, 30, 60, 50]
        box = label_row_to_xyxy(0.5, 0.5, 0.2, 0.25, 100, 80)
        self.assertEqual(box, (40.0, 30.0, 60.0, 50.0))
        self.assertEqual(normalized_box_to_pixels((0.4, 0.375, 0.6, 0.625), 100, 80),
                         (40.0, 30.0, 60.0, 50.0))

    def test_zero_size_label_is_rejected(self):
        project = self.project()
        stem = project.dataset["stems"]["val"][0]
        helpers.write_label(project.dataset["labels"]["val"] / f"{stem}.txt",
                            [(0, 0.5, 0.5, 0.0, 0.2)])
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "failed")
        details = " ".join(i["detail"] for i in audit["issues"] if i["kind"] == "invalid_label")
        self.assertIn("non-positive", details)


class DatasetViewTests(helpers.TempDirTestCase):
    def test_view_is_a_copy_and_source_is_untouched(self):
        project = self.project(n_train=5, n_val=3)
        config = project.config()
        run = open_run(config, "view_run")
        audit = audit_dataset(config)
        save_audit(audit, run)
        before = helpers.sha_tree(project.data_root)

        data_yaml = prepare_dataset_view(audit, run, config)
        self.assertTrue(data_yaml.exists())
        view_root = data_yaml.parent
        self.assertNotEqual(view_root.resolve(), project.data_root.resolve())
        self.assertTrue((view_root / "images" / "train").is_dir())
        self.assertEqual(len(list((view_root / "images" / "train").glob("*.jpg"))), 5)
        self.assertEqual(len(list((view_root / "images" / "val").glob("*.jpg"))), 3)

        import yaml as pyyaml

        payload = pyyaml.safe_load(data_yaml.read_text())
        self.assertEqual(payload["train"], "images/train")
        self.assertEqual(payload["val"], "images/val")
        self.assertEqual(payload["nc"], 10)
        self.assertEqual(Path(payload["path"]), view_root.resolve())
        self.assertEqual(payload["names"][0], "pedestrian")

        manifest = load_view_manifest(data_yaml)
        self.assertEqual(manifest["scope"], "formal")
        self.assertGreater(manifest["bytes_copied"], 0)
        self.assertEqual(len(view_split_entries(data_yaml, "train")), 5)

        self.assertEqual(helpers.sha_tree(project.data_root), before,
                         "preparing a view must not modify the source dataset")

    def test_identical_view_is_reused_not_recopied(self):
        project = self.project(n_train=4, n_val=2)
        config = project.config()
        run = open_run(config, "reuse_run")
        audit = audit_dataset(config)
        save_audit(audit, run)
        first = prepare_dataset_view(audit, run, config)
        manifest_first = load_view_manifest(first)
        second = prepare_dataset_view(audit, run, config)
        self.assertEqual(first, second)
        self.assertEqual(load_view_manifest(second)["created_at"], manifest_first["created_at"])
        self.assertTrue(run.manifest["dataset_view"]["reused"])

    def test_smoke_view_selects_the_configured_subset(self):
        project = self.project(n_train=24, n_val=12, scope="smoke")
        config = project.config()
        run = open_run(config, "smoke_view")
        audit = audit_dataset(config)
        self.assertEqual(audit["status"], "ok")
        save_audit(audit, run)
        data_yaml = prepare_dataset_view(audit, run, config)
        manifest = load_view_manifest(data_yaml)
        self.assertEqual(manifest["scope"], "smoke")
        self.assertEqual(len(view_split_entries(data_yaml, "train")), 20)
        self.assertEqual(len(view_split_entries(data_yaml, "val")), 10)
        self.assertIn("_smoke", data_yaml.parent.name)

    def test_failed_audit_blocks_view_creation(self):
        project = self.project(anomalies=("corrupt_image",))
        config = project.config()
        run = open_run(config, "bad_run")
        audit = audit_dataset(config)
        with self.assertRaises(DataError):
            prepare_dataset_view(audit, run, config)
        self.assertFalse((project.workspace_dir / "data_views").exists())

    def test_split_paths_follow_data_yaml(self):
        project = self.project()
        paths = split_dir_paths(project.data_root, "val")
        self.assertEqual(paths.images_dir.name, "images")
        self.assertEqual(paths.labels_dir.name, "labels")
        self.assertTrue(paths.images_dir.is_dir())


class DegenerateBoxRepairTests(helpers.TempDirTestCase):
    """`data.label_repair` is the only sanctioned repair and is fully recorded."""

    def _project_with_zero_height_box(self):
        project = self.project(n_train=4, n_val=2)
        stem = project.dataset["stems"]["train"][0]
        helpers.write_label(project.dataset["labels"]["train"] / f"{stem}.txt", [
            (0, 0.5, 0.5, 0.2, 0.2),
            (3, 0.4378, 0.1504, 0.002857, 0.0),
            (1, 0.2, 0.3, 0.05, 0.06),
        ])
        return project

    def test_strict_policy_rejects_zero_area_boxes(self):
        project = self._project_with_zero_height_box()
        audit = audit_dataset(project.config())
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["label_repair"]["policy"], "strict")
        self.assertEqual(audit["repairs"], [])
        self.assertIn("invalid_label", {i["kind"] for i in audit["issues"]})

    def test_drop_policy_records_every_dropped_line(self):
        project = self._project_with_zero_height_box()
        config = project.config({"data.label_repair": "drop_degenerate_boxes"})
        audit = audit_dataset(config)
        self.assertEqual(audit["status"], "ok", audit["failures"])
        self.assertEqual(audit["label_repair"]["dropped_boxes"], 1)
        self.assertEqual(audit["label_repair"]["repaired_label_files"], 1)
        repair = audit["repairs"][0]
        self.assertEqual(repair["line_no"], 2)
        self.assertIn("non-positive box size", repair["reason"])
        kinds = {i["kind"] for i in audit["issues"]}
        self.assertIn("dropped_degenerate_box", kinds)
        # the dropped line no longer counts as an instance
        split = audit["splits"]["train"]
        self.assertEqual(split["num_instances"], 2 * (4 - 1) + 2)

    def test_drop_policy_does_not_hide_other_label_errors(self):
        project = self._project_with_zero_height_box()
        stem = project.dataset["stems"]["val"][0]
        helpers.write_label(project.dataset["labels"]["val"] / f"{stem}.txt", [
            (3, 0.5, 0.5, 0.0, 0.2),
            (14, 0.5, 0.5, 0.2, 0.2),      # class out of range: never repaired
        ])
        config = project.config({"data.label_repair": "drop_degenerate_boxes"})
        audit = audit_dataset(config)
        self.assertEqual(audit["status"], "failed")
        self.assertIn("class id 14", " ".join(audit["failures"]))

    def test_view_copy_contains_the_repair_and_source_is_untouched(self):
        import yaml as pyyaml

        project = self._project_with_zero_height_box()
        config = project.config({"data.label_repair": "drop_degenerate_boxes"})
        run = open_run(config, "repair_view")
        before = helpers.sha_tree(project.data_root)
        audit = audit_dataset(config)
        save_audit(audit, run)
        data_yaml = prepare_dataset_view(audit, run, config)
        self.assertTrue((run.audit_dir / "label_repairs.csv").exists())
        manifest = load_view_manifest(data_yaml)
        repaired = [entry for split in manifest["splits"].values()
                    for entry in split["entries"] if entry.get("dropped_label_lines")]
        self.assertEqual(len(repaired), 1)
        view_label = data_yaml.parent / repaired[0]["label_rel"]
        self.assertEqual(len(view_label.read_text().splitlines()), 2)
        source_rel = repaired[0]["image_id"]
        original_label = next(project.dataset["labels"]["train"].glob(f"{source_rel}.txt"))
        self.assertEqual(len(original_label.read_text().splitlines()), 3,
                         "the source label file must keep all three lines")
        self.assertEqual(helpers.sha_tree(project.data_root), before)
        payload = pyyaml.safe_load(data_yaml.read_text())
        self.assertEqual(payload["nc"], 10)


if __name__ == "__main__":
    unittest.main()
