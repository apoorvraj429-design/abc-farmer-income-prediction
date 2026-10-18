"""Offline CSV boundaries, row diagnostics, and privacy checks for batch_tools."""
import io
import unittest

import numpy as np
import pandas as pd

import batch_tools
import core


class BatchUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.meta = core.load_artifacts()
        cls.sample = pd.read_csv(core.ROOT / "sample_batch.csv")

    def csv_bytes(self, frame=None):
        frame = self.sample if frame is None else frame
        return frame.to_csv(index=False).encode("utf-8")

    def test_exact_25mb_is_allowed_and_one_extra_byte_is_rejected(self):
        content = self.csv_bytes(self.sample.iloc[[0]])
        exact = content + b" " * (batch_tools.MAX_BYTES - len(content))
        self.assertEqual(len(exact), 25 * 1024 * 1024)
        parsed = batch_tools.read_batch(io.BytesIO(exact))
        self.assertEqual(len(parsed), 1)
        with self.assertRaisesRegex(batch_tools.BatchInputError, "25 MB"):
            batch_tools.read_batch(io.BytesIO(exact + b" "))

    def test_stream_upload_read_is_bounded(self):
        class Stream:
            def __init__(self):
                self.requested = None

            def read(self, size):
                self.requested = size
                return b"x" * size

        upload = Stream()
        with self.assertRaisesRegex(batch_tools.BatchInputError, "25 MB"):
            batch_tools.read_batch(upload)
        self.assertEqual(upload.requested, batch_tools.MAX_BYTES + 1)

    def test_15000_records_allowed_and_15001_rejected(self):
        allowed = pd.concat([self.sample.iloc[[0]]] * batch_tools.MAX_ROWS, ignore_index=True)
        parsed = batch_tools.read_batch(io.BytesIO(self.csv_bytes(allowed)))
        validated, issues, _ = batch_tools.inspect_rows(parsed, self.meta)
        self.assertEqual(len(validated), 15000)
        self.assertTrue(issues.empty)
        oversized = pd.concat([allowed, self.sample.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(batch_tools.BatchInputError, "15,000"):
            batch_tools.read_batch(io.BytesIO(self.csv_bytes(oversized)))
        with self.assertRaisesRegex(batch_tools.BatchInputError, "15,000"):
            batch_tools.inspect_rows(oversized, self.meta)

    def test_utf8_bom_and_plain_utf8_are_accepted(self):
        for prefix in (b"", b"\xef\xbb\xbf"):
            with self.subTest(bom=bool(prefix)):
                parsed = batch_tools.read_batch(io.BytesIO(prefix + self.csv_bytes()))
                validated, issues, _ = batch_tools.inspect_rows(parsed, self.meta)
                self.assertTrue(issues.empty)
                self.assertEqual(len(validated), len(self.sample))

    def test_non_utf8_upload_has_actionable_error(self):
        with self.assertRaisesRegex(batch_tools.BatchInputError, "CSV UTF-8"):
            batch_tools.read_batch(io.BytesIO(b"state,land_area\n\xff,3\n"))

    def test_empty_file_and_header_only_file_are_rejected_before_scoring(self):
        with self.assertRaisesRegex(batch_tools.BatchInputError, "empty|no data"):
            batch_tools.read_batch(io.BytesIO(b""))
        header = self.csv_bytes(self.sample.iloc[:0])
        parsed = batch_tools.read_batch(io.BytesIO(header))
        with self.assertRaisesRegex(batch_tools.BatchInputError, "no records"):
            batch_tools.inspect_rows(parsed, self.meta)

    def test_duplicate_friendly_and_source_csv_headers_are_rejected(self):
        for headers in (
            "state,land_area,land_area\nMADHYA PRADESH,1,2\n",
            "State,Total_Land_For_Agriculture,Total_Land_For_Agriculture\nMADHYA PRADESH,1,2\n",
            "state, land_area ,land_area\nMADHYA PRADESH,1,2\n",
        ):
            with self.subTest(header=headers.splitlines()[0]):
                with self.assertRaisesRegex(batch_tools.BatchInputError, "Duplicate column"):
                    batch_tools.read_batch(io.BytesIO(headers.encode()))

    def test_duplicate_dataframe_columns_are_also_rejected_during_extraction(self):
        for source_headers in (False, True):
            with self.subTest(source_headers=source_headers):
                frame = self.sample.rename(columns=core.SOURCE_COLUMNS) if source_headers else self.sample
                duplicate = pd.concat([frame, frame.iloc[:, [1]]], axis=1)
                with self.assertRaisesRegex(ValueError, "Duplicate column"):
                    batch_tools.inspect_rows(duplicate, self.meta)

    def test_original_source_headers_are_supported_and_identifiers_are_removed(self):
        raw = self.sample.rename(columns=core.SOURCE_COLUMNS)
        raw["FarmerID"] = ["PRIVATE-1", "PRIVATE-2", "PRIVATE-3"]
        raw["Phone"] = ["SECRET-1", "SECRET-2", "SECRET-3"]
        parsed = batch_tools.read_batch(io.BytesIO(self.csv_bytes(raw)))
        valid, issues, _ = batch_tools.inspect_rows(parsed, self.meta)
        self.assertTrue(issues.empty)
        self.assertEqual(valid.columns.tolist(), core.FEATURES)
        self.assertFalse(any("PRIVATE" in str(value) or "SECRET" in str(value) for value in valid.to_numpy().flat))
        expected, _ = core.validate_inputs(self.sample, self.meta)
        pd.testing.assert_frame_equal(valid, expected)

    def test_multiple_row_errors_keep_every_row_and_use_1based_data_positions(self):
        frame = self.sample.astype(object)
        frame.index = [10, 50, 90]
        frame.loc[10, "land_area"] = -1
        frame.loc[10, "non_ag_income"] = ""
        frame.loc[90, "state"] = "UNSUPPORTED"
        frame.loc[90, "kharif_rainfall"] = np.inf
        frame.loc[90, "rabi_rainfall"] = "bad"
        frame["FarmerID"] = ["PRIVATE-1", "PRIVATE-2", "PRIVATE-3"]
        returned, issues, warnings = batch_tools.inspect_rows(frame, self.meta)
        self.assertEqual(len(returned), 3)
        self.assertEqual(returned.index.tolist(), [0, 1, 2])
        self.assertEqual(returned.columns.tolist(), core.FEATURES)
        self.assertEqual(warnings, [])
        self.assertEqual(
            list(zip(issues.source_row, issues.field)),
            [(1, "land_area"), (1, "non_ag_income"), (3, "kharif_rainfall"), (3, "rabi_rainfall"), (3, "state")],
        )
        self.assertNotIn(2, issues.source_row.tolist())
        self.assertEqual(returned.iloc[1].state, self.sample.iloc[1].state)
        self.assertNotIn("FarmerID", issues.columns)

    def test_optional_missing_values_warn_but_are_not_invalid_rows(self):
        frame = self.sample.copy()
        frame.loc[1, "land_area"] = np.nan
        frame.loc[2, "agri_score"] = np.nan
        returned, issues, warnings = batch_tools.inspect_rows(frame, self.meta)
        self.assertTrue(issues.empty)
        self.assertEqual(len(returned), len(frame))
        self.assertEqual(len([w for w in warnings if "training median" in w]), 2)


if __name__ == "__main__":
    unittest.main()
