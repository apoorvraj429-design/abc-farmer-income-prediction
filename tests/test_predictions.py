"""Regression and input-contract checks using the retained, real trained artifacts.

Run from the project folder with: python -m unittest discover -s tests -v
No model training, API credentials or network access is required.
"""
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import core


class PredictionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.meta = core.load_artifacts()
        cls.baseline = json.loads(
            (Path(__file__).parent / "prediction_baseline.json").read_text()
        )
        cls.sample = pd.DataFrame(cls.baseline["inputs"])

    def single(self):
        return self.sample.iloc[[0]].copy()

    def test_original_model_artifacts_and_metadata_are_retained(self):
        for filename, expected in self.baseline["artifact_sha256"].items():
            with self.subTest(file=filename):
                actual = hashlib.sha256((core.ROOT / filename).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_all_original_sample_predictions_are_unchanged(self):
        validated, warnings = core.validate_inputs(self.sample, self.meta)
        self.assertEqual(warnings, [])
        expected = pd.DataFrame(self.baseline["predictions"])
        actual = core.score(validated, self.bundle, self.meta)
        self.assertEqual(actual.columns.tolist(), expected.columns.tolist())
        for column in expected:
            with self.subTest(output=column):
                if expected[column].dtype.kind in "biu":
                    np.testing.assert_array_equal(actual[column], expected[column])
                else:
                    np.testing.assert_allclose(
                        actual[column], expected[column], rtol=1e-12, atol=1e-7
                    )

    def test_original_source_headers_and_friendly_headers_score_identically(self):
        original = self.sample.rename(columns=core.SOURCE_COLUMNS)
        original.columns = "  " + original.columns + "  "
        original["  State  "] = "  madhya pradesh  "
        friendly = self.sample.copy()
        friendly["state"] = "MADHYA PRADESH"
        a, _ = core.validate_inputs(original, self.meta)
        b, _ = core.validate_inputs(friendly, self.meta)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(
            core.score(a, self.bundle, self.meta),
            core.score(b, self.bundle, self.meta),
        )

    def test_required_columns_are_reported(self):
        for column in core.FEATURES:
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "Missing columns.*" + column):
                    core.validate_inputs(self.single().drop(columns=column), self.meta)

    def test_duplicate_headers_are_rejected(self):
        duplicate = pd.concat([self.single(), self.single()[["land_area"]]], axis=1)
        with self.assertRaisesRegex(ValueError, "Duplicate column"):
            core.validate_inputs(duplicate, self.meta)

    def test_empty_upload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no records"):
            core.validate_inputs(self.sample.iloc[:0], self.meta)

    def test_row_limit_allows_15000_and_rejects_15001(self):
        allowed = pd.concat([self.single()] * 15000, ignore_index=True)
        validated, _ = core.validate_inputs(allowed, self.meta)
        self.assertEqual(len(validated), 15000)
        rejected = pd.concat([allowed, self.single()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "15,000"):
            core.validate_inputs(rejected, self.meta)

    def test_nonfarm_income_is_required_and_zero_is_preserved(self):
        for missing in (None, "", "   ", np.nan):
            with self.subTest(missing=repr(missing)):
                row = self.single().astype(object)
                row.loc[row.index[0], "non_ag_income"] = missing
                with self.assertRaisesRegex(ValueError, "income is required"):
                    core.validate_inputs(row, self.meta)
        row = self.single()
        row["non_ag_income"] = 0
        valid, _ = core.validate_inputs(row, self.meta)
        self.assertEqual(valid.non_ag_income.iloc[0], 0.0)

    def test_optional_numeric_blanks_use_training_medians_for_scoring(self):
        for column in core.NUM:
            if column == "non_ag_income":
                continue
            with self.subTest(column=column):
                blank = self.single()
                blank[column] = np.nan
                explicit = self.single()
                explicit[column] = self.meta["schema"][column]["median"]
                a, warnings = core.validate_inputs(blank, self.meta)
                b, _ = core.validate_inputs(explicit, self.meta)
                self.assertTrue(any("training median" in item for item in warnings))
                np.testing.assert_allclose(
                    core.score(a, self.bundle, self.meta).to_numpy(dtype=float),
                    core.score(b, self.bundle, self.meta).to_numpy(dtype=float),
                    rtol=1e-12,
                    atol=1e-7,
                )

    def test_invalid_numeric_values_are_rejected_in_every_numeric_column(self):
        for column in core.NUM:
            for bad in (-1.0, np.inf, -np.inf, "not a number"):
                with self.subTest(column=column, value=bad):
                    row = self.single().astype(object)
                    row.loc[row.index[0], column] = bad
                    with self.assertRaises(ValueError):
                        core.validate_inputs(row, self.meta)

    def test_unknown_and_blank_states_are_rejected(self):
        for state in ("UNKNOWN STATE", "OTHER", "", " ", None):
            with self.subTest(state=state):
                row = self.single()
                row["state"] = state
                with self.assertRaisesRegex(ValueError, "State is required|Unsupported state"):
                    core.validate_inputs(row, self.meta)

    def test_supported_rare_states_use_retained_other_encoding(self):
        rare = [s for s in self.meta["states"] if s not in self.bundle["common_states"]]
        self.assertTrue(rare)
        rows = pd.concat([self.single()] * len(rare), ignore_index=True)
        rows["state"] = rare
        valid, _ = core.validate_inputs(rows, self.meta)
        encoded = core.model_inputs(valid, self.bundle)
        self.assertEqual(encoded.state.tolist(), ["OTHER"] * len(rare))
        actual = core.score(valid, self.bundle, self.meta)
        self.assertTrue(np.isfinite(actual.to_numpy(dtype=float)).all())

    def test_out_of_range_nonnegative_numbers_warn_without_changing_values(self):
        for column in core.NUM:
            with self.subTest(column=column):
                row = self.single()
                value = self.meta["schema"][column]["max"] + 1
                row[column] = value
                valid, warnings = core.validate_inputs(row, self.meta)
                self.assertEqual(valid[column].iloc[0], value)
                self.assertTrue(any("outside the training range" in item for item in warnings))

    def test_identifiers_and_extra_source_columns_do_not_enter_outputs(self):
        raw = self.sample.copy()
        raw["FarmerID"] = ["SECRET-001", "SECRET-002", "SECRET-003"]
        raw["phone"] = ["PRIVATE-1", "PRIVATE-2", "PRIVATE-3"]
        valid, _ = core.validate_inputs(raw, self.meta)
        self.assertEqual(valid.columns.tolist(), core.FEATURES)
        scored = core.score(valid, self.bundle, self.meta)
        csv_text = core.safe_csv(scored).decode("utf-8-sig")
        self.assertNotIn("FarmerID", csv_text)
        self.assertNotIn("phone", csv_text)
        self.assertNotIn("SECRET", csv_text)
        self.assertNotIn("PRIVATE", csv_text)

    def test_csv_escapes_spreadsheet_formulas_without_changing_numbers(self):
        expressions = ["=1+1", "+SUM(A1:A2)", "-cmd", "@SUM(A1)", "  =HYPERLINK(\"x\")", "plain"]
        original = pd.DataFrame({"label": expressions, "amount": [-3, 0, 1, 2, 3, 4]})
        encoded = core.safe_csv(original)
        self.assertTrue(encoded.startswith(b"\xef\xbb\xbf"))
        exported = pd.read_csv(io.BytesIO(encoded), keep_default_na=False)
        self.assertEqual(exported.label.tolist(), ["'" + s for s in expressions[:-1]] + ["plain"])
        self.assertEqual(exported.amount.tolist(), original.amount.tolist())
        self.assertEqual(original.label.tolist(), expressions)

    def test_contributions_reconstruct_each_raw_original_sample_prediction(self):
        valid, _ = core.validate_inputs(self.sample, self.meta)
        scored = core.score(valid, self.bundle, self.meta)
        for i in range(len(valid)):
            with self.subTest(profile=i):
                parts, intercept = core.contributions(valid.iloc[[i]], self.bundle)
                self.assertEqual(len(parts), 8)
                self.assertEqual(set(parts.factor), set(core.LABELS.values()))
                self.assertAlmostEqual(
                    float(parts.contribution_inr.sum()) + intercept,
                    scored.raw_linear_prediction_inr.iloc[i],
                    places=7,
                )

    def test_negative_raw_estimates_are_clipped_and_inconsistency_is_flagged(self):
        row = self.single()
        row["kharif_rainfall"] = 1_000_000
        valid, warnings = core.validate_inputs(row, self.meta)
        self.assertTrue(warnings)
        result = core.score(valid, self.bundle, self.meta).iloc[0]
        self.assertLess(result.raw_linear_prediction_inr, 0)
        self.assertEqual(result.predicted_total_income_inr, 0)
        self.assertEqual(result.range_low_inr, 0)
        self.assertTrue(result.below_known_non_ag_income)

    def test_empirical_error_band_and_probability_threshold_are_preserved(self):
        valid, _ = core.validate_inputs(self.sample, self.meta)
        result = core.score(valid, self.bundle, self.meta)
        self.assertTrue(result.probability_below_benchmark.between(0, 1).all())
        np.testing.assert_array_equal(
            result.predicted_below_benchmark,
            result.probability_below_benchmark >= self.meta["probability_threshold"],
        )
        np.testing.assert_allclose(
            result.range_high_inr - result.predicted_total_income_inr,
            self.meta["error_band_radius"],
        )
        np.testing.assert_allclose(
            result.range_low_inr,
            np.maximum(0, result.predicted_total_income_inr - self.meta["error_band_radius"]),
        )


if __name__ == "__main__":
    unittest.main()
