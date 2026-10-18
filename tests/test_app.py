"""Streamlit integration checks with the real model and an entirely mocked AI.

The network transport is patched to fail in every test; no live API call is made.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import assistant
import core


class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config_patch = patch(
            "assistant.load_config", return_value=assistant.AssistantConfig()
        ).start()
        self.reply_patch = patch("assistant.stream_reply").start()
        self.network_patch = patch(
            "assistant.request.urlopen", side_effect=AssertionError("Real AI requests are forbidden in tests")
        ).start()
        self.addCleanup(patch.stopall)
        self.calls = []

    def tearDown(self):
        self.network_patch.assert_not_called()

    def app(self):
        at = AppTest.from_file(str(core.ROOT / "app.py"), default_timeout=20).run()
        self.assert_no_exception(at)
        return at

    def assert_no_exception(self, at):
        self.assertEqual(len(at.exception), 0, [e.message for e in at.exception])

    def navigate(self, at, page):
        at.radio(key="workspace").set_value(page).run()
        self.assert_no_exception(at)
        return at

    def button(self, at, label):
        return next(button for button in at.button if button.label == label)

    def example(self, at):
        self.navigate(at, "Income estimator")
        self.button(at, "Load a hypothetical example").click().run()
        self.button(at, "Estimate income").click().run()
        self.assert_no_exception(at)
        self.assertEqual(len(at.error), 0)
        return at.session_state["last_income"]["row"].copy()

    def ready(self, chunks=("A helpful ", "reply.")):
        # This deliberately fake key exists only in the test process. The reply
        # function is replaced, and the network transport also remains blocked.
        self.config_patch.return_value = assistant.AssistantConfig(api_key="test-only-fake-key")

        def fake(messages, config, prediction_context=None):
            self.calls.append({"messages": deepcopy(messages), "context": deepcopy(prediction_context)})
            return iter(chunks)

        self.reply_patch.side_effect = fake

    def send(self, at, prompt):
        at.chat_input[0].set_value(prompt).run()
        self.assert_no_exception(at)

    def test_exact_four_pages_render_without_credentials(self):
        at = self.app()
        expected = ["Overview", "Income estimator", "Batch predictions", "AI assistant"]
        self.assertEqual(at.radio(key="workspace").options, expected)
        for page in expected:
            with self.subTest(page=page):
                self.navigate(at, page)
                self.assertEqual(at.radio(key="workspace").options, expected)
        self.assertTrue(at.chat_input[0].disabled)
        self.assertTrue(all(at.button(key="starter_" + str(i)).disabled for i in range(3)))
        self.assertTrue(any("OPENAI_API_KEY" in item.value for item in at.info))
        self.reply_patch.assert_not_called()

    def test_missing_nonfarm_is_rejected_then_real_example_result_survives_no_key_ai(self):
        at = self.app()
        self.navigate(at, "Income estimator")
        at.selectbox(key="input_state").set_value("MADHYA PRADESH")
        self.button(at, "Estimate income").click().run()
        self.assert_no_exception(at)
        self.assertTrue(any("income is required" in item.value for item in at.error))
        self.assertNotIn("last_income", at.session_state.filtered_state)
        expected_row = self.example(at)
        displayed = "\n".join(item.value for item in at.markdown)
        self.assertIn("₹755,403", displayed)
        self.assertIn("ESTIMATED TOTAL INCOME", displayed)
        self.navigate(at, "AI assistant")
        self.assertTrue(at.chat_input[0].disabled)
        pd.testing.assert_frame_equal(at.session_state["last_income"]["row"], expected_row)
        self.navigate(at, "Income estimator")
        self.assertIn("₹755,403", "\n".join(item.value for item in at.markdown))
        self.reply_patch.assert_not_called()

    def test_unicode_streams_into_completed_history_and_starter_uses_history(self):
        self.ready(("नमस्ते ", "🌾 Your estimate is ready."))
        at = self.app()
        self.navigate(at, "AI assistant")
        self.assertFalse(at.chat_input[0].disabled)
        self.send(at, "Explain this app in Hindi.")
        expected_answer = "नमस्ते 🌾 Your estimate is ready."
        self.assertEqual(at.session_state["messages"], [
            {"role": "user", "content": "Explain this app in Hindi."},
            {"role": "assistant", "content": expected_answer},
        ])
        self.assertIn(expected_answer, "\n".join(item.value for item in at.markdown))
        at.button(key="starter_0").click().run()
        self.assert_no_exception(at)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1]["messages"][-1], {"role": "user", "content": "What inputs do I need?"})
        self.assertEqual(self.calls[1]["messages"][-2], {"role": "assistant", "content": expected_answer})
        self.assertIsNone(self.calls[0]["context"])
        self.assertIsNone(self.calls[1]["context"])
        self.assertEqual(len(at.session_state["messages"]), 4)

    def test_sharing_requires_optin_and_passes_only_the_exact_single_estimate(self):
        self.ready()
        at = self.app()
        row = self.example(at)
        self.navigate(at, "AI assistant")
        self.assertFalse(at.checkbox(key="share_estimate").value)
        self.assertFalse(at.checkbox(key="share_estimate").disabled)
        self.send(at, "What does the error range mean?")
        self.assertIsNone(self.calls[-1]["context"])
        at.checkbox(key="share_estimate").check().run()
        self.assert_no_exception(at)
        self.send(at, "Explain my shared estimate.")
        context = self.calls[-1]["context"]
        self.assertIsNotNone(context)
        self.assertEqual(set(context), {"inputs", "prediction", "notes"})
        self.assertEqual(context["inputs"], row.iloc[0].to_dict())
        bundle, meta = core.load_artifacts()
        result = core.score(row, bundle, meta).iloc[0]
        self.assertEqual(context["prediction"], {
            "predicted_total_income_inr": float(result.predicted_total_income_inr),
            "range_low_inr": float(result.range_low_inr),
            "range_high_inr": float(result.range_high_inr),
            "probability_below_benchmark": float(result.probability_below_benchmark),
            "benchmark_inr": float(meta["benchmark_inr"]),
        })
        self.assertIn("Reporting period and land units unspecified", context["notes"])
        self.assertNotIn("FarmerID", json.dumps(context))
        at.checkbox(key="share_estimate").uncheck().run()
        self.send(at, "Now explain generally.")
        self.assertIsNone(self.calls[-1]["context"])

    def test_partial_reply_is_marked_incomplete_and_not_retained_as_a_completed_turn(self):
        self.ready(("A completed answer.",))
        at = self.app()
        row = self.example(at)
        self.navigate(at, "AI assistant")
        self.send(at, "First question")
        completed = deepcopy(at.session_state["messages"])

        def partial(messages, config, prediction_context=None):
            yield "An unfinished answer"
            raise assistant.AssistantError("connection", "The connection was interrupted.", partial=True)

        self.reply_patch.side_effect = partial
        self.send(at, "Second question")
        self.assertTrue(any("stopped early" in warning.value for warning in at.warning))
        self.assertTrue(any("interrupted" in error.value for error in at.error))
        self.assertEqual(at.session_state["messages"], completed)
        self.navigate(at, "Income estimator")
        pd.testing.assert_frame_equal(at.session_state["last_income"]["row"], row)
        self.assertIn("₹755,403", "\n".join(item.value for item in at.markdown))

    def test_service_failure_does_not_break_prediction_page_or_retain_failed_turn(self):
        self.ready()
        self.reply_patch.side_effect = assistant.AssistantError("rate_limit", "Please try again shortly.")
        at = self.app()
        self.navigate(at, "AI assistant")
        self.send(at, "A question during a service outage")
        self.assertEqual(at.session_state["messages"], [])
        self.assertTrue(any("try again" in error.value for error in at.error))
        self.example(at)
        self.assertIn("₹755,403", "\n".join(item.value for item in at.markdown))

    def test_clear_conversation_removes_history_and_resets_sharing(self):
        self.ready()
        at = self.app()
        row = self.example(at)
        self.navigate(at, "AI assistant")
        at.checkbox(key="share_estimate").check().run()
        self.send(at, "Explain the result")
        self.assertEqual(len(at.session_state["messages"]), 2)
        self.button(at, "Clear conversation").click().run()
        self.assert_no_exception(at)
        self.assertEqual(at.session_state["messages"], [])
        self.assertFalse(at.checkbox(key="share_estimate").value)
        self.assertIn("Hello!", "\n".join(item.value for item in at.markdown))
        pd.testing.assert_frame_equal(at.session_state["last_income"]["row"], row)


if __name__ == "__main__":
    unittest.main()
