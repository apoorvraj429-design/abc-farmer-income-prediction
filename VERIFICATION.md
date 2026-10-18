# Verification report

Release: Kisan Saathi redesign, 18 October 2026.

## Source reviewed

The connected GitHub repository `apoorvraj429-design/abc-farmer-income-prediction` was accessible. Work began from main commit `b6fc4bac7b0653b0fd2c0cf0f582ac80eaeaf83b`. Its application sources match the `APP_SOURCES` embedded in the supplied Colab notebook exactly. The newly attached notebook, Word guide, and one-page presentation were byte-identical to their earlier attached copies, which were already read. The deployed Streamlit page could not be inspected through the web retrieval tool; the repository source and local application were used for implementation and verification.

## Model preservation

`core.py`, `requirements.txt`, `model_bundle.joblib`, `metrics.json`, `dashboard_data.json`, and `sample_batch.csv` are unchanged. The original fitted linear and logistic models were loaded and exercised. No training or synthetic replacement data was introduced.

| Template profile | Original and updated total estimate in INR |
| --- | ---: |
| Madhya Pradesh | 755,403.2753825553 |
| Maharashtra | 1,074,961.4397740634 |
| Karnataka | 1,320,055.0594677641 |

These are saved-model predictions for hypothetical example inputs, not observed farmer outcomes. Existing training/holdout performance was not reevaluated because the raw labelled source data is not included.

## Automated verification

Run from the project root:

```bash
python -m unittest discover -s tests -v
```

54 tests passed in the implementation environment (Python 3.12.14, scikit-learn 1.8.0, pandas 2.2.3, NumPy 2.3.5, SciPy 1.17.0, joblib 1.5.3, Streamlit 1.50.0).

- 18 prediction tests check baseline values, artifact preservation, both input-header formats, required non-farm income, optional numeric imputation, unknown states, numeric validation, 15,000 rows, additive explanations, error bands, clipping, and safe CSV exports.
- 11 CSV tests check the exact 25 MiB boundary, one byte beyond it, bounded reading, UTF-8/BOM, duplicate headers, empty inputs, multiple row-specific errors, original headers, and identifier exclusion.
- 18 assistant transport tests use mocked HTTP/SSE. They check successful streaming, Unicode, bounded history, context whitelisting, placeholder/missing credentials, authentication, quota, rate limit, timeout, network failure, malformed events, empty replies, and interrupted/incomplete output.
- 7 Streamlit AppTest integration tests exercise all four pages, blank-income rejection, actual saved-model results, retained estimates, simulated replies, history, sharing off/on/off, failed replies, and clearing both history and the sharing selection.

A configured-chat icon incompatibility found by the UI tests was corrected. The completed tests make no real API requests and incur no API charges.

## Desktop and mobile browser verification

The running Streamlit app was inspected in Firefox at 1440 × 1000 and 390 × 844. All four pages rendered without browser console errors, JavaScript page errors, or Streamlit exceptions. There was no horizontal page overflow at either size. The farmer remained visible in the mobile image, and forms and results stacked correctly.

The browser flow exercised blank-input feedback, loading the hypothetical example, the actual ₹755,403 result, the individual prediction download, a three-row batch and its download, row-specific input errors, missing CSV headers, the unconfigured chatbot, reviewing an explicitly shared estimate, clearing the sharing selection, and returning to the estimator. The original trained model produced the results. A successful authenticated AI response was not part of this browser run.

Caption contrast was improved after inspecting computed styles. Remaining server notices concern Streamlit's deprecated `use_container_width` argument; it is supported by the pinned Streamlit 1.50.0 release and did not prevent any checked flow.

## Account-specific checks still required

No API key was supplied, so a real authenticated OpenAI response and the account’s billing/model access could not be verified. Add your private key to Streamlit Secrets, send a short question, then confirm a reply. No live weather, mandi-price, scheme, or search connector is included. The assistant has explicit instructions to explain this limitation.

The code is prepared on a separate review branch. The existing deployed app receives changes only after the deployed branch is updated. Verify the deployed app with the START_HERE checklist after that update.
