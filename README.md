# Kisan Saathi

A redesigned Streamlit app for estimating total income in INR and asking an AI assistant for explanations. It uses the original trained farmer-income models. There are four sections: **Overview**, **Income estimator**, **Batch predictions**, and **AI assistant**.

**Start with [START_HERE.md](START_HERE.md)** for backup, installation, GitHub upload, secrets, deployment, and rollback instructions.

## What changed

- A green and earthy design with local photographs, readable text, and layouts that adapt to smaller screens.
- A clearer individual estimator, a labelled example profile, and explanations beside the result.
- CSV preview, input checks, row-level validation feedback, and downloadable predictions.
- A conversational AI assistant with streamed replies, starter questions, session history, and a clear-chat control.
- The Model evidence, Manager study, and Study analysis sections and their obsolete study code were removed.

The connected GitHub source matched the notebook's embedded application sources. The original `core.py`, model bundle, metadata, dashboard data, CSV template, and pinned prediction dependencies are retained. There is no retraining in this release. The live deployed page could not be inspected through web retrieval; the repository and local app were used for this update.

## Files

| File or folder | Action for the existing app | Purpose |
| --- | --- | --- |
| `app.py` | Replace | Complete redesigned Streamlit application |
| `assistant.py` | Add | AI request handling, prompt rules, streaming, and friendly failures |
| `batch_tools.py` | Add | Bounded CSV parsing and row-specific input errors |
| `assets/` | Add | Local photographs and design assets |
| `.streamlit/config.toml` | Add or replace | Theme and app configuration |
| `.streamlit/secrets.toml.example` | Add | Credential example containing placeholders only |
| `.gitignore` | Add or merge | Keeps local secrets and temporary files out of GitHub |
| `README.md`, `START_HERE.md` | Add | Project information and beginner setup guide |
| `CHATBOT.md` | Add | Assistant implementation, provider, pricing, and research notes |
| `tests/` | Add | Model, CSV, assistant transport, and app integration checks |
| `VERIFICATION.md` | Add | What was tested and remaining account-specific checks |
| `requirements.txt` | Retain supplied pinned versions | Python dependencies compatible with the saved models |
| `core.py` | Retain unchanged | Eight-input schema, preprocessing, validation, scoring, CSV protection |
| `model_bundle.joblib` | Retain unchanged | Fitted linear and logistic prediction pipelines |
| `metrics.json` | Retain unchanged | Matching feature schema, states, benchmark, model errors, versions |
| `dashboard_data.json` | Retain unchanged | Existing aggregate dataset summaries |
| `sample_batch.csv` | Retain unchanged | Working input template with example rows |
| `study_tools.py` | Remove | Retired manager-study functionality |
| `.streamlit/secrets.toml` | Create locally only | Your real credentials; never upload this file |

Keep `model_bundle.joblib`, `metrics.json`, and `core.py` together.

## Run locally

Use Python 3.12 to match the original training environment. Open a terminal in the folder containing `app.py`.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS or Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`. Press `Ctrl+C` in the terminal to stop it. Prediction pages work without an AI key.

## Enable the AI assistant

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace only the key placeholder. Alternatively, set the equivalent environment variables on the server. In Streamlit Community Cloud, use the app's **Settings → Secrets**, not GitHub.

```toml
OPENAI_API_KEY = "replace-with-your-own-key"
OPENAI_MODEL = "gpt-5.4-mini"
```

The key is read on the server. It is not collected in the public app. OpenAI API usage is billed separately from a ChatGPT subscription. See the official model and pricing links in the setup guide before enabling public use.

The assistant uses OpenAI's API over HTTPS through Python's standard library. No extra AI SDK or change to the prediction dependencies is required. The model name is configurable; changing providers requires adapting the API integration.

### What the assistant can see

It sends the current question, a limited recent conversation, and app instructions to OpenAI. It includes the latest submitted individual estimate and its eight input values only when you select **Share my last submitted estimate with the AI**. The box starts unchecked and is unavailable before you save an estimate. Open **Review the profile and result that will be sent** to inspect the context first. It does not automatically send uploaded batch files, names, phone numbers, or record IDs. Select **Clear conversation** to remove the visible chat history and uncheck estimate sharing. Clearing a local session does not delete requests already processed by the API provider. Requests use `store=false`, which disables stored response objects but does not imply zero retention; provider abuse-monitoring rules can still apply. See [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data).

### Immediate replies and current information

This release has conversational AI, without a connected weather, mandi-price, government-scheme, or web-search feed. The assistant must say that it cannot verify current information. It can explain a concept or help interpret a source you provide, but it cannot claim today's prices or weather. A future live-data integration must show its source, retrieval date, and relevant location.

If credentials are missing or the service fails, the app explains the problem and leaves the prediction sections usable. There are no fabricated chatbot answers or simulated model results.

## Inputs and interpretation

| CSV column | Meaning and unit |
| --- | --- |
| `state` | A supported state; original spellings remain in the model metadata |
| `land_area` | Agricultural land area in the original dataset's unspecified unit |
| `non_ag_income` | Known non-farm income in INR; required, enter zero only when truly zero |
| `mandi_distance` | Distance to the nearest mandi in km |
| `kharif_rainfall` | Kharif rainfall in mm, corresponding to the model's 2022 feature |
| `rabi_rainfall` | Rabi rainfall in mm, corresponding to the model's 2022 feature |
| `cropping_density` | Original 2022 source index; do not substitute an assumed percentage |
| `agri_score` | Original 2022 agricultural score index |

The template uses friendly headers; the original source headers are also accepted. Batch files support up to 15,000 rows and a 25 MB upload limit. The template contains three hypothetical profiles. Downloads use source row numbers to match inputs and exclude loan IDs and other unused fields. Invalid data is identified before scoring. Missing numeric inputs may be filled with training medians, except required non-farm income. Unknown states, negative or infinite values, and invalid text in numeric fields are rejected. Out-of-training-range values produce warnings.

The estimate is **total income, including farm and non-farm income**. The source does not define its monthly or annual period, nor the land unit. Use source-consistent values. The comparison benchmark comes from the dataset; it is not a poverty line or an eligibility rule. The displayed error band comes from saved model validation and is not a guarantee for an individual. The chatbot explains stored outputs; it does not calculate replacement predictions.

Keep personal identifiers out of uploads. Downloaded predictions retain the existing identifier safeguards and spreadsheet-formula protection.

## Verification and limits

[VERIFICATION.md](VERIFICATION.md) records the checks performed. **54 automated tests passed:** 18 prediction checks, 11 CSV checks, 18 mocked assistant transport checks, and 7 Streamlit AppTest integration checks. The original trained artifacts were loaded, their preservation checked, and three template predictions matched their baseline outputs exactly.

No API key was supplied, so real authenticated OpenAI replies, account billing, and model access remain unverified. Mocked streams verify transport and error behaviour; they do not establish that a real API account works. The original training data is not bundled, so training/holdout metrics were not reevaluated.

See the verification report for browser inspection details. After deployment, run the checklist in `START_HERE.md` with your credentials and target devices.

## Official references

Instructions checked against official documentation on 18 October 2026:

- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Streamlit app settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings)
- [Streamlit management and Cloud logs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [OpenAI API keys](https://platform.openai.com/api-keys)
- [OpenAI API pricing](https://openai.com/api/pricing/)
