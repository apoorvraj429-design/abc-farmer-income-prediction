# Start here: update your Farmer Income app

The new app is called **Kisan Saathi**. Your original prediction model is retained. The update replaces the three study sections with an AI assistant and improves the design.

You can first test the app without paying for AI. The estimator and CSV predictions work without an API key.

## 1. Understand the five basic terms

- **Repository:** your project folder stored on GitHub, including its change history.
- **Dependency:** a Python package the app needs, such as Streamlit or pandas. `requirements.txt` lists the exact versions.
- **Terminal:** the window where you enter commands. On Windows, use PowerShell.
- **API:** a way for your app to ask an external service, such as OpenAI, to do work. An API key is a private password authorising these requests.
- **Deployment:** running your app on an online server so other people can open its web address.

A **branch** is a separate line of changes. A **commit** is a saved set of changes. A **pull request** is a review page proposing to merge a branch into the app's deployed branch.

## 2. Back up the working version

1. Open [your GitHub repository](https://github.com/apoorvraj429-design/abc-farmer-income-prediction).
2. Select the branch your existing app uses, normally `main`.
3. Click **Code → Download ZIP**. Save it as your old-version backup and extract a separate copy if needed.
4. Note the current commit ID from GitHub's history. Keep the original ZIP separate from the new deployment package.
5. In [Streamlit Community Cloud](https://share.streamlit.io), note the app's repository, branch, main file path, Python version, and current URL. Save existing secrets privately; they are not part of a GitHub ZIP.

## 3. Prepare the new files

Extract the new deployment ZIP. Open the folder that directly contains `app.py`; this is the project root. The file-action table in `README.md` explains every item.

Use this new guide and deployment ZIP for the redesign. Your attached Colab notebook remains a training reference, but its original `APP_SOURCES` export still creates the previous six-section interface. Do not overwrite this redesign with that older exported `app.py`.

Replace `app.py`; add `assistant.py`, `batch_tools.py`, documentation (`README.md`, this guide, `CHATBOT.md`, and `VERIFICATION.md`), `tests/`, the supplied assets, and `.streamlit` configuration. Remove `study_tools.py`. Retain the original `core.py`, `model_bundle.joblib`, `metrics.json`, `dashboard_data.json`, `sample_batch.csv`, and pinned `requirements.txt`. The complete deployment ZIP includes the necessary retained files as well.

Do not put a second whole project folder inside the existing repository. On GitHub, `app.py` must appear directly at the path Streamlit already uses. Keep the `assets/` and `.streamlit/` subfolders and their contents together. File and folder names are case-sensitive on the server.

## 4. Test on your computer

Install Python 3.12 from [python.org](https://www.python.org/downloads/) if it is unavailable. This matches the original model's Python version family.

On Windows, open the extracted project folder in File Explorer. Right-click its empty area and choose **Open in Terminal** if available, or open PowerShell and use `cd` to enter the folder. For example:

```powershell
cd "C:\Users\YOUR_NAME\Downloads\Kisan_Saathi"
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Replace the example path with the folder containing your `app.py`. Run one command at a time. The `.venv` folder is a private environment for this project's installed packages. These commands do not need environment activation, so PowerShell execution-policy changes are unnecessary.

On macOS or Linux, open Terminal in the project root:

```bash
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

When a local address appears, open it in your browser. Keep the terminal running while using the app. Press `Ctrl+C` to stop it.

## 5. Obtain and configure your AI key

OpenAI is the supplied integration because its API supports conversational text and streaming, and the implementation keeps your ML dependencies unchanged.

1. Open [OpenAI's API platform](https://platform.openai.com/), sign in, and select or create an API project.
2. Review [current API pricing](https://openai.com/api/pricing/) and configure API billing in your account. A ChatGPT subscription does not itself pay for API requests.
3. Create a secret key on the [API keys page](https://platform.openai.com/api-keys). Copy it privately when shown.
4. Inside the project's `.streamlit` folder, copy `secrets.toml.example` and name the copy exactly `secrets.toml`. Ensure Windows has not appended `.txt`.
5. Put your key in the local copy:

```toml
OPENAI_API_KEY = "replace-with-your-own-key"
OPENAI_MODEL = "gpt-5.4-mini"
```

6. Restart the local Streamlit process if necessary. Open **AI assistant** and send a short question.

`secrets.toml.example` is safe to share because it contains placeholders. Your real `secrets.toml` must stay private. The supplied `.gitignore` excludes it, but GitHub's manual upload still requires you to check the file list yourself. Never paste the key into `app.py`, the public app, GitHub, or screenshots. If a key is exposed, revoke it in the API platform and create a replacement.

The app also accepts `OPENAI_API_KEY` and `OPENAI_MODEL` as server environment variables. Choose one setup method to keep configuration simple.

### Costs and a cheaper option

AI requests are billed using tokens, small pieces of text. Requests include recent chat and app instructions as well as your question. Official model pages checked on 18 October 2026 give these standard text rates in USD:

| Configurable model | Input per 1 million tokens | Output per 1 million tokens | Role |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | $0.75 | $4.50 | Supplied default |
| `gpt-6-luna` | $0.10 | $0.50 | Lower-cost alternative to try |

For illustration, 2,000 input tokens plus 500 output tokens cost about **$0.00375** with the default, or **$0.00045** with the cheaper option. Actual charges depend on usage, history, and applicable billing terms; check model access and current prices in your account. The supplied output cap is 1,600 tokens per answer. To try the cheaper model, set `OPENAI_MODEL = "gpt-6-luna"`, save, and test its explanations before relying on it.

Limits inside one browser session do not impose an account-wide spending ceiling. Check API usage and account controls, especially before sharing a public app.

Official model references: [GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini) and [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

## 6. Save the update to GitHub

If a prepared pull request is included with this delivery, open it and review the files and test report. If it is marked Draft, choose **Ready for review**, then merge it when you are ready to update the branch used by Streamlit. A draft pull request itself does not change your live app.

For a manual upload:

1. Open your repository and create a branch, for example `kisan-saathi-redesign`, from `main` using GitHub's branch selector.
2. Choose **Add file → Upload files**. Upload the new project's files and folders, preserving their paths. Drag the **contents** of the extracted project root, not the outer project folder and not the ZIP.
3. Check that `assets/` appears as a folder with its files. If the upload interface omits `.streamlit`, use **Add file → Create new file** and enter the full name `.streamlit/config.toml`, then paste the supplied contents. Repeat for `.streamlit/secrets.toml.example`; these are configuration examples, not your actual key.
4. Do not upload `.venv`, `__pycache__`, `.git`, or your real `.streamlit/secrets.toml`. Check for them before committing.
5. Save the upload as a commit. Delete `study_tools.py` from this branch using its file menu and commit that deletion too.
6. Open a pull request from your update branch into the deployed branch. Confirm the Files changed page includes the intended files and no credentials.
7. Merge after review. This is the point at which the existing Streamlit deployment will receive the code update.

Keep one release's model bundle and metadata together; do not mix old and new model files. No new notebook training run is required for this redesign.

## 7. Configure the online app

For your existing deployment, sign in to [Streamlit Community Cloud](https://share.streamlit.io) with the GitHub account that can manage this app. Select the workspace for `apoorvraj429-design`. Use the app's **three-dot menu → Settings → Secrets**. Paste the two TOML lines above, with your actual private key, and click **Save**. Secrets are configured separately from the GitHub files.

If the app monitors `main`, merging into `main` triggers its update automatically. A dependency change can take longer because the server reinstalls packages. Keep the existing app and its URL; there is no need to create another app for an ordinary code update.

For a separate preview or new deployment, choose **Create app**, use the existing-app path, and select:

| Setting | Value |
| --- | --- |
| Repository | `apoorvraj429-design/abc-farmer-income-prediction` |
| Branch | Your update branch for a preview, or `main` after merge |
| Main file path | `app.py` |
| Python | `3.12` in Advanced settings |
| Secrets | Your private configuration from step 5 |

If your existing app runs another Python version, leave it alone if the pinned dependencies work. Streamlit's current documentation says changing the Python interpreter itself requires a redeployment. That is separate from this code update; record settings and secrets before planning it.

## 8. Use and check the new app

1. Confirm exactly four navigation entries: Overview, Income estimator, Batch predictions, AI assistant.
2. In Income estimator, try the labelled example and calculate an estimate. Example values are a demonstration, not your own farm information.
3. Read the result note: total income in INR, original income period unspecified, original land unit unspecified. Keep the source's units when entering real values.
4. Download the batch template, preview it, and generate predictions. Keep uploads within 15,000 rows and 25 MB. Then try an unsupported state or a negative value and confirm it is flagged.
5. Ask the assistant “What does non-farm income mean?” Confirm a real streamed answer after adding your key.
6. To discuss a result, select **Share my last submitted estimate with the AI**. It starts unchecked and is disabled until you have submitted an estimate. Open **Review the profile and result that will be sent** to inspect the inputs and result first. Uploaded batch data is not attached automatically.
7. Select **Clear conversation** and confirm the history clears and sharing becomes unchecked. Refreshing or closing a browser session may also reset history.
8. Temporarily remove the key in local testing. The AI page should explain that setup is needed while both prediction sections keep working. Restore it afterwards.
9. Ask for today's local mandi price. This release has no live-data connection, so the assistant should explain that it cannot verify the current price.
10. Open the app on your phone and check navigation, forms, results, and CSV instructions. Verify your deployed app after merging, even if local checks passed.

A failed AI request should show a useful message without revealing the key. This release passed 54 automated checks covering preserved predictions, batch validation, mocked API transport, and Streamlit integration. The three template predictions matched their original saved-model outputs exactly; this redesign does not require retraining. See [VERIFICATION.md](VERIFICATION.md) for details.

No real API key was available for implementation testing, so use this checklist to finish the authenticated AI check. The local source and model artifacts were verified; the remote deployed page could not be inspected through web retrieval.

## 9. Read errors and fix common problems

While signed in, open the deployed app and click **Manage app** at the lower right. The panel shows Cloud logs: a technical record of startup and errors. Its menu includes **Download log**, **Settings**, and **Reboot app**. Logs require the appropriate repository permissions. Locally, read the terminal where Streamlit is running.

| What you see | First thing to check |
| --- | --- |
| `ModuleNotFoundError` | `requirements.txt` is present at the expected path and installation completed |
| Model-version warning or failure | Original pinned package versions and Python family; do not casually upgrade scikit-learn |
| Missing model or metadata | `model_bundle.joblib` and `metrics.json` sit beside `app.py` and `core.py` |
| Photos missing | Preserve `assets/` names and paths; the app should still show a text fallback |
| Assistant needs configuration | Save the private key in the deployed app's Secrets tab, not only on your computer |
| Authentication/401 failure | Key is valid, active, and copied without extra whitespace |
| Rate limit/429 or quota failure | Check billing, usage, and provider limits; retry later where appropriate |
| Model not found or unavailable | Check model access and `OPENAI_MODEL`; use the verified supported default |
| Timeout/service error | Retry later and check connectivity or provider status |
| Old design remains | You merged into the branch that the deployed app actually monitors; wait for rebuilding, then reload |
| Invalid CSV | Use template headers, supported state spellings, finite non-negative numbers, and required non-farm income |

Fix the cause before repeated reboots. Copy the error message if you need help, and remove credentials or personal records from any shared log.

## 10. Restore the previous version

If you used a pull request, GitHub may offer **Revert** on the merged pull request; review and merge the resulting reversal. This creates a new commit restoring the earlier code without deleting the history. If that option is unavailable, restore the old files from your saved backup as a new commit, including the previous `app.py` and `study_tools.py`, and remove the redesign-only files.

The monitored branch will deploy the restored files automatically. The model and metadata were preserved, so they do not need retraining. Recheck app secrets separately: a code rollback does not undo changes made in Streamlit's Secrets settings. Keep your backup until the new deployed version has passed your checks.

## Official help

Interface instructions were checked on 18 October 2026. Buttons may move as services change.

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Manage secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [App settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings)
- [Cloud logs and updates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Changing Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)
- [OpenAI API keys](https://platform.openai.com/api-keys)
- [OpenAI pricing](https://openai.com/api/pricing/)
