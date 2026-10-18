"""Kisan Saathi: historical income estimates and a connected AI assistant.
Run with: python -m streamlit run app.py
"""
import base64
import html
import io
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import (ROOT, NUM, FEATURES, LABELS, UNITS, load_artifacts,
                  validate_inputs, score, contributions, safe_csv, inr)
from batch_tools import inspect_rows, read_batch, BatchInputError
from assistant import load_config, configured, stream_reply, AssistantError

st.set_page_config(page_title="Kisan Saathi | Farmer income", page_icon="🌾", layout="wide")
PAGES = ["Overview", "Income estimator", "Batch predictions", "AI assistant"]
GREEN = "#24583e"
GOLD = "#b78731"

style_path = ROOT / "assets" / "styles.css"
if style_path.exists():
    st.markdown("<style>" + style_path.read_text() + "</style>", unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_dashboard():
    return json.loads((ROOT / "dashboard_data.json").read_text())


@st.cache_data
def photo_data(filename):
    path = ROOT / "assets" / filename
    if not path.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


try:
    BUNDLE, META = get_artifacts()
    MODEL_ERROR = None
except Exception:
    BUNDLE, META = None, None
    MODEL_ERROR = "The trained model files could not be loaded. Check the deployment files and pinned package versions in START_HERE.md."
try:
    DASH = get_dashboard()
except (OSError, ValueError):
    DASH = {}


def go(page):
    st.session_state["workspace"] = page


def heading(kicker, title, description):
    st.markdown(f'<p class="eyebrow">{html.escape(kicker)}</p>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)


def tile(number, title, copy):
    st.markdown(f'<div class="feature-tile"><span class="step-number">{number}</span>'
                f'<h3>{html.escape(title)}</h3><p>{html.escape(copy)}</p></div>', unsafe_allow_html=True)


def require_model():
    if MODEL_ERROR:
        st.error(MODEL_ERROR)
        return False
    return True


def styled(chart, height=270):
    return chart.properties(height=height).configure(
        background="transparent", view={"stroke": None},
        axis={"labelColor": "#56675a", "titleColor": "#263e2c", "labelFontSize": 13,
              "titleFontSize": 13, "gridColor": "#e6e8dd", "domain": False},
        legend={"labelFontSize": 13, "titleFontSize": 13})


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">🌾</span><div>Kisan Saathi'
                '<small>FARMER INCOME COMPANION</small></div></div>', unsafe_allow_html=True)
    st.radio("Explore Kisan Saathi", PAGES, key="workspace", label_visibility="collapsed")
    st.divider()
    st.markdown("**A little clarity for your next decision.**")
    st.caption("Estimate a comparable profile’s total income, work with a CSV, or ask a question.")
    st.markdown('<div class="sidebar-note">Historical data prototype<br>Income in Indian rupees</div>', unsafe_allow_html=True)
    st.caption("Built by Apoorv Raj · ABC Limited project")


def overview():
    image = photo_data("farmer_wheat.jpg")
    photo = (f'<img class="hero-photo" src="{image}" alt="A woman farmer tending a wheat field in India"/>'
             if image else '<div class="photo-fallback">🌾<br>Rooted in agriculture</div>')
    st.markdown('<div class="hero"><div class="hero-copy"><p class="eyebrow">YOUR FARMER INCOME COMPANION</p>'
                '<h1>Understand income.<br>Plan with perspective.</h1>'
                '<p>A simple place to estimate total farmer income, explore profiles, and ask questions along the way.</p>'
                '<div class="hero-note">Historical estimates · Farm + non-farm income · INR</div></div>'
                f'<div class="hero-visual">{photo}<span class="photo-label">People at the heart of agriculture</span></div></div>',
                unsafe_allow_html=True)
    a, b, _ = st.columns([1.2, 1.2, 2.3])
    a.button("Estimate a profile", type="primary", use_container_width=True, on_click=go, args=("Income estimator",))
    b.button("Ask the AI assistant", use_container_width=True, on_click=go, args=("AI assistant",))
    st.caption("Photograph: EqualStock IN / Pexels. Used under the Pexels License.")
    st.markdown("## Choose what you need")
    columns = st.columns(3, gap="medium")
    items = [("01", "One profile", "Enter eight inputs and see an income estimate with an explanation.", "Income estimator"),
             ("02", "Many profiles", "Upload a CSV, check its rows, and download the predictions.", "Batch predictions"),
             ("03", "A conversation", "Ask about the app, farming, business, or a result you choose to share.", "AI assistant")]
    for col, (n, title, text, destination) in zip(columns, items):
        with col:
            tile(n, title, text)
            st.button("Open " + destination.lower(), key="open_" + n, use_container_width=True,
                      on_click=go, args=(destination,))
    st.markdown('<div class="context-note"><b>Before you estimate</b><br>This model needs known non-farm income, '
                'which is already part of the total it estimates. It uses historical inputs, including 2022 indicators. '
                'The source does not specify an income period or the land-area unit.</div>', unsafe_allow_html=True)
    if META and DASH.get("scopes"):
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            st.subheader("A little context from the dataset")
            scopes = list(DASH["scopes"])
            scope = st.selectbox("Dataset group", scopes)
            hist = pd.DataFrame(DASH["scopes"][scope]["histogram"])
            chart = alt.Chart(hist).mark_bar(color=GREEN, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("band:N", sort=hist.band.tolist(), title="Recorded total income (INR)", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("count:Q", title="Records"), tooltip=["band:N", "count:Q"])
            st.altair_chart(styled(chart), use_container_width=True)
            st.caption("Recorded income patterns in the source dataset. These do not represent all Indian farmers.")
        with right:
            st.subheader("What to have ready")
            st.markdown("- State and land area in the original source units.\n- Known non-farm income for the same period.\n- Distance to mandi and rainfall.\n- Historical cropping-density and agricultural-score indices.")
            with st.expander("What does total income mean here?"):
                st.write("It includes agricultural and non-agricultural income in INR. The source does not establish whether the period is monthly or annual. Rainfall and agricultural indicators refer to 2022. This is not a validated forecast of next season’s income.")
                st.write("The dataset contains loan-linked observations. Record counts should not be described as verified counts of unique farmers.")
    elif MODEL_ERROR:
        st.info(MODEL_ERROR)


def load_example():
    if not META:
        return
    demo = DASH.get("profiles", [{}])[0].get("inputs")
    if demo is None:
        demo = {"state": "MAHARASHTRA", **{c: META["schema"][c]["median"] for c in NUM}}
    for field in FEATURES:
        st.session_state["input_" + field] = demo[field]
    st.session_state["example_loaded"] = True


def result_panel(row):
    result = score(row, BUNDLE, META).iloc[0]
    st.markdown('<div class="result-card"><span>ESTIMATED TOTAL INCOME</span>'
                f'<h2>{inr(result.predicted_total_income_inr)}</h2>'
                f'<p>₹{result.predicted_total_income_inr:,.0f} · farm + non-farm income</p></div>', unsafe_allow_html=True)
    st.markdown(f'**Empirical error range**  \n{inr(result.range_low_inr)} to {inr(result.range_high_inr)}')
    st.caption(f'This validation-based band covered {META["error_band_holdout_coverage"]:.1%} of held-out records. It is not a guarantee for a particular person.')
    p = float(result.probability_below_benchmark)
    st.progress(p, text=f'{p:.1%} chance of total income below {inr(META["benchmark_inr"])}')
    st.caption("A separate logistic model provides this probability. Its benchmark is the training median, not a poverty line or an eligibility rule.")
    if result.below_known_non_ag_income:
        st.warning("The total estimate falls below the known non-farm component. This inconsistent result needs manual review.")
    if result.predicted_total_income_inr > META["train_target_p99"] or row.non_ag_income.iloc[0] > META["schema"]["non_ag_income"]["p99"]:
        st.warning("This is an extreme-income profile. The general error range may understate uncertainty.")
    with st.expander("What contributed to this estimate?"):
        parts, intercept = contributions(row, BUNDLE)
        parts["contribution_lakh"] = parts.contribution_inr / 1e5
        chart = alt.Chart(parts).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("contribution_lakh:Q", title="Contribution to raw estimate (₹ lakh)"),
            y=alt.Y("factor:N", sort="-x", title=None),
            color=alt.condition(alt.datum.contribution_inr >= 0, alt.value(GREEN), alt.value(GOLD)),
            tooltip=["factor", alt.Tooltip("contribution_inr:Q", format=",.0f")])
        st.altair_chart(styled(chart, 310), use_container_width=True)
        st.write(f'The intercept is ₹{intercept:,.0f}. Adding the contributions produces the raw estimate of ₹{result.raw_linear_prediction_inr:,.0f}. Negative raw estimates are displayed as zero.')
        st.caption("These learned associations do not show that changing a factor will cause income to change.")
    return result


def estimator():
    heading("ONE PROFILE", "Farmer income estimator", "Enter a comparable profile. Use the original data definitions and the same reporting period for all income values.")
    if not require_model():
        return
    st.info("Known non-farm income is required. Enter zero only when it is truly zero. Other blank numeric fields use the training median and will be flagged.")
    st.button("Load a hypothetical example", on_click=load_example)
    if st.session_state.get("example_loaded"):
        st.caption("Example starting values are loaded. Review or replace them before estimating. They are not an observed farmer outcome.")
    inputs, output = st.columns([1.1, 1], gap="large")
    with inputs:
        with st.form("income_form"):
            st.subheader("Farmer and location")
            values = {"state": st.selectbox("State", META["states"], index=None, placeholder="Select a state", format_func=str.title, key="input_state")}
            fields = {
                "land_area": ("Agricultural land area (source units)", "The source does not confirm acres or hectares. Use its original land-area unit."),
                "non_ag_income": ("Known non-farm income (₹)", "Required. Use the same reporting period as total income. Unknown is different from zero."),
                "mandi_distance": ("Distance to nearest mandi (km)", "Distance from the source’s Kharif 2022 proximity field."),
                "kharif_rainfall": ("Kharif rainfall (mm)", "Seasonal average from the 2022 source field."),
                "rabi_rainfall": ("Rabi rainfall (mm)", "Seasonal average from the 2022 source field."),
                "cropping_density": ("Cropping-density index (2022)", "Use the original source index. Its scale is not established as a percentage."),
                "agri_score": ("Agricultural score (2022)", "Historical source score. Its scale is not fully documented.")}
            a, b = st.columns(2)
            for i, field in enumerate(["land_area", "non_ag_income", "mandi_distance"]):
                with a if i % 2 == 0 else b:
                    label, help_text = fields[field]
                    values[field] = st.number_input(label, min_value=0.0, value=None, step=1000.0 if field == "non_ag_income" else 0.1,
                                                    key="input_" + field, help=help_text, placeholder="Enter value")
            st.subheader("Rainfall and agricultural indicators")
            a, b = st.columns(2)
            for i, field in enumerate(NUM[3:]):
                with a if i % 2 == 0 else b:
                    label, help_text = fields[field]
                    values[field] = st.number_input(label, min_value=0.0, value=None, step=0.1,
                                                    key="input_" + field, help=help_text, placeholder="Enter value")
            submitted = st.form_submit_button("Estimate income", type="primary", use_container_width=True)
        if submitted:
            try:
                row, warnings = validate_inputs(pd.DataFrame([values]), META)
                st.session_state["last_income"] = {"row": row, "warnings": warnings}
            except ValueError as exc:
                st.error(str(exc))
    with output:
        if "last_income" not in st.session_state:
            st.markdown('<div class="empty-result"><span>🌱</span><h3>Your estimate starts here</h3>'
                        '<p>Enter a profile or load the example, then select Estimate income.</p></div>', unsafe_allow_html=True)
        else:
            saved = st.session_state["last_income"]
            row = saved["row"]
            st.caption("Result for your last submitted profile. Unsubmitted edits do not change it.")
            for warning in saved["warnings"]:
                st.warning(warning)
            result_panel(row)
            out = pd.concat([row.reset_index(drop=True), score(row, BUNDLE, META)], axis=1)
            st.download_button("Download this estimate", safe_csv(out), "kisan_saathi_estimate.csv", "text/csv", use_container_width=True)
            st.button("Discuss this estimate with AI", use_container_width=True, on_click=go, args=("AI assistant",))
    st.caption("The model learns from historical records and depends strongly on knowing the non-farm component of total income. It has not demonstrated reliable current-season forecasting.")


def batch():
    heading("MANY PROFILES", "Batch predictions", "Upload a CSV, review its inputs, and download income estimates for every valid row.")
    if not require_model():
        return
    a, b = st.columns([2, 1])
    a.write("Use the eight-column template or the original source headers. Maximum 15,000 rows and 25 MB. Files stay in this app session and are not sent to the AI assistant.")
    if (ROOT / "sample_batch.csv").exists():
        b.download_button("Download input template", (ROOT / "sample_batch.csv").read_bytes(), "sample_batch.csv", "text/csv", use_container_width=True)
    with st.expander("Required columns and units"):
        definitions = [{"Column": field, "Meaning": LABELS[field], "Unit": "category" if field == "state" else UNITS[field]} for field in FEATURES]
        st.dataframe(pd.DataFrame(definitions), hide_index=True, use_container_width=True)
        st.write("The template contains hypothetical examples. Known non-farm income is mandatory. Blank values in the other numeric fields are filled using training medians and reported. The income period and land-area unit are not specified by the source.")
    upload = st.file_uploader("Choose a farmer input CSV", type=["csv"])
    if upload is None:
        return
    try:
        raw = read_batch(upload)
        x, issues, warnings = inspect_rows(raw, META)
    except (BatchInputError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(str(exc))
        return
    st.subheader("Input preview")
    st.dataframe(raw[FEATURES].head(25) if all(c in raw for c in FEATURES) else x.head(25), hide_index=True, use_container_width=True)
    st.caption("Up to 25 rows are shown. Unused columns and loan identifiers are excluded from this preview and prediction downloads.")
    if not issues.empty:
        st.error(f"Found {len(issues):,} input issue(s). Correct the listed rows and upload the CSV again. No rows have been silently dropped.")
        st.dataframe(issues.head(100), hide_index=True, use_container_width=True)
        st.download_button("Download input issues", safe_csv(issues), "input_issues.csv", "text/csv")
        return
    for warning in warnings:
        st.warning(warning)
    pred = score(x, BUNDLE, META)
    out = pred.copy()
    out.insert(0, "source_row", np.arange(1, len(out) + 1))
    a, b, c = st.columns(3)
    a.metric("Profiles estimated", f"{len(out):,}")
    b.metric("Median predicted income", inr(out.predicted_total_income_inr.median()))
    c.metric("Below sample benchmark", f"{out.predicted_below_benchmark.mean():.1%}")
    st.subheader("Prediction preview")
    st.dataframe(out.head(100), hide_index=True, use_container_width=True)
    st.caption("First 100 results. Source row 1 is the first data row after the header. Downloads contain all scored rows, without loan IDs. Without observed incomes, this upload has no accuracy score.")
    st.download_button("Download all predictions", safe_csv(out), "kisan_saathi_batch_predictions.csv", "text/csv", type="primary")
    st.caption(f"The comparison benchmark is {inr(META['benchmark_inr'])}, the training median. The empirical range is broad and does not guarantee an individual outcome.")


def estimate_context():
    saved = st.session_state.get("last_income")
    if saved is None or not META or not BUNDLE:
        return None
    row = saved["row"]
    result = score(row, BUNDLE, META).iloc[0]
    values = {field: (None if pd.isna(row.iloc[0][field]) else str(row.iloc[0][field]) if field == "state" else float(row.iloc[0][field])) for field in FEATURES}
    return {"inputs": values, "prediction": {
        "predicted_total_income_inr": float(result.predicted_total_income_inr),
        "range_low_inr": float(result.range_low_inr), "range_high_inr": float(result.range_high_inr),
        "probability_below_benchmark": float(result.probability_below_benchmark),
        "benchmark_inr": float(META["benchmark_inr"])},
        "notes": "Total income is farm plus non-farm income, in INR. Reporting period and land units unspecified. Indicators include 2022 data. Range is empirical, no individual guarantee. Benchmark is training median. Blank numeric inputs use training medians."}


def clear_conversation():
    st.session_state["messages"] = []
    st.session_state["share_estimate"] = False
    st.session_state.pop("pending_prompt", None)


def ai_assistant():
    heading("A CONVERSATION", "AI assistant", "Ask a question in plain language. You can ask in English or Hindi.")
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    config = load_config(secrets)
    ready = configured(config)
    st.caption("AI replies through OpenAI · No connected live weather, mandi-price, or web-search feed")
    if not ready:
        st.info("The AI assistant is ready to connect. The app owner needs to add OPENAI_API_KEY in Streamlit Settings → Secrets. Income predictions work without it.")
        with st.expander("Connection instructions for the app owner", expanded=True):
            st.write("An API key is a private credential that lets this app call the AI service. Create it in your own OpenAI API account, set a usage budget, then paste the configuration below into the app’s Secrets settings. Keep real keys out of GitHub.")
            st.code('OPENAI_API_KEY = "paste-your-private-key-here"\nOPENAI_MODEL = "gpt-5.4-mini"', language="toml")
            st.write("Full instructions and current pricing links are in START_HERE.md. ChatGPT subscription billing and API billing are separate. This screen does not simulate AI replies.")
    else:
        st.success("AI service configured. Send a message to check the connection.", icon="✅")
    context = estimate_context()
    share = st.checkbox("Share my last submitted estimate with the AI", value=False, disabled=context is None, key="share_estimate",
                        help="Sends only the single profile and result shown below with each new message. Batch uploads are never included.")
    if context is None:
        st.caption("Make an estimate first to enable sharing. General questions do not need a profile.")
    elif share:
        with st.expander("Review the profile and result that will be sent", expanded=False):
            st.json(context)
        st.caption("This single profile will be sent to OpenAI with your message. Turning sharing off does not erase earlier conversation text; use Clear conversation to remove that text from this session.")
    left, right = st.columns([3, 1])
    left.markdown("**Try a question**")
    right.button("Clear conversation", use_container_width=True, on_click=clear_conversation)
    starters = ["What inputs do I need?", "Explain the income error range.", "How can I compare farm business options?"]
    columns = st.columns(3)
    for i, question in enumerate(starters):
        if columns[i].button(question, disabled=not ready, use_container_width=True, key="starter_" + str(i)):
            st.session_state["pending_prompt"] = question
    messages = st.session_state.setdefault("messages", [])
    if not messages:
        with st.chat_message("assistant", avatar="🌾"):
            st.write("Hello! I’m Kisan Saathi’s AI assistant. Ask about this app, farming, business, or a general topic. I can explain your actual income estimate when you choose to share it.")
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    prompt = st.chat_input("Ask Kisan Saathi a question…", disabled=not ready, max_chars=4000)
    prompt = prompt or st.session_state.pop("pending_prompt", None)
    if prompt and ready:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant", avatar="🌾"):
            chunks = []
            def collect():
                for chunk in stream_reply(messages, config, prediction_context=context if share else None):
                    chunks.append(chunk)
                    yield chunk
            try:
                with st.spinner("Thinking…"):
                    st.write_stream(collect())
                response = "".join(chunks)
                messages.append({"role": "assistant", "content": response})
            except AssistantError as exc:
                if chunks:
                    st.warning("The response stopped early. Please try again.")
                st.error(str(exc))
                # Do not retain a partial response as a complete answer.
                messages.pop()
            except Exception:
                st.error("The AI service could not finish this reply. Please try again. Income prediction tools remain available.")
                messages.pop()
        st.session_state["messages"] = messages[-20:]


{"Overview": overview, "Income estimator": estimator, "Batch predictions": batch,
 "AI assistant": ai_assistant}[st.session_state["workspace"]]()
st.markdown('<div class="footer">Kisan Saathi · Historical income estimates for comparable profiles · Built by Apoorv Raj</div>', unsafe_allow_html=True)
with st.expander("Photo credits"):
    st.markdown("Hero: [EqualStock IN / Pexels](https://www.pexels.com/photo/farmer-in-india-20445181/) · [Pexels License](https://www.pexels.com/license/). Additional photograph: [Siddharth Patil / Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Indian_farmers.jpg) · CC0 1.0.")
