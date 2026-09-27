"""ABC Limited Farmer Income Lab. Run: streamlit run app.py"""
import base64
from datetime import datetime, timezone
import html
import io
import json
import uuid
import zipfile
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from core import ROOT,NUM,FEATURES,LABELS,UNITS,load_artifacts,validate_inputs,score,contributions,safe_csv,inr
from study_tools import (STUDY_VERSION,RATINGS,OPEN,THEMES,BANDS,COLUMNS,merge_responses,
    rating_summary,decision_summary,coding_template,theme_summary)

st.set_page_config(page_title='ABC Limited | Farmer Income Lab',page_icon='🌱',layout='wide')
REQUIRED=['core.py','study_tools.py','model_bundle.joblib','metrics.json','dashboard_data.json','sample_batch.csv']
missing=[p for p in REQUIRED if not (ROOT/p).exists()]
if missing:
    st.error('Some project files were not uploaded: '+', '.join(missing))
    st.info('Upload every file from the extracted app folder into the same GitHub folder as app.py, then reboot the app.')
    st.stop()

@st.cache_resource
def artifacts():return load_artifacts()
@st.cache_data
def charts():return json.loads((ROOT/'dashboard_data.json').read_text())
BUNDLE,M=artifacts();D=charts()
GREEN='#1e604a';GOLD='#c79537';INK='#19382d';MUTED='#64776d';PALE='#e8efe8'
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#19382d;--muted:#64776d;--green:#1e604a;--cream:#f4f6ef}
.stApp,[data-testid="stAppViewContainer"]{background:var(--cream);color:var(--ink);font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:rgba(244,246,239,.94)}
.block-container{padding-top:2.6rem;padding-bottom:3rem;max-width:1450px}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;color:var(--ink)!important;letter-spacing:-.035em}
h2{font-size:1.65rem!important}h3{font-size:1.2rem!important}
[data-testid="stMain"] p,[data-testid="stMain"] label,[data-testid="stMain"] li{color:var(--ink)}
[data-testid="stSidebar"]{background:#e8eee4;border-right:1px solid #d4dfd2}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{color:#244434}
[data-testid="stSidebar"] .block-container{padding-top:2rem}
.brand{display:flex;align-items:center;gap:10px;margin:0 0 20px;font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:.05em}
.brandmark{width:39px;height:39px;display:grid;place-items:center;background:#1e604a;color:#fff!important;border-radius:12px;font-size:23px}
.brand small{display:block;font:10px 'DM Sans',sans-serif;letter-spacing:.15em;color:#667d6a;margin-top:2px}
.eyebrow{font:700 11px 'DM Sans',sans-serif;letter-spacing:.19em;text-transform:uppercase;color:#607668;margin:0 0 10px}
.hero{position:relative;overflow:hidden;border-radius:24px;background:#133e2f;padding:34px 38px;min-height:270px;margin-bottom:25px;display:flex;align-items:center;gap:20px}
.hero-copy{position:relative;z-index:2;max-width:61%}.hero h1{font-size:clamp(28px,3.4vw,46px);line-height:1.13;color:#f6f5e9!important;margin:8px 0 14px;letter-spacing:-.045em}
.hero p{color:#cee0cc!important;font-size:15px;line-height:1.6;margin:0;max-width:500px}.hero .eyebrow{color:#d9be76!important}
.hero-art{position:absolute;width:46%;right:0;bottom:0;height:100%;object-fit:cover;opacity:.94}
.hero-tag{display:inline-block;padding:5px 11px;border:1px solid #66886c;border-radius:99px;color:#dfebd8;font-size:11px;margin-top:20px;margin-right:6px}
.kpi{background:#fff;border:1px solid #dce5da;border-radius:16px;padding:19px 20px;margin:0 0 14px;min-height:120px}
.kpi-label{font-size:12px;color:#687a6b;font-weight:600}.kpi-value{font-family:'Manrope',sans-serif;font-size:29px;letter-spacing:-.04em;font-weight:800;color:#19382d;margin:5px 0 2px}.kpi-note{font-size:11px;color:#78867c}
.section-label{font-size:11px;letter-spacing:.14em;color:#78856c;margin:23px 0 6px;font-weight:700}
.callout{background:#e6eedb;border-radius:14px;padding:16px 20px;border-left:4px solid #8ba469;color:#284834;margin:14px 0}
.result-card{background:#173f31;color:#fff;border-radius:20px;padding:25px 27px;margin:8px 0 18px}.result-card .big{font-family:'Manrope',sans-serif;font-weight:800;font-size:40px;letter-spacing:-.045em;line-height:1.3;color:#f3f3da}.result-card .sub{color:#c2d7c0;font-size:13px}
.empty-card{background:#e6eddf;border:1px dashed #b3c7a9;border-radius:20px;padding:45px 25px;text-align:center;min-height:265px;margin:18px 0}.empty-card p{color:#63755d!important;font-size:14px}
.footnote{font-size:11px;color:#718071;border-top:1px solid #d4dfd1;padding-top:18px;margin-top:36px}
[data-testid="stMetric"]{background:#fff;border:1px solid #dce5da;border-radius:14px;padding:14px}
[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *{color:#19382d!important}
[data-testid="stForm"]{background:#fff;border:1px solid #dce5da;border-radius:18px;padding:24px}
div[data-baseweb="input"],div[data-baseweb="base-input"],div[data-baseweb="select"]>div,textarea{background:#f8faf5!important;color:#19382d!important;border-color:#d0dccf!important}
input,textarea{color:#19382d!important;-webkit-text-fill-color:#19382d!important}
[data-baseweb="select"] span{color:#19382d!important}[role="listbox"]{background:#fff!important;color:#19382d!important}
button[kind="primary"],button[kind="primaryFormSubmit"]{background:#1e604a!important;border-color:#1e604a!important;border-radius:10px!important}
button[kind="primary"] p,button[kind="primaryFormSubmit"] p{color:#fff!important}
button[kind="secondary"],button[kind="secondaryFormSubmit"]{background:#fff;color:#19382d;border-color:#c8d8c5;border-radius:10px}
[data-testid="stExpander"]{background:#fff;border-color:#dce5da}
[data-testid="stCaptionContainer"] p{color:#6e8073!important}
@media(max-width:760px){.hero{padding:24px;min-height:260px}.hero-copy{max-width:83%}.hero-art{opacity:.25;width:80%}.kpi-value{font-size:25px}.block-container{padding-top:2rem}.result-card .big{font-size:32px}}
</style>''',unsafe_allow_html=True)

LANDSCAPE='''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420"><defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#244e38"/><stop offset="1" stop-color="#94ae66"/></linearGradient></defs><rect width="640" height="420" fill="url(#sky)"/><circle cx="460" cy="105" r="57" fill="#e4c779"/><path d="M0 250Q160 108 320 222T680 202V420H0Z" fill="#739866"/><path d="M0 280Q150 194 340 263T640 241V420H0Z" fill="#456f49"/><path d="M0 315Q230 194 640 330V420H0Z" fill="#b6b970"/><path d="M-50 390Q235 259 650 394M0 430Q260 285 640 422M75 470Q300 330 690 451" fill="none" stroke="#1f523c" stroke-width="20"/><path d="M260 284L151 420M321 284L262 420M385 291L380 420M450 303L500 420M517 318L620 420" stroke="#dce0a7" stroke-width="5" opacity=".6"/><g transform="translate(500 203)"><rect x="0" y="4" width="52" height="39" fill="#e7dcac"/><path d="M-8 4L26-24L61 4Z" fill="#244631"/><rect x="19" y="23" width="15" height="20" fill="#6f8151"/></g><path d="M426 261V192M413 204Q393 187 409 179Q430 182 426 211M428 222Q454 190 465 207Q466 223 428 233" stroke="#173f31" stroke-width="7" fill="#173f31"/><path d="M351 83q12-14 24 0q12-14 24 0M393 63q8-9 16 0q8-9 16 0" fill="none" stroke="#c9dba2" stroke-width="3"/></svg>'''
ART='data:image/svg+xml;base64,'+base64.b64encode(LANDSCAPE.encode()).decode()
def hero(title,subtitle):
    st.markdown(f'<div class="hero"><div class="hero-copy"><div class="eyebrow">ABC Limited · Farmer Income Lab</div><h1>{title}</h1><p>{subtitle}</p><span class="hero-tag">Income estimation</span><span class="hero-tag">Managerial decision support</span></div><img class="hero-art" src="{ART}" alt="Illustrated agricultural fields, farmhouse and sunrise"/></div>',unsafe_allow_html=True)
def card(label,value,note=''):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div><div class="kpi-value">{html.escape(str(value))}</div><div class="kpi-note">{html.escape(note)}</div></div>',unsafe_allow_html=True)
def title(kicker,heading,caption=None):
    st.markdown(f'<div class="eyebrow">{kicker}</div>',unsafe_allow_html=True);st.title(heading)
    if caption:st.caption(caption)
def styled(chart,height=300):
    return chart.properties(height=height).configure(background='#ffffff',view={'stroke':None},
        axis={'labelColor':MUTED,'titleColor':INK,'gridColor':'#edf1e9','domain':False,'labelFont':'sans-serif','titleFont':'sans-serif'},
        legend={'labelColor':INK,'titleColor':INK})
def bar(df,x,y,x_title='',y_title='',height=290,color=GREEN,sort=None):
    c=alt.Chart(df).mark_bar(color=color,cornerRadiusEnd=5).encode(
        x=alt.X(x,title=x_title,axis=alt.Axis(labelAngle=0)), y=alt.Y(y,title=y_title,sort=sort),tooltip=list(df.columns))
    st.altair_chart(styled(c,height),use_container_width=True)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brandmark">⌁</div><div>ABC LIMITED<small>FARMER INCOME LAB</small></div></div>',unsafe_allow_html=True)
    page=st.radio('Workspace',['Overview','Income estimator','Batch predictions','Model evidence','Manager study','Study analysis'],label_visibility='collapsed')
    st.divider();st.caption('ASSIGNMENT 2')
    st.write('Predictive analytics & managerial AI adoption')
    st.caption('Linear regression + logistic regression')
    st.caption('Version 1.0 · Historical data prototype')
    st.caption('Income is in INR. Its reporting period is not specified in the supplied dictionary.')

def overview():
    hero('A clearer view of<br>farmer income.','Explore the data, estimate total income and understand the evidence behind every result.')
    st.info('Taking part in the user study? Start with Manager study before exploring predictions.')
    a,b=st.columns([3,1]);a.markdown('## The dataset at a glance')
    scope=b.selectbox('Dataset scope',list(D['scopes']),label_visibility='collapsed');s=D['scopes'][scope]
    cols=st.columns(4)
    for c,l,v,n in zip(cols,['Records analysed','Median total income','Median non-farm income','Below sample benchmark'],
        [f'{s["n"]:,}',inr(s['median']),inr(s['non_ag_median']),f'{s["below_benchmark_pct"]:.1f}%'],
        ['After repeated loan IDs are removed','Agriculture + non-agriculture','Known component of total income',f'Training benchmark: {inr(M["benchmark_inr"])}']):
        with c:card(l,v,n)
    left,right=st.columns([1.1,1],gap='large')
    with left:
        st.subheader('How income is distributed')
        st.caption(f'{scope.title()} · all income values included, with an open-ended top band')
        h=pd.DataFrame(s['histogram']);c=alt.Chart(h).mark_bar(color=GREEN,cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(
            x=alt.X('band:N',sort=h.band.tolist(),title='Total income (INR)',axis=alt.Axis(labelAngle=-25)),
            y=alt.Y('count:Q',title='Records'),tooltip=['band:N','count:Q'])
        st.altair_chart(styled(c),use_container_width=True)
    with right:
        st.subheader('Land area and median income')
        st.caption('Observed associations across land-area groups. Area uses the original source units.')
        l=pd.DataFrame(s['land_bands']);l['median_lakh']=l['median']/1e5
        c=alt.Chart(l).mark_bar(color=GOLD,cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(
            x=alt.X('band:N',sort=l.band.tolist(),title='Land-area group',axis=alt.Axis(labelAngle=0)),
            y=alt.Y('median_lakh:Q',title='Median total income (₹ lakh)'),tooltip=['band','n',alt.Tooltip('median:Q',format=',.0f')])
        st.altair_chart(styled(c),use_container_width=True)
    st.subheader('Income across states')
    st.caption('All states in the dataset, sorted by record count. Small samples are labelled and do not represent state populations.')
    states=pd.DataFrame(D['states']).sort_values('n',ascending=False);states['median_lakh']=states['median']/1e5
    states['label']=states.apply(lambda r:f'{r.state}  (n={r.n:,})',axis=1)
    c=alt.Chart(states).mark_bar(cornerRadiusEnd=4).encode(x=alt.X('median_lakh:Q',title='Median total income (₹ lakh)'),
        y=alt.Y('label:N',sort=states.label.tolist(),title=None),color=alt.condition(alt.datum.n>=100,alt.value(GREEN),alt.value('#a9b9a6')),
        tooltip=['state','n',alt.Tooltip('median:Q',format=',.0f')])
    st.altair_chart(styled(c,410),use_container_width=True)
    st.markdown('<div class="callout"><b>What this tool estimates</b><br>Total income recorded in the source, including farm and non-farm income. It is an estimate for a comparable profile, not a forecast for a specified future season.</div>',unsafe_allow_html=True)
    with st.expander('Data definitions and coverage'):
        st.write(f'The uploaded training file has {M["original_labeled_rows"]:,} rows and {M["original_column_count"]} columns. We retained {M["deduplicated_rows"]:,} distinct loan IDs after removing {M["removed_duplicate_loan_ids"]} repeated IDs. FarmerID is defined as a loan agreement number, so these are records, not a verified count of unique farmers.')
        st.write(f'The separate scoring file has {M["unlabeled_rows"]:,} rows with no observed income. It cannot be used to calculate prediction accuracy.')
        st.write('The dictionary confirms INR but does not identify an income reporting period or the units of the farmer land-holding field. Environmental inputs used by this model refer to 2022. The model has not been validated on current-year outcomes.')
        st.write('Data source: the three CSVs supplied for this project. A matching dataset listing was located on Kaggle; original collection, consent and licensing were not independently verified. Raw records and loan IDs are excluded from this deployment package.')
        st.link_button('Matching Kaggle dataset listing','https://www.kaggle.com/datasets/manishpatil115/farmer-income-prediction-dataset')

def show_result(row,heading='Estimated total income',key='result'):
    r=score(row,BUNDLE,M).iloc[0]
    st.markdown(f'<div class="result-card"><div class="sub">{heading}</div><div class="big">{inr(r.predicted_total_income_inr)}</div><div class="sub">₹{r.predicted_total_income_inr:,.0f} · agriculture + non-agriculture</div></div>',unsafe_allow_html=True)
    st.markdown(f'**Empirical error range: {inr(r.range_low_inr)} to {inr(r.range_high_inr)}**')
    st.caption(f'Built from validation errors. It covered {M["error_band_holdout_coverage"]:.1%} of held-out records. This is a broad error band, not a guarantee for an individual farmer.')
    p=float(r.probability_below_benchmark)
    st.progress(p,text=f'{p:.1%} estimated probability of income below {inr(M["benchmark_inr"])}')
    st.caption('This comes from a separate logistic regression model. The benchmark is the training median, not a poverty line or an eligibility rule.')
    if r.below_known_non_ag_income:st.warning('The total estimate is lower than the known non-farm component. This inconsistent result needs manual review.')
    if r.predicted_total_income_inr>M['train_target_p99'] or row.non_ag_income.iloc[0]>M['schema']['non_ag_income']['p99']:
        st.warning('This is a high-income or extreme profile. The overall error range may understate uncertainty here.')
    with st.expander('Why did the linear model produce this estimate?'):
        parts,intercept=contributions(row,BUNDLE);parts=parts.sort_values('contribution_inr')
        parts['contribution_lakh']=parts.contribution_inr/1e5
        c=alt.Chart(parts).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X('contribution_lakh:Q',title='Contribution to raw estimate (₹ lakh)'),
            y=alt.Y('factor:N',sort=parts.factor.tolist(),title=None),
            color=alt.condition(alt.datum.contribution_inr>=0,alt.value(GREEN),alt.value(GOLD)),
            tooltip=['factor',alt.Tooltip('contribution_inr:Q',format=',.0f')])
        st.altair_chart(styled(c,280),use_container_width=True)
        st.write(f'Model intercept: ₹{intercept:,.0f}. Adding these contributions gives the raw estimate of ₹{r.raw_linear_prediction_inr:,.0f}. Negative raw predictions, if any, are displayed as zero.')
        st.caption('Numeric effects are relative to training means after imputation and scaling; the state effect is relative to the encoded reference state. These are associations learned by the model, not proven causes.')
    return r

def estimator():
    title('Estimate','Farmer income estimator','Enter a comparable profile. Values must use the same definitions and reporting period as the source.')
    st.warning('Known non-agricultural income is a required input and is already part of total income. If it is unknown, this model cannot provide a reliable estimate. Do not substitute zero for an unknown value.')
    a,b=st.columns([1.15,1],gap='large');demo=D['profiles'][0]['inputs']
    with a:
        with st.form('income_form'):
            st.subheader('Farmer and location details')
            state=st.selectbox('State',M['states'],index=M['states'].index(demo['state']))
            c1,c2=st.columns(2);values={'state':state}
            labels={
                'land_area':('Agricultural land area (source units)','The dictionary does not confirm acres or hectares for this field. Use the original data units.'),
                'non_ag_income':('Known non-agricultural income (₹)','Income from non-farm sources for the same reporting period as total income.'),
                'mandi_distance':('Distance to nearest mandi (km)','Source proximity field for Kharif 2022.'),
                'kharif_rainfall':('Kharif rainfall (mm)','Seasonal average rainfall, 2022.'),
                'rabi_rainfall':('Rabi rainfall (mm)','Seasonal average rainfall, 2022.'),
                'cropping_density':('Kharif cropping-density index','Use the source index. Do not interpret it as a crop yield or a measured percentage without confirmation.'),
                'agri_score':('Kharif agricultural score','Historical village-level source score for 2022. Its scale is not fully documented.')}
            for i,c in enumerate(NUM):
                with (c1 if i%2==0 else c2):
                    values[c]=st.number_input(labels[c][0],min_value=0.0,value=float(demo[c]),step=1000.0 if c=='non_ag_income' else .1,help=labels[c][1],key='input_'+c)
            st.caption('Prefilled values are a hypothetical demonstration profile. Replace them with measured values for your case.')
            submitted=st.form_submit_button('Estimate income',type='primary',use_container_width=True)
        if submitted:
            try:
                row,warnings=validate_inputs(pd.DataFrame([values]),M)
                st.session_state['last_income']={'row':row,'warnings':warnings}
            except ValueError as e:st.error(str(e))
    with b:
        if 'last_income' not in st.session_state:
            st.markdown('<div class="empty-card"><div style="font-size:44px;color:#5f845c">⌁</div><h3>Your estimate will appear here</h3><p>Complete the profile and select Estimate income.<br>You will see the estimate, an error range and an explanation.</p></div>',unsafe_allow_html=True)
        else:
            saved=st.session_state.last_income;row=saved['row']
            st.caption('Result for the last submitted profile')
            for w in saved['warnings']:st.warning(w)
            r=show_result(row)
            download=pd.concat([row.reset_index(drop=True),score(row,BUNDLE,M)],axis=1)
            st.download_button('Download this estimate',safe_csv(download),'farmer_income_estimate.csv','text/csv',use_container_width=True)
    if 'last_income' in st.session_state:
        st.divider();st.subheader('Compare a different profile')
        st.caption('Change land area and known non-farm income while keeping other inputs fixed. Differences are model associations, not evidence that an intervention will cause an income change.')
        base=st.session_state.last_income['row'];a,b,c=st.columns([1,1,1])
        land=a.number_input('Alternative land area (source units)',min_value=0.0,value=float(base.land_area.iloc[0]),key='alt_land')
        other=b.number_input('Alternative known non-farm income (₹)',min_value=0.0,value=float(base.non_ag_income.iloc[0]),step=1000.,key='alt_nonag')
        changed=base.copy();changed['land_area']=land;changed['non_ag_income']=other
        _,w=validate_inputs(changed,M)
        for msg in w:st.caption(msg)
        new=score(changed,BUNDLE,M).predicted_total_income_inr.iloc[0];old=score(base,BUNDLE,M).predicted_total_income_inr.iloc[0]
        c.metric('Alternative estimate',inr(new),delta=f'₹{new-old:+,.0f} from submitted profile',delta_color='off')

def batch():
    title('Batch predictions','Estimate income for a CSV','Upload the supplied lte_test.csv or a file with the eight template columns. Maximum 15,000 rows.')
    a,b=st.columns([2,1])
    with a:st.write('The original 105-column file is accepted. Only the eight model inputs are used. Uploaded data is processed in this app session and is not written to a central database.')
    with b:st.download_button('Download input template',(ROOT/'sample_batch.csv').read_bytes(),'farmer_input_template.csv','text/csv',use_container_width=True)
    st.caption('The template contains three hypothetical example profiles. Outputs contain source row numbers to match the input. Loan IDs and other unused fields are excluded from downloads.')
    upload=st.file_uploader('Farmer input CSV',type=['csv'])
    if upload:
        if upload.size>25*1024*1024:st.error('Use a CSV under 25 MB.');return
        try:
            raw=pd.read_csv(upload,low_memory=False)
            x,warnings=validate_inputs(raw,M)
            for msg in warnings:st.warning(msg)
            pred=score(x,BUNDLE,M);out=pred.copy();out.insert(0,'source_row',np.arange(1,len(out)+1))
            a,b,c=st.columns(3)
            a.metric('Records scored',f'{len(out):,}');b.metric('Median predicted income',inr(out.predicted_total_income_inr.median()))
            c.metric('Predicted below benchmark',f'{out.predicted_below_benchmark.mean():.1%}')
            st.dataframe(out.head(100),hide_index=True,use_container_width=True)
            st.caption('Preview shows the first 100 rows. The download includes every scored row. Unlabeled records have no accuracy score.')
            st.download_button('Download all predictions',safe_csv(out),'farmer_income_batch_predictions.csv','text/csv',type='primary')
        except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as e:st.error(str(e))

def evidence():
    title('Model evidence','See how the models performed','All headline results use 9,552 held-out records from villages excluded from training and validation.')
    a,b,c,d=st.columns(4);r=M['linear_holdout'];l=M['logistic_holdout']
    with a:card('Linear regression R²',f'{r["r2"]:.3f}','Variance explained; not percentage accuracy')
    with b:card('Mean absolute error',inr(r['mae']),'Average absolute income-estimation error')
    with c:card('Logistic ROC-AUC',f'{l["roc_auc"]:.3f}','Below-benchmark classification')
    with d:card('Logistic accuracy',f'{l["accuracy"]:.1%}','Threshold fixed at 0.50')
    t1,t2,t3=st.tabs(['Income regression','Income category','Data and method'])
    with t1:
        st.subheader('Does the model beat simple alternatives?')
        rows=[]
        for label,k in [('Linear regression','linear_holdout'),('Known non-farm income + typical farm component','known_component_baseline_holdout'),
            ('Training median for everyone','baseline_median_holdout'),('Training mean for everyone','baseline_mean_holdout'),
            ('Linear model without non-farm income','without_non_ag_income_holdout')]:
            m=M[k];rows.append({'Model':label,'MAE (₹ lakh)':m['mae']/1e5,'RMSE (₹ lakh)':m['rmse']/1e5,'R²':m['r2']})
        st.dataframe(pd.DataFrame(rows).round(3),hide_index=True,use_container_width=True)
        st.warning(f'The model depends heavily on known non-agricultural income. Without this input, held-out R² falls to {M["without_non_ag_income_holdout"]["r2"]:.3f}. Good overall R² does not imply precise estimates for every farmer.')
        improvement=100*(1-r['mae']/M['known_component_baseline_holdout']['mae'])
        st.write(f'Against the stronger known-component baseline, the model reduces MAE by {improvement:.1f}%. Median absolute error is {inr(r["median_absolute_error"])} and mean absolute percentage error is {r["mape_pct"]:.1f}%.')
        left,right=st.columns([1.15,1],gap='large')
        with left:
            st.subheader('Actual versus estimated income')
            st.caption('Aggregated counts for all holdout records. Bands are in ₹ lakh; a perfect model lies on the diagonal.')
            grid=pd.DataFrame(D['density'])
            chart=alt.Chart(grid).mark_rect().encode(x=alt.X('actual_band:N',sort=D['density_order'],title='Actual income (₹ lakh)',axis=alt.Axis(labelAngle=-50)),
                y=alt.Y('predicted_band:N',sort=list(reversed(D['density_order'])),title='Estimated income (₹ lakh)'),
                color=alt.Color('count:Q',scale=alt.Scale(type='log',scheme='greens'),title='Records'),tooltip=['actual_band','predicted_band','count'])
            st.altair_chart(styled(chart,330),use_container_width=True)
        with right:
            st.subheader('Understand the error range')
            st.write(f'The range extends approximately {inr(M["error_band_radius"])} on either side of an estimate, with a lower limit of zero. Its radius is the 90th percentile of absolute errors on the separate validation set.')
            st.metric('Observed holdout coverage',f'{M["error_band_holdout_coverage"]:.1%}')
            st.caption('This is empirical coverage, not a formal guarantee. Errors can differ by state and income level, and repeated village-level observations violate simple independence assumptions.')
        with st.expander('Error by state on the holdout set'):
            q=pd.DataFrame(D['holdout_segments']);q['mae_in_lakh']=q.mae/1e5
            st.dataframe(q[['state','n','mae_in_lakh','band_coverage']].round(3),hide_index=True,use_container_width=True)
            st.caption('Small state samples give unstable estimates. These comparisons do not establish fairness or national representativeness.')
    with t2:
        st.write(f'Logistic regression estimates whether total income is below **{inr(M["benchmark_inr"])}**, the training-set median. This is a relative sample category, not a poverty or creditworthiness label.')
        cols=st.columns(4)
        for col,key in zip(cols,['precision','recall','f1','brier']):col.metric(key.replace('_',' ').title(),f'{l[key]:.3f}')
        a,b=st.columns(2,gap='large')
        with a:
            st.subheader('Ranking performance')
            roc=alt.Chart(pd.DataFrame(D['roc'])).mark_line(color=GREEN,strokeWidth=3).encode(
                x=alt.X('false_positive_rate:Q',title='False positive rate',scale=alt.Scale(domain=[0,1])),
                y=alt.Y('true_positive_rate:Q',title='True positive rate',scale=alt.Scale(domain=[0,1])))
            diagonal=alt.Chart(pd.DataFrame({'x':[0,1],'y':[0,1]})).mark_line(color='#a8b5aa',strokeDash=[5,5]).encode(x='x:Q',y='y:Q')
            st.altair_chart(styled(roc+diagonal,310),use_container_width=True)
        with b:
            st.subheader('Correct and incorrect classifications')
            labels=['At/above benchmark','Below benchmark'];cm=M['logistic_holdout']['confusion_matrix']
            q=pd.DataFrame([{'Actual':labels[i],'Predicted':labels[j],'Records':cm[i][j]} for i in range(2) for j in range(2)])
            base=alt.Chart(q).encode(x=alt.X('Predicted:N',sort=labels,axis=alt.Axis(labelAngle=0)),y=alt.Y('Actual:N',sort=labels))
            cells=base.mark_rect().encode(color=alt.Color('Records:Q',scale=alt.Scale(range=['#f0f5e9','#7fad80']),legend=None),tooltip=list(q.columns))
            text=base.mark_text(fontSize=23,fontWeight='bold',color=INK).encode(text='Records:Q')
            st.altair_chart(styled(cells+text,260),use_container_width=True)
        st.caption(f'Majority-class baseline accuracy: {M["majority_baseline_accuracy"]:.1%}. C was selected using validation ROC-AUC; the probability threshold was fixed at 0.50 before holdout evaluation. A classification probability is not confidence in the rupee estimate.')
    with t3:
        st.subheader('A reproducible workflow')
        st.dataframe(pd.DataFrame([{'Stage':k.title(),**v} for k,v in M['split'].items()]),hide_index=True,use_container_width=True)
        st.write('1. Keep the most complete row for each repeated loan ID; break ties by original row order. Income and village must agree across repeats.\n2. Split villages defined by state, district and village name into approximately 60% training, 20% validation and 20% holdout groups, using fixed seeds.\n3. Fit median imputation, standardisation and state encoding on training data only. States with fewer than 100 training rows share an OTHER category.\n4. Fit ordinary least-squares linear regression and tune logistic regression C on validation ROC-AUC.\n5. Use validation residuals for the empirical error range and report final metrics on held-out villages. The deployed models are the same training-fit models used in this evaluation.')
        st.dataframe(pd.DataFrame([{'Input':LABELS[c],'Unit':UNITS.get(c,'category'),'Training minimum':M['schema'][c]['min'],'Training maximum':M['schema'][c]['max'],'Missing training rows':M['schema'][c]['missing_train']} for c in NUM]),hide_index=True,use_container_width=True)
        st.write('Excluded predictors include loan IDs, gender, marital status, precise addresses, bureau loan amounts and the target itself. A compact eight-input model was chosen for usability. These omissions do not remove all possible bias or confounding.')
        st.write('Use the model for an academic demonstration and informed review. Income alone does not determine creditworthiness, and this app does not make lending or benefit-eligibility decisions.')
        st.download_button('Download complete model metrics',(ROOT/'metrics.json').read_bytes(),'farmer_model_metrics.json','application/json')

def profile_table(profile):
    return pd.DataFrame({'Input':[LABELS[c] for c in FEATURES],
        'Value':[profile[c].title() if c=='state' else (f'₹{profile[c]:,.0f}' if c=='non_ag_income' else f'{profile[c]:,.2f} {UNITS[c]}') for c in FEATURES]})

def begin_study():
    ss=st.session_state
    before={s:ss.get('before_'+s) for s in ['A','B','C']}
    if not ss.get('study_consent') or ss.get('study_role')=='Select a role' or not ss.get('study_context','').strip() or any(v is None for v in before.values()):
        ss['study_error']='Please record consent, role, decision context and all three initial judgments.'
        return
    ss['study_error']=''
    ss['study_before']={'study_version':STUDY_VERSION,'respondent_id':str(uuid.uuid4())[:12],
        'consent':True,'role':ss.study_role,'years_experience':ss.study_years,'decision_context':ss.study_context.strip(),
        **{s+'_before_band':v for s,v in before.items()}}

def finish_study():
    ss=st.session_state
    after={s:ss.get('after_'+s) for s in ['A','B','C']}
    reasons={s:ss.get('reason_'+s,'') for s in ['A','B','C']}
    ratings={c:ss.get('rate_'+c) for c in RATINGS}
    answers={c:ss.get('open_'+c,'') for c in OPEN}
    if any(v is None for v in list(after.values())+list(ratings.values())):
        ss['study_error']='Please complete all final judgments and ratings.'
        return
    if any(not v.strip() for v in list(reasons.values())+list(answers.values())):
        ss['study_error']='Please answer each open question. You may write "Prefer not to answer".'
        return
    ss['study_error']=''
    row={**ss.study_before,**ratings,**answers,'submitted_utc':datetime.now(timezone.utc).isoformat(),
        **{s+'_after_band':v for s,v in after.items()},**{s+'_reason':v for s,v in reasons.items()}}
    ss['study_complete']=pd.DataFrame([row],columns=COLUMNS)

def study():
    title('Managerial user study','How would you use this tool?','About 10–15 minutes. Intended for managers, team leaders, supervisors, entrepreneurs and experienced professionals.')
    st.info('Responses are not stored centrally. At the end, download your response CSV and return it privately to the researcher. Avoid names, contact details and confidential business information.')
    if st.session_state.get('study_error'):st.error(st.session_state.study_error)
    if 'study_complete' in st.session_state:
        st.success('Your response is ready. Download it now and return the file to the researcher.')
        d=st.session_state.study_complete
        st.download_button('Download my response',safe_csv(d),f'farmer_study_{d.respondent_id.iloc[0]}.csv','text/csv',type='primary')
        st.caption('Closing or reloading the session can erase an undownloaded response. No participant responses are bundled with this project.')
        return
    if 'study_before' not in st.session_state:
        st.subheader('1. Give your initial view')
        st.write(f'Imagine reviewing total-income estimates for farmer profiles at ABC Limited. For each hypothetical profile, judge whether its income is below {inr(M["benchmark_inr"])}. There is no observed true outcome for these demonstration profiles.')
        with st.form('study_first'):
            consent=st.checkbox('I voluntarily agree to take part and allow my anonymous response and anonymous quotes to be used in this assignment.',key='study_consent')
            a,b=st.columns(2);role=a.selectbox('Your role',['Select a role','Manager','Team leader','Supervisor','Entrepreneur','Working professional','Other experienced decision-maker'],key='study_role')
            years=b.number_input('Years of work experience',min_value=0,max_value=60,value=3,step=1,key='study_years')
            context=st.text_input('What types of decisions do you make?',key='study_context')
            before={}
            for s,p in zip(['A','B','C'],D['profiles']):
                st.markdown('#### '+p['name']);st.dataframe(profile_table(p['inputs']),hide_index=True,use_container_width=True)
                before[s]=st.radio(f'Your initial income category for {s}',BANDS,index=None,horizontal=True,key='before_'+s)
            st.form_submit_button('Continue to model estimates',type='primary',on_click=begin_study)
        return
    st.subheader('2. Review the estimates and evidence')
    st.write(f'On held-out villages, linear regression had an average absolute error of {inr(M["linear_holdout"]["mae"])}. The empirical range covered {M["error_band_holdout_coverage"]:.1%} of held-out records. Logistic classification accuracy was {M["logistic_holdout"]["accuracy"]:.1%}. The model depends heavily on known non-farm income and historical 2022 inputs. Income period and land units are not fully documented.')
    for s,p in zip(['A','B','C'],D['profiles']):
        with st.container(border=True):
            st.markdown('#### '+p['name']+' · estimate and explanation')
            st.caption('Your initial judgment: '+st.session_state.study_before[s+'_before_band'])
            show_result(pd.DataFrame([p['inputs']]),heading=p['name']+' · estimated total income',key=s)
    st.subheader('3. Give your final view and feedback')
    with st.form('study_final'):
        after={};reasons={}
        for s in ['A','B','C']:
            after[s]=st.radio(f'Your final income category for {s}',BANDS,index=None,horizontal=True,key='after_'+s)
            reasons[s]=st.text_area(f'Why did you keep or change your view for {s}?',key='reason_'+s)
        st.markdown('#### Rate your experience')
        st.caption('1 = strongly disagree · 2 = disagree · 3 = neutral · 4 = agree · 5 = strongly agree')
        ratings={c:st.radio(q,[1,2,3,4,5],index=None,horizontal=True,key='rate_'+c) for c,q in RATINGS.items()}
        st.markdown('#### Tell us in your own words')
        answers={c:st.text_area(q,key='open_'+c) for c,q in OPEN.items()}
        st.form_submit_button('Finish and prepare my response',type='primary',on_click=finish_study)

def analysis():
    title('Study analysis','Turn real feedback into findings','Upload the response CSVs that participants downloaded. No example participants or invented results are included.')
    uploads=st.file_uploader('Completed response CSVs',type=['csv'],accept_multiple_files=True)
    if not uploads:
        st.write('Recruit approximately 8–12 relevant participants if feasible, record the actual sample, and collect consented feedback. This is a suggested target rather than a claimed completed study.')
        st.dataframe(pd.DataFrame(THEMES.items(),columns=['Theme code','Meaning']),hide_index=True,use_container_width=True)
        st.download_button('Download blank response structure',safe_csv(pd.DataFrame(columns=COLUMNS)),'blank_farmer_study_responses.csv','text/csv')
        return
    try:
        d=merge_responses([pd.read_csv(u,keep_default_na=False,dtype={'respondent_id':str}) for u in uploads])
        if d.empty:st.warning('The uploaded files contain no completed responses.');return
        ratings=rating_summary(d);decisions=decision_summary(d);coding=coding_template(d)
        a,b,c=st.columns(3);a.metric('Actual respondents',len(d));b.metric('Roles represented',d.role.nunique());c.metric('Average usefulness',f'{d.usefulness.mean():.2f} / 5')
        st.subheader('Ratings and changes in judgment')
        st.caption('Exact duplicate responses are removed. Conflicting responses with the same ID are rejected. Do not reverse the meaning of the wrong-decision-concern rating: a high score means greater concern.')
        q=ratings.copy();q['measure']=q.measure.str.replace('_',' ').str.title()
        c=alt.Chart(q).mark_bar(color=GREEN,cornerRadiusEnd=5).encode(x=alt.X('mean:Q',scale=alt.Scale(domain=[0,5]),title='Average rating (1–5)'),y=alt.Y('measure:N',title=None),tooltip=['measure','n','mean','median','agree_4_or_5_pct'])
        st.altair_chart(styled(c,260),use_container_width=True)
        st.dataframe(ratings.round(2),hide_index=True,use_container_width=True);st.dataframe(decisions.round(2),hide_index=True,use_container_width=True)
        st.caption('A changed judgment measures influence, not improved decision accuracy. The three demonstration profiles have no observed outcomes.')
        st.subheader('Interpret the open responses')
        st.write('Read each response, assign one or more theme codes separated by semicolons, copy exact quotations, and add your interpretation. Retain disagreements and negative feedback. Counts describe this sample, not the wider population.')
        st.dataframe(pd.DataFrame(THEMES.items(),columns=['Theme code','Meaning']),hide_index=True,use_container_width=True)
        st.download_button('Download qualitative coding template',safe_csv(coding),'farmer_qualitative_coding.csv','text/csv')
        coded=st.file_uploader('Completed qualitative coding CSV',type=['csv'],key='coded')
        summary=None
        if coded:
            summary=theme_summary(pd.read_csv(coded,keep_default_na=False,dtype={'respondent_id':str}),d)
            st.dataframe(summary,hide_index=True,use_container_width=True)
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            for n,f in [('responses',d),('ratings',ratings),('decision_changes',decisions),('qualitative_coding',coding)]:z.writestr(n+'.csv',safe_csv(f))
            if summary is not None:z.writestr('theme_counts.csv',safe_csv(summary))
        st.download_button('Download study analysis files',buf.getvalue(),'Farmer_Study_Analysis_PRIVATE.zip','application/zip',type='primary')
    except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as e:st.error(str(e))

{'Overview':overview,'Income estimator':estimator,'Batch predictions':batch,'Model evidence':evidence,'Manager study':study,'Study analysis':analysis}[page]()
st.markdown('<div class="footnote">ABC Limited · Farmer Income Lab · Total income in INR · Historical data prototype · Use alongside verified records and professional judgment</div>',unsafe_allow_html=True)
