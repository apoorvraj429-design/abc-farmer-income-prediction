"""Shared feature definitions, validation and prediction for ABC Limited."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET = 'Target_Variable/Total Income'
SOURCE_COLUMNS = {
    'state': 'State',
    'land_area': 'Total_Land_For_Agriculture',
    'non_ag_income': 'Non_Agriculture_Income',
    'mandi_distance': 'K022-Proximity to nearest mandi (Km)',
    'kharif_rainfall': 'K022-Seasonal Average Rainfall (mm)',
    'rabi_rainfall': 'R022-Seasonal Average Rainfall (mm)',
    'cropping_density': 'Kharif Seasons  Cropping density in 2022',
    'agri_score': 'Kharif Seasons  Agricultural Score in 2022',
}
NUM = list(SOURCE_COLUMNS)[1:]
FEATURES = list(SOURCE_COLUMNS)
LABELS = {'state':'State', 'land_area':'Agricultural land area',
    'non_ag_income':'Known non-agricultural income', 'mandi_distance':'Distance to mandi',
    'kharif_rainfall':'Kharif rainfall (2022)', 'rabi_rainfall':'Rabi rainfall (2022)',
    'cropping_density':'Kharif cropping density (2022)', 'agri_score':'Kharif agricultural score (2022)'}
UNITS = {'land_area':'source units', 'non_ag_income':'INR', 'mandi_distance':'km',
    'kharif_rainfall':'mm','rabi_rainfall':'mm','cropping_density':'source index','agri_score':'source index'}

def extract_features(frame):
    """Accept the original CSV headers or the eight friendly template headers."""
    out = frame.copy()
    out.columns = out.columns.str.strip()
    rename = {v.strip():k for k,v in SOURCE_COLUMNS.items() if k not in out}
    out = out.rename(columns=rename)
    missing = set(FEATURES)-set(out.columns)
    if missing:
        raise ValueError('Missing columns: '+', '.join(sorted(missing))+'. Download the input template.')
    if out.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    out = out[FEATURES].copy()
    out['state'] = out.state.astype('string').str.strip().str.upper().fillna('')
    return out

def validate_inputs(frame, meta):
    if frame.empty: raise ValueError('The file contains no records.')
    if len(frame)>15000: raise ValueError('Use at most 15,000 records per upload.')
    x=extract_features(frame)
    warnings=[]
    for c in NUM:
        original=x[c]
        v=pd.to_numeric(original,errors='coerce')
        bad=v.isna() & original.notna() & original.astype(str).str.strip().ne('')
        if bad.any(): raise ValueError(f'{LABELS[c]} must contain numbers or blank cells.')
        if np.isinf(v).any() or v.lt(0).any():
            raise ValueError(f'{LABELS[c]} must contain finite, non-negative numbers.')
        if c=='non_ag_income' and v.isna().any():
            raise ValueError('Known non-agricultural income is required. Enter 0 only when it is truly zero.')
        n=int(v.isna().sum())
        if n: warnings.append(f'{LABELS[c]}: {n} missing value(s) replaced with the training median.')
        bounds=meta['schema'][c]
        outside=v.lt(bounds['min'])|v.gt(bounds['max'])
        if outside.any(): warnings.append(f'{LABELS[c]}: {int(outside.sum())} value(s) outside the training range.')
        x[c]=v.astype(float)
    if x.state.eq('').any(): raise ValueError('State is required.')
    unknown=~x.state.isin(meta['states'])
    if unknown.any(): raise ValueError('Unsupported state(s): '+', '.join(sorted(x.loc[unknown,'state'].unique())))
    return x, warnings

def load_artifacts():
    return joblib.load(ROOT/'model_bundle.joblib'), json.loads((ROOT/'metrics.json').read_text())

def model_inputs(frame, bundle):
    x=frame.copy()
    x['state']=x.state.where(x.state.isin(bundle['common_states']),'OTHER')
    return x

def score(frame, bundle, meta):
    x=model_inputs(frame,bundle)
    raw=bundle['linear'].predict(x)
    estimate=np.maximum(0,raw)
    probability=bundle['logistic'].predict_proba(x)[:,1]
    radius=meta['error_band_radius']
    return pd.DataFrame({'predicted_total_income_inr':estimate,
        'range_low_inr':np.maximum(0,estimate-radius),
        'range_high_inr':estimate+radius,
        'probability_below_benchmark':probability,
        'predicted_below_benchmark':(probability>=meta['probability_threshold']).astype(int),
        'raw_linear_prediction_inr':raw,
        'below_known_non_ag_income':estimate<frame.non_ag_income.to_numpy()})

def contributions(row,bundle):
    """Exact additive decomposition of the raw linear estimate in INR."""
    model=bundle['linear']; x=model_inputs(row,bundle)
    z=model.named_steps['prep'].transform(x)[0]
    weights=model.named_steps['model'].coef_
    parts=z*weights
    records=[{'factor':LABELS[c],'contribution_inr':float(parts[i])} for i,c in enumerate(NUM)]
    records.append({'factor':'State','contribution_inr':float(parts[len(NUM):].sum())})
    intercept=float(model.named_steps['model'].intercept_)
    return pd.DataFrame(records),intercept

def safe_csv(frame):
    x=frame.copy()
    for c in x.select_dtypes(include=['object','string']).columns:
        x[c]=x[c].map(lambda v: "'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v)
    return x.to_csv(index=False).encode('utf-8-sig')

def inr(value):
    value=float(value)
    if abs(value)>=1e7:return f'₹{value/1e7:,.2f} crore'
    return f'₹{value/1e5:,.2f} lakh'
