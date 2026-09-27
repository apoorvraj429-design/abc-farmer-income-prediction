"""Managerial feedback collection and analysis. Contains no invented responses."""
import argparse
from pathlib import Path
import pandas as pd
from core import safe_csv

STUDY_VERSION='farmer-income-v1'
RATINGS={
 'usefulness':'The tool would help me prepare for farmer-income discussions.',
 'ease_of_use':'I found the tool easy to understand and use.',
 'trust':'I would consider this estimate alongside other evidence.',
 'explanation_value':'The explanation helped me understand the estimate.',
 'adoption_intent':'I would consider using a suitably validated version at work.',
 'wrong_decision_concern':'Fear of a wrong decision would discourage me from using this tool.'}
OPEN={
 'useful_tasks':'Where would this tool be useful in your work, and where would it not help?',
 'likes':'What did you like about the tool?',
 'dislikes':'What did you dislike or find confusing?',
 'trust_reasons':'What made you trust or distrust the estimate? What would increase your trust?',
 'experience_conflict':'If your experience and the model disagree, what would you do and why?',
 'explainability':'How did the explanation affect your willingness to use the tool?',
 'adoption_barriers':'What personal or organisational factors could discourage adoption?',
 'data_limits':'How do the broad error range, historical inputs and reliance on known non-farm income affect your view?',
 'improvements':'What would need to change before you would use the tool at work?'}
THEMES={
 'USEFULNESS':'Time saving, preparation, screening for discussion or decision relevance.',
 'TRUST':'Validation, reliability, uncertainty or data quality.',
 'EXPLAINABILITY':'Understanding model drivers or needing clearer explanations.',
 'HUMAN_JUDGMENT':'Experience, verification, overrides or shared decisions.',
 'ERROR_ACCOUNTABILITY':'Fear of mistakes, responsibility or consequences.',
 'ORGANISATION':'Training, leadership, cost, access, integration or workflow.',
 'USABILITY':'Interface, language, effort or accessibility.',
 'DATA_LIMITS':'Population, historical data, known-income dependency or missing context.',
 'OTHER':'An emergent theme. Explain it in the interpretation field.'}
BANDS=['Below benchmark','At or above benchmark','Unsure']
BASE=['study_version','respondent_id','consent','role','years_experience','decision_context','submitted_utc']
CASE_FIELDS=[f'{s}_{v}' for s in ['A','B','C'] for v in ['before_band','after_band','reason']]
COLUMNS=BASE+CASE_FIELDS+list(RATINGS)+list(OPEN)

def merge_responses(frames):
    if not frames: raise ValueError('Upload at least one completed response CSV.')
    for f in frames:
        missing=set(COLUMNS)-set(f.columns)
        if missing:raise ValueError('Response file missing columns: '+', '.join(sorted(missing)))
    d=pd.concat([f[COLUMNS] for f in frames],ignore_index=True).fillna('').drop_duplicates()
    if d.empty:return d
    d.respondent_id=d.respondent_id.astype(str).str.strip()
    if d.respondent_id.eq('').any():raise ValueError('Every response needs an anonymous respondent ID.')
    if d.respondent_id.duplicated().any():raise ValueError('Conflicting responses share an ID. Resolve them before analysis.')
    if not d.study_version.eq(STUDY_VERSION).all():raise ValueError('These responses belong to a different study version.')
    if not d.consent.astype(str).str.lower().isin(['true','1','yes']).all():raise ValueError('Only consented responses can be analysed.')
    for c in RATINGS:
        v=pd.to_numeric(d[c],errors='coerce')
        if v.isna().any() or not (v.between(1,5)&v.eq(v.round())).all():raise ValueError(f'{c} must be an integer from 1 to 5.')
        d[c]=v.astype(int)
    for c in [c for c in CASE_FIELDS if not c.endswith('reason')]:
        if not d[c].isin(BANDS).all():raise ValueError('Invalid income-category response in '+c)
    return d.reset_index(drop=True)

def rating_summary(d):
    return pd.DataFrame([{'measure':c,'n':len(d),'mean':d[c].mean(),'median':d[c].median(),
        'agree_4_or_5_pct':100*d[c].ge(4).mean(),**{f'rating_{i}_count':int(d[c].eq(i).sum()) for i in range(1,6)}} for c in RATINGS])

def decision_summary(d):
    return pd.DataFrame([{'profile':s,'n':len(d),
        'changed_category_count':int((d[f'{s}_before_band']!=d[f'{s}_after_band']).sum()),
        'changed_category_pct':float(100*(d[f'{s}_before_band']!=d[f'{s}_after_band']).mean())} for s in ['A','B','C']])

def coding_template(d):
    rows=[]
    for _,r in d.iterrows():
        for q in list(OPEN)+[f'{s}_reason' for s in ['A','B','C']]:
            if str(r[q]).strip():rows.append({'respondent_id':r.respondent_id,'question':q,'response':r[q],
                'themes_semicolon_separated':'','exact_quote':'','interpretation':'','coder':''})
    return pd.DataFrame(rows,columns=['respondent_id','question','response','themes_semicolon_separated','exact_quote','interpretation','coder'])

def theme_summary(coded,responses):
    required=set(coding_template(responses).columns)
    if not required.issubset(coded.columns):raise ValueError('Use the downloaded qualitative coding template.')
    truth=coding_template(responses).set_index(['respondent_id','question']).response.to_dict()
    seen=set();rows=[]
    for _,r in coded.fillna('').iterrows():
        key=(str(r.respondent_id),str(r.question))
        if key not in truth or str(r.response)!=str(truth[key]):raise ValueError('Coded text does not match an original response.')
        if key in seen:raise ValueError('Duplicate respondent/question row in coding file.')
        seen.add(key)
        if r.exact_quote and str(r.exact_quote) not in str(r.response):raise ValueError('Quotes must match the original response exactly.')
        for code in [c.strip() for c in str(r.themes_semicolon_separated).split(';') if c.strip()]:
            if code not in THEMES:raise ValueError('Unknown theme: '+code)
            rows.append({'respondent_id':r.respondent_id,'theme':code})
    if not rows:return pd.DataFrame(columns=['theme','respondents_mentioning','share_of_sample_pct'])
    out=pd.DataFrame(rows).drop_duplicates().groupby('theme').size().rename('respondents_mentioning').reset_index()
    out['share_of_sample_pct']=100*out.respondents_mentioning/len(responses)
    return out.sort_values('respondents_mentioning',ascending=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('responses',nargs='+');p.add_argument('--coded');p.add_argument('--out',default='private_study_results')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=merge_responses([pd.read_csv(x,keep_default_na=False,dtype={'respondent_id':str}) for x in a.responses])
    if d.empty:raise ValueError('No completed responses found.')
    for n,f in [('responses',d),('ratings',rating_summary(d)),('decisions',decision_summary(d)),('qualitative_coding',coding_template(d))]:
        (out/f'{n}.csv').write_bytes(safe_csv(f))
    if a.coded:(out/'themes.csv').write_bytes(safe_csv(theme_summary(pd.read_csv(a.coded,keep_default_na=False),d)))
    print(f'Analysed {len(d)} actual responses. Interpret qualitative themes after human coding.')

if __name__=='__main__':main()
