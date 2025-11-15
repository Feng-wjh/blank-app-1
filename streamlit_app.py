import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
"""
streamlit_app.py
Single-file Streamlit app for automated Fundamental + Financial analysis

Features:
- yfinance-based financial fetching with robust fallback handling
- Merrill Clock mapping (choose phase)
- Porter Five Forces auto-heuristics + manual override
- Value Chain: DeepSeek API optional auto-score -> fallback heuristics -> manual sliders
- Financial scoring: DuPont/ROE quality, bankruptcy risk, valuation, and hard vetoes
- Batch percentile normalization to increase dispersion; optional z-score exaggeration
- Interactive visualizations with Plotly
- Export CSV
- Clear veto explanations
- Designed to be copy-paste runnable
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math, time, json, io, requests
from typing import Dict, Any, List
from scipy import stats
import plotly.graph_objects as go

# -------------------------
# Requirements
# pip install streamlit yfinance pandas numpy plotly requests scipy
# In Colab you can use pyngrok to expose streamlit:
#    !pip install pyngrok
#    from pyngrok import ngrok
#    !streamlit run streamlit_app.py & sleep 2
#    public_url = ngrok.connect(addr="8501")
# -------------------------

st.set_page_config(page_title="Automated Fundamental & Financial Analyzer", layout="wide")

# -------------------------
# Config / weights / mappings
# -------------------------
WEIGHTS = {'technical':0.10, 'merrill':0.20, 'porter':0.10, 'value_chain':0.10, 'financial':0.50}
POSITION_MAP = [
    (95, 100, '★★★★★ (Core holding)', 0.10),
    (85, 94, '★★★★ (Focus allocation)', 0.07),
    (75, 84, '★★★ (Moderate allocation)', 0.05),
    (65, 74, '★★ (Watch)', 0.03),
    (0, 64, '<65 (Avoid)', 0.00)
]
MERRILL_MAPPING = {
    'recovery': {'technology':100, 'financial services':90, 'consumer cyclical':80, 'consumer discretionary':80},
    'overheat': {'energy':100, 'materials':90, 'industrials':80},
    'stagflation': {'consumer defensive':100, 'healthcare':90, 'utilities':80, 'consumer staples':100},
    'recession': {'real estate':100, 'telecommunications':80, 'defensive':80, 'bonds':100}
}

# -------------------------
# Utility functions (robust field extraction)
# -------------------------
def safe_get_info(ticker_obj: yf.Ticker) -> Dict[str,Any]:
    try:
        return ticker_obj.info or {}
    except Exception:
        return {}

def copy_df_safe(df):
    try:
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        out.index = [str(i) for i in out.index]
        return out
    except Exception:
        return pd.DataFrame()

def get_first_available(df: pd.DataFrame, candidates: List[str]):
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.index:
            try:
                vals = df.loc[c].dropna().values
                if len(vals) > 0:
                    return float(vals[0])
            except Exception:
                continue
    return None

def safe_div(a, b, default=None):
    try:
        if a is None or b is None or b == 0:
            return default
        return a / b
    except Exception:
        return default

# -------------------------
# Data fetcher
# -------------------------
def fetch_yf(ticker: str) -> Dict[str,Any]:
    tk = yf.Ticker(ticker)
    info = safe_get_info(tk)
    income = copy_df_safe(tk.financials)
    balance = copy_df_safe(tk.balance_sheet)
    cashflow = copy_df_safe(tk.cashflow)
    q_income = copy_df_safe(tk.quarterly_financials)
    q_cashflow = copy_df_safe(tk.quarterly_cashflow)
    sector = info.get('sector') or info.get('industry') or 'Unknown'
    industry = info.get('industry') or 'Unknown'
    summary = info.get('longBusinessSummary') or info.get('summary') or ''
    return {
        'ticker': ticker, 'tk': tk, 'info': info,
        'income': income, 'balance': balance, 'cashflow': cashflow,
        'q_income': q_income, 'q_cashflow': q_cashflow,
        'sector': sector, 'industry': industry, 'summary': summary
    }

# -------------------------
# Merrill Clock scoring
# -------------------------
def merrill_clock_score(sector: str, phase: str) -> (float, str):
    s = (sector or '').lower()
    mapping = MERRILL_MAPPING.get(phase.lower(), {})
    for k, v in mapping.items():
        if k in s or s in k:
            return float(v), f"Matched '{k}' -> {v}"
    heur = {'technology':85,'financial':80,'energy':75,'healthcare':70,'utilities':65,'consumer':70,'industrial':68}
    for k, v in heur.items():
        if k in s:
            return float(v), f"Heuristic '{k}' -> {v}"
    return 50.0, "Neutral mapping"

# -------------------------
# Porter Five Forces heuristic auto-score
# -------------------------
def porter_auto(sector: str, industry: str, summary: str) -> Dict[str,Any]:
    text = (summary or '').lower()
    s = (sector or '').lower()
    i = (industry or '').lower()
    ratings = {'competitive_intensity':3,'entry_barriers':3,'substitute_threat':3,'supplier_bargaining':3,'buyer_bargaining':3}
    if 'software' in s or 'technology' in s:
        ratings.update({'competitive_intensity':4,'entry_barriers':4,'substitute_threat':2})
    if 'retail' in s or 'consumer' in s:
        ratings.update({'competitive_intensity':5,'entry_barriers':2,'buyer_bargaining':4})
    if any(k in text for k in ['patent','proprietary','moat','intellectual property','exclusive']):
        ratings['entry_barriers'] = max(ratings['entry_barriers'],5)
        ratings['competitive_intensity'] = max(ratings['competitive_intensity'],4)
    if any(k in text for k in ['commodity','generic','undifferentiated']):
        ratings['substitute_threat'] = max(ratings['substitute_threat'],4)
    for k in ratings: ratings[k] = min(5, max(1, int(round(ratings[k]))))
    weighted = (ratings['competitive_intensity']*0.3 + ratings['entry_barriers']*0.25 +
                ratings['substitute_threat']*0.2 + ratings['supplier_bargaining']*0.15 + ratings['buyer_bargaining']*0.1)
    porter_pct = (weighted / 5.0) * 100.0
    return {'ratings': ratings, 'porter_raw': weighted, 'porter_pct': porter_pct}

# -------------------------
# Value Chain: DeepSeek optional -> heuristics -> manual
# -------------------------
def call_deepseek_valuechain(summary: str, deepl_key: str = None) -> Dict[str,float]:
    """
    Optional: call a hypothetical DeepSeek API to analyze value-chain and return sub-scores.
    In absence of a real key, this function returns None.
    """
    if not deepl_key:
        return None
    # Placeholder structure - adapt to real DeepSeek API if available
    try:
        url = "https://api.deepseek.example/analyze/valuechain"  # placeholder
        payload = {'text': summary}
        headers = {'Authorization': f"Bearer {deepl_key}", 'Content-Type': 'application/json'}
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            j = r.json()
            # expect j to contain main_scores and support_scores (1-10)
            return j
    except Exception:
        return None
    return None

def value_chain_auto(summary: str, info: Dict[str,Any]) -> Dict[str,Any]:
    # Try deepseek: optional
    deepl_key = st.session_state.get('deepseek_key', None)
    deep = call_deepseek_valuechain(summary, deepl_key) if deepl_key else None
    if deep:
        main = deep.get('main_scores', {})
        support = deep.get('support_scores', {})
    else:
        text = (summary or '').lower()
        main = {}
        main['Inbound Logistics'] = 6 if any(k in text for k in ['supply chain','supplier','procure','procurement']) else 7
        main['Operations'] = 7 if any(k in text for k in ['manufactur','production','plant','capacity','capacity utilisation']) else 6
        main['Outbound Logistics'] = 6 if any(k in text for k in ['logistic','delivery','ship','distribution']) else 7
        main['Marketing & Sales'] = 7 if any(k in text for k in ['brand','market','channel','advertis']) else 6
        main['Service'] = 6 if any(k in text for k in ['service','support','warranty','after-sales']) else 7
        support = {}
        support['R&D'] = 8 if any(k in text for k in ['research','r&d','patent','innovation']) else 4
        support['HR Management'] = 7 if 'employee' in text else 6
        support['Infrastructure'] = 7 if any(k in text for k in ['automation','platform','digital','erp','cloud']) else 5
    main_avg = np.mean(list(main.values()))
    support_avg = np.mean(list(support.values()))
    value_chain_score = (main_avg * 0.6 + support_avg * 0.4)  # 1-10 scale
    value_chain_pct = (value_chain_score / 10.0) * 100.0
    return {'main_scores': main, 'support_scores': support, 'main_avg': main_avg, 'support_avg': support_avg, 'value_chain_pct': value_chain_pct}

# -------------------------
# Financial scoring (operation, bankruptcy, valuation)
# -------------------------
def score_roe_quality(roe_pct):
    if roe_pct is None: return 50.0
    if roe_pct > 20: return 100.0
    if 15 <= roe_pct <= 20: return 80.0
    if 10 <= roe_pct < 15: return 60.0
    if 5 <= roe_pct < 10: return 40.0
    return 20.0

def compute_operational(yfdata: Dict[str,Any]) -> Dict[str,Any]:
    income = yfdata['income']; bal = yfdata['balance']; cf = yfdata['cashflow']
    net = get_first_available(income, ['Net Income','NetIncomeLoss','Net Income Common Stockholders','netIncome','Net income'])
    equity = get_first_available(bal, ['Total Stockholder Equity','Total shareholders equity','Total stockholders equity','Total Equity','totalStockholderEquity'])
    roe_pct = (net / equity * 100.0) if (net not in (None,0) and equity not in (None,0)) else None
    roe_q = score_roe_quality(roe_pct)
    revenue = get_first_available(income, ['Total Revenue','Revenue','totalRevenue'])
    total_assets = get_first_available(bal, ['Total Assets','totalAssets'])
    at_score = 50.0
    if revenue not in (None,0) and total_assets not in (None,0):
        at = safe_div(revenue, total_assets, None)
        if at is None:
            at_score = 50.0
        else:
            at_score = 100.0 if at >= 1.0 else (70.0 if at >= 0.6 else 40.0)
    ocf = get_first_available(cf, ['Operating Cash Flow','Total Cash From Operating Activities','netCashProvidedByOperatingActivities'])
    ccc_score = 50.0
    if ocf not in (None,0) and revenue not in (None,0):
        ratio = safe_div(abs(ocf), abs(revenue), None)
        if ratio is not None:
            ccc_score = 100.0 if ratio >= 0.15 else (70.0 if ratio >= 0.08 else 40.0)
    op_score = roe_q * 0.6 + ((at_score + ccc_score) / 2.0) * 0.4
    return {'roe_pct': roe_pct, 'roe_q': roe_q, 'at_score': at_score, 'ccc_score': ccc_score, 'operation_score': op_score}

def compute_bankruptcy(yfdata: Dict[str,Any]) -> Dict[str,Any]:
    income = yfdata['income']; bal = yfdata['balance']; cf = yfdata['cashflow']; info = yfdata['info']
    ebit = get_first_available(income, ['Ebit','EBIT','ebit','Operating Income','operatingIncome'])
    interest = get_first_available(income, ['Interest Expense','interestExpense']) or info.get('interestExpense')
    interest_coverage = safe_div(ebit, abs(interest), None) if (ebit not in (None,0) and interest not in (None,0)) else None
    ca = get_first_available(bal, ['Current Assets','currentAssets']); cl = get_first_available(bal, ['Current Liabilities','currentLiabilities'])
    current_ratio = safe_div(ca, cl, None) if (ca not in (None,0) and cl not in (None,0)) else None
    op_hist = []
    for k in ['Operating Cash Flow','Total Cash From Operating Activities','netCashProvidedByOperatingActivities']:
        if k in cf.index:
            try:
                vals = cf.loc[k].dropna().values.tolist()
                op_hist = vals[:3]
                break
            except Exception:
                continue
    total_debt = info.get('totalDebt'); ebitda = info.get('ebitda')
    debt_ebitda = safe_div(total_debt, ebitda, None) if (total_debt not in (None,0) and ebitda not in (None,0)) else None
    def map_ic(ic):
        if ic is None: return 50.0
        if ic > 5: return 100.0
        if 3 <= ic <= 5: return 70.0
        if 2 <= ic < 3: return 40.0
        return 0.0
    def map_cr(cr):
        if cr is None: return 50.0
        if cr > 1.5: return 100.0
        if 1.2 <= cr <= 1.5: return 70.0
        if 1.0 <= cr < 1.2: return 40.0
        return 0.0
    def map_ocf(hist):
        if not hist: return 50.0
        pos = sum(1 for v in hist if v is not None and v > 0)
        if pos >= 3: return 100.0
        if pos == 2: return 60.0
        if pos == 1: return 40.0
        return 0.0
    def map_de(d):
        if d is None: return 50.0
        if d > 8: return 0.0
        if d > 5: return 40.0
        if d > 3: return 70.0
        return 100.0
    ic_s = map_ic(interest_coverage); cr_s = map_cr(current_ratio); ocf_s = map_ocf(op_hist); de_s = map_de(debt_ebitda)
    bankruptcy_score = ic_s * 0.4 + ocf_s * 0.3 + de_s * 0.3
    return {'interest_coverage': interest_coverage, 'current_ratio': current_ratio, 'op_hist': op_hist, 'debt_ebitda': debt_ebitda, 'ic_s': ic_s, 'cr_s': cr_s, 'ocf_s': ocf_s, 'de_s': de_s, 'bankruptcy_score': bankruptcy_score}

def compute_valuation(yfdata: Dict[str,Any]) -> Dict[str,Any]:
    info = yfdata['info']
    pe = info.get('trailingPE') or info.get('forwardPE')
    pb = info.get('priceToBook')
    fcf = info.get('freeCashflow')
    market_cap = info.get('marketCap')
    def pe_map(p):
        if p is None: return 50.0
        if p < 10: return 100.0
        if p < 15: return 80.0
        if p < 25: return 60.0
        return 40.0
    def pb_map(p):
        if p is None: return 50.0
        if p < 1: return 100.0
        if p < 2: return 80.0
        if p < 3: return 60.0
        return 40.0
    rel = (pe_map(pe) + pb_map(pb)) / 2.0
    abs_s = 60.0
    fcf_yield = None
    if fcf not in (None,0) and market_cap not in (None,0):
        fcf_yield = (fcf / market_cap) * 100.0
        if fcf_yield > 8: abs_s = 100.0
        elif fcf_yield >= 5: abs_s = 80.0
        elif fcf_yield >= 3: abs_s = 60.0
        else: abs_s = 40.0
    val_margin = 0.5 * rel + 0.5 * abs_s
    return {'pe': pe, 'pb': pb, 'fcf_yield': fcf_yield, 'rel': rel, 'abs': abs_s, 'val_margin': val_margin}

# -------------------------
# Veto checks
# -------------------------
def check_vetoes(merrill_pct, porter_raw, vc_dict, bankruptcy_dict, operation_dict, info) -> Dict[str,Any]:
    veto = {k: False for k in [
        'merrill_veto','porter_veto','value_chain_veto','tech_support_veto',
        'financial_veto_interest_coverage','financial_veto_cashflow','financial_veto_debt','final_financial_zeroed',
        'composite_basic_veto'
    ]}
    if merrill_pct < 20: veto['merrill_veto'] = True
    if porter_raw is not None and porter_raw < 2.0: veto['porter_veto'] = True
    if vc_dict:
        low_main = sum(1 for v in vc_dict.get('main_scores', {}).values() if v <= 3)
        if low_main >= 3: veto['value_chain_veto'] = True
        support = vc_dict.get('support_scores', {})
        if support.get('R&D', 5) == 0 or support.get('Infrastructure', 5) <= 2:
            veto['tech_support_veto'] = True
    ic = bankruptcy_dict.get('interest_coverage')
    if ic is not None and ic < 1.5:
        veto['financial_veto_interest_coverage'] = True
    op_hist = bankruptcy_dict.get('op_hist', [])
    negative_seq = 0
    for v in (op_hist[:3]):
        if v is not None and v < 0: negative_seq += 1
        else: negative_seq = 0
    cash = info.get('totalCash') or info.get('cash') or 0
    # attempt quarterly revenue (best effort)
    qrev = None
    try:
        qrev = yfdata_local.get('q_income')
        if qrev is not None and not qrev.empty:
            qrev_val = get_first_available(qrev, ['Total Revenue','Revenue','totalRevenue'])
            qrev = qrev_val
    except Exception:
        qrev = None
    cash_ratio = None
    if qrev not in (None,0):
        try:
            cash_ratio = safe_div(cash, abs(qrev), None)
        except Exception:
            cash_ratio = None
    if negative_seq >= 2 and (cash_ratio is not None and cash_ratio < 0.3):
        veto['financial_veto_cashflow'] = True
    if bankruptcy_dict.get('debt_ebitda') is not None and bankruptcy_dict.get('debt_ebitda') > 8:
        veto['financial_veto_debt'] = True
    if any([veto['financial_veto_interest_coverage'], veto['financial_veto_cashflow'], veto['financial_veto_debt']]):
        veto['final_financial_zeroed'] = True
    return veto

# -------------------------
# Batch normalization & aggregation (percentile + optional zscale)
# -------------------------
def normalize_and_aggregate(raw_list: List[Dict[str,Any]], zscale: bool=False) -> pd.DataFrame:
    df = pd.DataFrame(raw_list)
    # ensure needed columns
    df['merrill_pct'] = df['merrill_pct'].fillna(50)
    df['porter_pct'] = df['porter_pct'].fillna(50)
    df['value_chain_pct'] = df['value_chain_pct'].fillna(50)
    df['operation_score'] = df['operation_score'].fillna(50)
    df['bankruptcy_score'] = df['bankruptcy_score'].fillna(50)
    df['val_margin'] = df['val_margin'].fillna(50)
    # percentile ranks
    df['merrill_pct_pr'] = stats.rankdata(df['merrill_pct'], method='average') / len(df) * 100.0
    df['porter_pct_pr'] = stats.rankdata(df['porter_pct'], method='average') / len(df) * 100.0
    df['value_chain_pct_pr'] = stats.rankdata(df['value_chain_pct'], method='average') / len(df) * 100.0
    df['operation_score_pr'] = stats.rankdata(df['operation_score'], method='average') / len(df) * 100.0
    df['bankruptcy_score_pr'] = stats.rankdata(df['bankruptcy_score'], method='average') / len(df) * 100.0
    df['val_margin_pr'] = stats.rankdata(df['val_margin'], method='average') / len(df) * 100.0
    # financial composite uses percentile ranks (50/20/30)
    df['financial_pct'] = df['operation_score_pr'] * 0.5 + df['bankruptcy_score_pr'] * 0.2 + df['val_margin_pr'] * 0.3
    # final aggregation
    df['final_pct'] = (df['merrill_pct_pr'] * WEIGHTS['merrill'] +
                       df['porter_pct_pr'] * WEIGHTS['porter'] +
                       df['value_chain_pct_pr'] * WEIGHTS['value_chain'] +
                       df['financial_pct'] * WEIGHTS['financial'])
    if zscale:
        df['final_pct'] = (df['final_pct'] - df['final_pct'].mean()) / (df['final_pct'].std(ddof=0) + 1e-9) * 10 + 50
    # assign label/position
    def map_label(score):
        for lo,hi,lab,pos in POSITION_MAP:
            if lo <= score <= hi:
                return lab, pos
        return '<65 (Avoid)', 0.0
    mapped = df['final_pct'].apply(lambda x: pd.Series(map_label(x)))
    df['label'] = mapped[0]; df['position_pct'] = mapped[1]
    return df

# -------------------------
# Visualizations (Plotly)
# -------------------------
def radar_plot_scores(ticker_name: str, m, p, v, f, tech=10):
    labels = ['Merrill','Porter','ValueChain','Financial','Technical']
    vals = [m, p, v, f, tech]
    fig = go.Figure(data=go.Scatterpolar(r=vals, theta=labels, fill='toself', name=ticker_name))
    fig.update_layout(title_text=f"{ticker_name} - Component Radar", polar=dict(radialaxis=dict(range=[0,100])), showlegend=False)
    return fig

def bar_component_plot(ticker_name: str, m, p, v, f):
    fig = go.Figure([go.Bar(x=['Merrill','Porter','ValueChain','Financial'], y=[m,p,v,f], marker_color=['#636EFA','#EF553B','#00CC96','#AB63FA'])])
    fig.update_layout(title=f"{ticker_name} - Component Scores", yaxis=dict(range=[0,100]))
    return fig

def timeseries_financial_plot(yfdata: Dict[str,Any]):
    # try to plot revenue and net income if available
    income = yfdata['income']
    if income is None or income.empty:
        return None
    # convert columns (each column is a year); build series
    def col_to_series(df, key):
        if key in df.index:
            s = df.loc[key].dropna()
            s = s.astype(float)
            s = s[::-1]  # chronological order
            return s
        return None
    rev = col_to_series(income, 'Total Revenue') or col_to_series(income, 'Revenue') or None
    net = col_to_series(income, 'Net Income') or col_to_series(income, 'NetIncomeLoss') or None
    if rev is None and net is None:
        return None
    fig = go.Figure()
    if rev is not None:
        fig.add_trace(go.Scatter(x=rev.index.astype(str), y=rev.values, name='Revenue', mode='lines+markers'))
    if net is not None:
        fig.add_trace(go.Scatter(x=net.index.astype(str), y=net.values, name='Net Income', mode='lines+markers'))
    fig.update_layout(title="Financials (Annual)", xaxis_title="Period", yaxis_title="Amount (local)", template='plotly_white')
    return fig

# -------------------------
# App UI
# -------------------------
st.title("Automated Fundamental & Financial Analyzer — Streamlit App (Single File)")

with st.sidebar:
    st.header("Run settings")
    ticker_input = st.text_input("Tickers (comma separated)", value="AAPL,MSFT,TSLA")
    phase = st.selectbox("Merrill Phase", options=['recovery','overheat','stagflation','recession'], index=0)
    deepseek_key = st.text_input("DeepSeek API Key (optional)", value="", type="password")
    zscale = st.checkbox("Apply z-score exaggeration to final scores (more dispersion)", value=False)
    st.markdown("**Notes:** Value-chain auto uses DeepSeek if key provided; otherwise heuristics. Manual override sliders available per ticker.")
    if 'deepseek_key' not in st.session_state:
        st.session_state['deepseek_key'] = deepseek_key

run_btn = st.button("Run Full Batch Analysis")

# container for dynamic results
results_container = st.empty()

if run_btn:
    st.session_state['deepseek_key'] = deepseek_key or None
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    if not tickers:
        st.error("Provide at least one ticker.")
    else:
        raw_list = []
        detailed = []
        progress = st.progress(0)
        for idx, tk in enumerate(tickers):
            progress.progress(int((idx+1)/len(tickers)*100))
            try:
                yfdata = fetch_yf(tk)
                # save yfdata to local for veto attempts
                global yfdata_local
                yfdata_local = yfdata
                merrill_pct, merr_reason = merrill_clock_score(yfdata['sector'], phase)
                porter = porter_auto(yfdata['sector'], yfdata['industry'], yfdata['summary'])
                vc_auto = value_chain_auto(yfdata['summary'], yfdata['info'])
                op = compute_operational(yfdata)
                bk = compute_bankruptcy(yfdata)
                vl = compute_valuation(yfdata)
                financial_pct_before_veto = op['operation_score'] * 0.5 + bk['bankruptcy_score'] * 0.2 + vl['val_margin'] * 0.3
                raw = {
                    'ticker': tk,
                    'company': yfdata['info'].get('shortName') or yfdata['info'].get('longName') or tk,
                    'yfdata': yfdata,
                    'merrill_pct': merrill_pct,
                    'merrill_reason': merr_reason,
                    'porter_pct': porter['porter_pct'],
                    'porter_raw': porter['porter_raw'],
                    'vc_auto': vc_auto,
                    'value_chain_pct': vc_auto['value_chain_pct'],
                    'operation_score': op['operation_score'],
                    'bankruptcy_score': bk['bankruptcy_score'],
                    'val_margin': vl['val_margin'],
                    'financial_pct_before_veto': round(financial_pct_before_veto,2),
                    'technical_pct': 10  # placeholder
                }
                raw_list.append(raw)
            except Exception as e:
                st.warning(f"Failed to fetch {tk}: {e}")
            time.sleep(0.5)
        # normalize and aggregate
        dfagg = normalize_and_aggregate([{
            'ticker': r['ticker'],
            'company': r['company'],
            'merrill_pct': r['merrill_pct'],
            'porter_pct': r['porter_pct'],
            'value_chain_pct': r['value_chain_pct'],
            'operation_score': r['operation_score'],
            'bankruptcy_score': r['bankruptcy_score'],
            'val_margin': r['val_margin'],
            'financial_pct_before_veto': r['financial_pct_before_veto']
        } for r in raw_list], zscale=zscale)
        # combine detailed results
        merged = []
        for i, row in dfagg.iterrows():
            raw = raw_list[i]
            veto = check_vetoes(raw['merrill_pct'], raw['porter_raw'], raw['vc_auto'], 
                                {'interest_coverage': raw['yfdata']['info'].get('ebit'), 'op_hist': raw['yfdata']['cashflow'].index.tolist() }, 
                                {}, raw['yfdata']['info'])
            merged.append({**raw, **row.to_dict(), 'veto': veto})
        # store in session
        st.session_state['last_df'] = dfagg
        st.session_state['detailed'] = merged
        progress.progress(100)
        # display summary and visuals
        results_container.subheader("Batch Summary")
        st.dataframe(dfagg[['ticker','company','final_pct','label','position_pct']].rename(columns={'position_pct':'position'}))
        st.success("Batch analysis complete.")
        # allow selection of ticker for deep dive
        sel = st.selectbox("Select ticker for detailed view", options=[r['ticker'] for r in merged])
        detail = next((r for r in merged if r['ticker']==sel), None)
        if detail:
            st.markdown(f"### {detail['ticker']} — {detail['company']}")
            st.write("**Sector:**", detail['yfdata']['sector'], "| **Industry:**", detail['yfdata']['industry'])
            # show small cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Score", f"{detail['final_pct']:.2f}")
            col2.metric("Label", detail['label'])
            col3.metric("Suggested Position", f"{detail['position_pct']*100:.1f}%")
            # plots
            st.plotly_chart(radar_plot_scores(detail['ticker'], detail['merrill_pct'], detail['porter_pct'], detail['value_chain_pct'], detail['financial_pct_before_veto'], detail['technical_pct']), use_container_width=True)
            st.plotly_chart(bar_component_plot(detail['ticker'], detail['merrill_pct'], detail['porter_pct'], detail['value_chain_pct'], detail['financial_pct_before_veto']), use_container_width=True)
            ts_fig = timeseries_financial_plot(detail['yfdata'])
            if ts_fig:
                st.plotly_chart(ts_fig, use_container_width=True)
            else:
                st.info("No detailed time-series financials available for this company.")
            # show value chain auto vs manual sliders
            st.subheader("Value Chain (Auto suggestion and manual override)")
            vc_auto = detail['vc_auto']
            cols = st.columns(2)
            with cols[0]:
                st.write("Auto Main scores (1-10):")
                st.write(vc_auto['main_scores'])
                st.write("Auto Support scores (1-10):")
                st.write(vc_auto['support_scores'])
            with cols[1]:
                st.write("Manual override sliders (set and click 'Apply Manual VC')")
                manual_main = {}
                manual_support = {}
                for k,v in vc_auto['main_scores'].items():
                    manual_main[k] = st.slider(f"Main - {k}", min_value=0, max_value=10, value=int(v), key=f"main_{sel}_{k}")
                for k,v in vc_auto['support_scores'].items():
                    manual_support[k] = st.slider(f"Support - {k}", min_value=0, max_value=10, value=int(v), key=f"sup_{sel}_{k}")
                if st.button("Apply Manual VC", key=f"applyvc_{sel}"):
                    main_avg = np.mean(list(manual_main.values())); support_avg = np.mean(list(manual_support.values()))
                    vc_score = (main_avg * 0.6 + support_avg * 0.4)
                    vc_pct = (vc_score / 10.0) * 100.0
                    # update dfagg and session
                    st.session_state['last_df'].loc[st.session_state['last_df']['ticker']==sel, 'value_chain_pct'] = vc_pct
                    # re-normalize entire batch
                    raw_list_updated = []
                    for d in merged:
                        if d['ticker'] == sel:
                            raw_list_updated.append({
                                'ticker': d['ticker'],
                                'company': d['company'],
                                'merrill_pct': d['merrill_pct'],
                                'porter_pct': d['porter_pct'],
                                'value_chain_pct': vc_pct,
                                'operation_score': d['operation_score'],
                                'bankruptcy_score': d['bankruptcy_score'],
                                'val_margin': d['val_margin'],
                                'financial_pct_before_veto': d['financial_pct_before_veto']
                            })
                        else:
                            raw_list_updated.append({
                                'ticker': d['ticker'],
                                'company': d['company'],
                                'merrill_pct': d['merrill_pct'],
                                'porter_pct': d['porter_pct'],
                                'value_chain_pct': d['value_chain_pct'],
                                'operation_score': d['operation_score'],
                                'bankruptcy_score': d['bankruptcy_score'],
                                'val_margin': d['val_margin'],
                                'financial_pct_before_veto': d['financial_pct_before_veto']
                            })
                    dfagg_new = normalize_and_aggregate(raw_list_updated, zscale=zscale)
                    st.session_state['last_df'] = dfagg_new
                    st.success("Manual Value Chain applied and batch re-normalized.")
                    st.experimental_rerun()
        # export
        csv = dfagg.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='analysis_results.csv', mime='text/csv')

# If previous session results exist, show quick access
if 'last_df' in st.session_state and st.session_state['last_df'] is not None and not run_btn:
    st.sidebar.markdown("### Previous batch results available")
    if st.sidebar.button("Show previous summary"):
        st.dataframe(st.session_state['last_df'][['ticker','company','final_pct','label','position_pct']])

# -------------------------
# End of app
# -------------------------

