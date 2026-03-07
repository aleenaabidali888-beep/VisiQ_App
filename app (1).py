"""
VisiQ — AI Data Intelligence (Streamlit Edition)
Matches features of the original HTML app:
  · File upload (CSV, Excel, JSON, TSV) + text/manual input + sample datasets
  · 9 chart types: Pie, Line, Bar, Radar, Scatter, Histogram, Trend, Grouped Bar, H-Bar
  · Stats row + Model Accuracy (R2, MAE, RMSE, MAPE, Forecast Acc.)
  · Data Quality gauge (completeness, consistency, uniformity)
  · AI Insights panel
  · AI Summary panel
  · Data Science Workbench (Column Profiling, Deep Stats, Distributions,
    Scatter Matrix, Advanced Charts, Correlation, Data Quality report)

Deploy:
    pip install streamlit pandas numpy plotly openpyxl scipy
    streamlit run visiq_app.py
"""

import io, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisiQ — AI Data Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── BRAND COLORS ──────────────────────────────────────────────────────────────
COBALT   = "#1639d6"
TEAL     = "#0d9e7a"
AMBER    = "#b86c00"
ROSE     = "#c0313a"
VIOLET   = "#6d28d9"
COLORS   = [COBALT, TEAL, AMBER, ROSE, VIOLET, "#0891b2", "#be185d", "#15803d", "#b45309", "#1d4ed8"]

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] { background: #0a0c12 !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }

[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e8eaf2;
    border-radius: 12px; padding: 14px 18px !important;
    box-shadow: 0 1px 3px rgba(10,12,18,.06);
}
[data-testid="stMetricLabel"] { font-size: 10px !important; text-transform: uppercase; letter-spacing: .12em; color: #7a849e !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700 !important; color: #0a0c12 !important; }

.hero-banner {
    background: #0a0c12; border-radius: 16px; padding: 32px 38px; margin-bottom: 24px;
    background-image: linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
                      linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
    background-size: 40px 40px;
}
.hero-title { font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 8px; }
.hero-title em { color: #7d9bf8; font-style: italic; }
.hero-desc  { font-size: 13px; color: rgba(255,255,255,.65); margin-bottom: 14px; }
.hero-tag {
    display: inline-block; font-size: 11px; font-weight: 500; padding: 3px 12px;
    border-radius: 20px; background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.14); color: rgba(255,255,255,.85); margin: 2px;
}
.section-title { font-size: 16px; font-weight: 700; color: #0a0c12; margin-bottom: 4px; }

.insight-card {
    background: #fff; border: 1px solid #e8eaf2; border-radius: 12px;
    padding: 14px 16px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(10,12,18,.06);
}
.gauge-card {
    background: #fff; border: 1px solid #e8eaf2; border-radius: 12px;
    padding: 22px 24px; box-shadow: 0 1px 3px rgba(10,12,18,.06);
    height: 100%;
}
.mbar-wrap { margin-bottom: 12px; }
.mbar-label { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; color: #1c2033; margin-bottom: 4px; }
.mbar-track { height: 7px; background: #e8eaf2; border-radius: 4px; overflow: hidden; }
.sum-block {
    background: #f8f9fc; border: 1px solid #e8eaf2; border-radius: 10px;
    padding: 16px; margin-bottom: 12px;
}
.sum-block-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: #7a849e; margin-bottom: 10px; }
.sum-line { display: flex; justify-content: space-between; font-size: 12.5px; color: #374061; padding: 4px 0; border-bottom: 1px solid #f2f4f9; }
.sum-line-val { font-weight: 600; font-family: monospace; color: #0a0c12; }
.verdict-box {
    background: linear-gradient(135deg, rgba(22,57,214,.06), rgba(13,158,122,.06));
    border: 1px solid rgba(22,57,214,.15); border-radius: 12px; padding: 18px 22px; margin-top: 14px;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt(v):
    try:
        f = float(v)
        if abs(f) >= 1_000_000: return f"{f/1_000_000:.1f}M"
        if abs(f) >= 1_000:     return f"{f/1_000:.1f}K"
        return f"{f:.2f}"
    except: return str(v)

def linreg(xs, ys):
    n = len(xs)
    if n < 2: return 0.0, float(np.mean(ys)) if ys else 0.0, 0.0
    sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    d = n*sxx-sx*sx
    if d==0: return 0.0, sy/n, 0.0
    slope=(n*sxy-sx*sy)/d; intercept=(sy-slope*sx)/n
    ym=sy/n; ss_tot=sum((y-ym)**2 for y in ys); ss_res=sum((y-(slope*x+intercept))**2 for x,y in zip(xs,ys))
    return slope, intercept, max(0.0, 1-ss_res/ss_tot) if ss_tot>0 else 0.0

def compute_acc(vals):
    if len(vals)<3: return None
    xs=list(range(len(vals))); ys=list(vals)
    slope,intercept,r2=linreg(xs,ys); n=len(vals)
    mae=sum(abs(v-(slope*i+intercept)) for i,v in enumerate(ys))/n
    rmse=math.sqrt(sum((v-(slope*i+intercept))**2 for i,v in enumerate(ys))/n)
    mape=sum(abs((v-(slope*i+intercept))/v) for i,v in enumerate(ys) if v!=0)/n*100
    return dict(r2=round(r2,4),mae=round(mae,4),rmse=round(rmse,4),
                mape=round(mape,1),forecast_acc=round(max(0,min(75,100-mape)),1),
                slope=round(slope,4),intercept=round(intercept,4))

def mbar(icon, label, count, pct, color, note=""):
    p=min(100,float(pct))
    return (f'<div class="mbar-wrap">'
            f'<div class="mbar-label"><span>{icon} {label}</span>'
            f'<span style="font-family:monospace;color:{color}">{count}&nbsp;<small>({pct}%)</small></span></div>'
            f'<div class="mbar-track"><div style="height:7px;width:{p}%;background:{color};border-radius:4px"></div></div>'
            +(f'<div style="font-size:10.5px;color:#7a849e;margin-top:2px">{note}</div>' if note else '')
            +'</div>')

def load_file(f):
    ext=Path(f.name).suffix.lower()
    if ext in(".xlsx",".xls"): return pd.read_excel(f)
    if ext==".json":
        raw=json.load(f)
        if isinstance(raw,list): return pd.DataFrame(raw)
        if isinstance(raw,dict):
            k=next((k for k,v in raw.items() if isinstance(v,list)),None)
            return pd.DataFrame(raw[k]) if k else pd.DataFrame([raw])
    if ext==".tsv": return pd.read_csv(f,sep="\t")
    try: return pd.read_csv(f)
    except: f.seek(0); return pd.read_csv(f,sep=None,engine="python")

def parse_text(text):
    pairs=[]
    for m in re.finditer(r'([A-Za-z][^:,\n]{0,40}):\s*([\d,.]+)\s*%?',text):
        pairs.append((m.group(1).strip(), float(m.group(2).replace(",",""))))
    if not pairs:
        for m in re.finditer(r'(\b[A-Za-z][^\d\n]{0,20})?\s*([\d,]+(?:\.\d+)?)',text):
            lbl=(m.group(1) or "").strip() or f"Item {len(pairs)+1}"
            pairs.append((lbl, float(m.group(2).replace(",",""))))
    return pairs[:25]

SAMPLES={
    "Revenue Q4 2024":"Revenue Q4 2024: Product Sales 45%, Services 28%, Subscriptions 18%, Licensing 9%",
    "Monthly Users":  "Monthly Users — Jan: 12000, Feb: 13500, Mar: 14200, Apr: 15800, May: 17200, Jun: 19500, Jul: 21000, Aug: 23400",
    "Market Share":   "Market Share — Alpha Corp: 34%, Beta Ltd: 27%, Gamma Inc: 19%, Delta: 12%, Others: 8%",
    "Budget FY2025":  "Budget FY2025: R&D 30%, Marketing 25%, Operations 22%, HR 13%, Misc 10%",
    "Quarterly KPI":  "Quarterly KPI — Q1: 72, Q2: 78, Q3: 85, Q4: 91",
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:22px;font-weight:700;color:#fff'>🔍 VisiQ</div>"
                "<div style='font-size:10px;letter-spacing:.18em;text-transform:uppercase;"
                "color:rgba(255,255,255,.3);margin-bottom:16px'>Intelligence Platform</div>",
                unsafe_allow_html=True)
    for item in ["📊 Dashboard","📈 Analytics","🔬 AI Insights","📋 Reports","🗄️ Data Sources"]:
        st.markdown(f"<div style='padding:8px 0;font-size:13px'>{item}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**⚙️ Settings**")
    z_thresh = st.slider("Anomaly Z-threshold", 1.0, 4.0, 2.0, 0.1)
    st.divider()
    st.markdown("<div style='font-size:11px;color:rgba(255,255,255,.35)'>🟢 ML Engine Active</div>"
                "<div style='font-size:13px;color:rgba(255,255,255,.5);margin-top:8px'>"
                "Made by <span style='color:#7d9bf8'>Aleena</span></div>", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#7d9bf8;margin-bottom:10px">— VisiQ Intelligence Platform —</div>
  <div class="hero-title">Transform Data into <em>Intelligent Insight</em></div>
  <div class="hero-desc">Upload your CSV / Excel / JSON dataset — or type data manually — and watch VisiQ's ML engine render charts, detect anomalies, and generate a full statistical summary.</div>
  <div>
    <span class="hero-tag">📁 File Upload</span><span class="hero-tag">📊 9 Chart Types</span>
    <span class="hero-tag">🤖 ML Regression</span><span class="hero-tag">🔮 Forecasting</span>
    <span class="hero-tag">⚡ Anomaly Detection</span><span class="hero-tag">🔬 Data Science Mode</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📥 Data Input</div>', unsafe_allow_html=True)
tab_file, tab_text = st.tabs(["📁 File Upload", "✏️ Text / Manual Input"])

df=None; pairs=[]; label_col=value_col=None

with tab_file:
    uploaded=st.file_uploader("Drop file here", type=["csv","xlsx","xls","json","tsv"],
                               label_visibility="collapsed")
    if uploaded:
        try:
            df=load_file(uploaded)
            st.success(f"✅ **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} cols · {uploaded.size/1024:.1f} KB")
            c1,c2=st.columns(2)
            label_col=c1.selectbox("Label column", df.columns.tolist(), 0, key="lc")
            num_cols=df.select_dtypes(include="number").columns.tolist()
            dv=num_cols[0] if num_cols else (df.columns[1] if len(df.columns)>1 else df.columns[0])
            value_col=c2.selectbox("Value column", df.columns.tolist(),
                                   df.columns.tolist().index(dv), key="vc")
            with st.expander("Preview (first 6 rows)"):
                st.dataframe(df[[label_col,value_col]].head(6), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not load: {e}")

with tab_text:
    sc=st.selectbox("Load sample", ["— none —"]+list(SAMPLES.keys()), key="sc")
    user_text=st.text_area("Type or paste data",
                           value=SAMPLES[sc] if sc!="— none —" else "",
                           height=110,
                           placeholder="Jan: 12000, Feb: 13500, Mar: 14200 …")
    if user_text.strip():
        pairs=parse_text(user_text)
        if pairs: st.caption(f"Parsed {len(pairs)} data points")

run=st.button("🚀 Analyze", type="primary", use_container_width=True)

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
if run or (sc!="— none —" and user_text):

    if df is not None and label_col and value_col:
        src_labels=df[label_col].astype(str).tolist()
        raw_vals=pd.to_numeric(df[value_col].astype(str).str.replace(r"[$,%\s]","",regex=True),errors="coerce")
        skipped=int(raw_vals.isna().sum())
        src_vals=raw_vals.dropna().tolist()
        clean_labels=[src_labels[i] for i in raw_vals.dropna().index]
        data_pairs=list(zip(clean_labels,src_vals))[:25]
        total_rows=len(df); file_headers=df.columns.tolist()
        empty_cells=int((df=="").sum().sum())+int(df.isnull().sum().sum())
        total_cells=df.size; source_name=uploaded.name
    elif pairs:
        data_pairs=pairs[:25]; src_vals=[v for _,v in data_pairs]
        clean_labels=[l for l,_ in data_pairs]; skipped=0
        total_rows=len(data_pairs); total_cells=len(data_pairs)
        empty_cells=0; file_headers=["Label","Value"]; source_name="text input"
        df=pd.DataFrame(data_pairs,columns=["Label","Value"])
        label_col,value_col="Label","Value"
    else:
        st.warning("No data to analyze. Upload a file or type data above.")
        st.stop()

    if len(src_vals)<2:
        st.error("Need at least 2 numeric values."); st.stop()

    labels=[l[:25] for l in clean_labels]; vals=src_vals
    xs=list(range(len(vals))); n=len(vals)
    tot=sum(vals); avg=tot/n
    med=float(np.median(vals)); sd=float(np.std(vals)) or 1.0
    vmin=min(vals); vmax=max(vals); cv=round(abs(sd/avg*100),1) if avg!=0 else 0.0

    slope,intercept,r2=linreg(xs,vals)
    acc=compute_acc(vals)
    fc3=[round(slope*(n+i)+intercept,2) for i in range(1,4)]

    anom_mask=[abs(v-avg)>z_thresh*sd for v in vals]
    anom_list=[(labels[i],vals[i],round((vals[i]-avg)/sd,2)) for i,f in enumerate(anom_mask) if f]

    nan_pct  =round(skipped/max(1,total_rows)*100,1)
    miss_pct =round(empty_cells/max(1,total_cells)*100,1)
    anom_pct =round(len(anom_list)/max(1,n)*100,1)
    completeness=max(0.0,100-nan_pct); consistency=max(0.0,100-anom_pct*3)
    uniformity=100 if cv<20 else max(10,round(100-cv,1))
    qs=round(completeness*.4+consistency*.35+uniformity*.25,1)
    qg="Excellent" if qs>=90 else "Good" if qs>=75 else "Fair" if qs>=55 else "Poor"
    qc=TEAL if qs>=90 else COBALT if qs>=75 else AMBER if qs>=55 else ROSE
    qd=("Very clean. Low noise." if qs>=90 else "Good quality with minor issues." if qs>=75
        else "Fair — notable errors or outliers." if qs>=55 else "Poor quality — significant errors.")

    st.divider()

    # Dataset summary
    ds_items=[("Rows",f"{total_rows:,}"),("Columns",len(file_headers)),("Plotted",n),
              ("Total",fmt(tot)),("Average",fmt(avg)),("Max",fmt(vmax))]
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(22,57,214,.05),rgba(13,158,122,.05));'
        f'border:1px solid rgba(22,57,214,.12);border-radius:12px;padding:16px 22px;margin-bottom:20px">'
        f'<div style="font-size:17px;font-weight:700;color:#0a0c12;margin-bottom:4px">📄 {source_name}</div>'
        f'<div style="font-size:12px;color:#7a849e;margin-bottom:12px">Label: "{label_col}" · Value: "{value_col}" · {total_rows} rows</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px">'
        +''.join(f'<div style="background:#fff;border:1px solid #e8eaf2;border-radius:8px;padding:8px 14px">'
                 f'<div style="font-size:18px;font-weight:700;color:#0a0c12;line-height:1">{v}</div>'
                 f'<div style="font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:#7a849e;margin-top:3px">{l}</div></div>'
                 for l,v in ds_items)
        +'</div></div>', unsafe_allow_html=True)

    # Stats row
    st.markdown('<div class="section-title">📊 Key Statistics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Data Points",f"{n:,}"); c2.metric("Total Value",fmt(tot))
    c3.metric("Average",fmt(avg));     c4.metric("Minimum",fmt(vmin)); c5.metric("Maximum",fmt(vmax))

    # Accuracy row
    if acc:
        st.markdown('<div class="section-title" style="margin-top:20px">🎯 Model Accuracy</div>', unsafe_allow_html=True)
        st.caption("Linear regression model — meaningful for sequential/time-series data.")
        fa=acc["forecast_acc"]; ag="Excellent" if fa>=90 else "Good" if fa>=75 else "Moderate" if fa>=55 else "Weak"
        a1,a2,a3,a4,a5=st.columns(5)
        a1.metric("R² Score",str(acc["r2"]),delta="Model Fit")
        a2.metric("MAE",fmt(acc["mae"])); a3.metric("RMSE",fmt(acc["rmse"]))
        a4.metric("MAPE",f"{acc['mape']}%"); a5.metric("Forecast Acc.",f"{fa}%",delta=ag)
    else:
        fa=0; ag="N/A"

    st.divider()

    # ── 9 CHARTS ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Charts</div>', unsafe_allow_html=True)

    reg_y=[slope*i+intercept for i in xs]

    def ch_pie():
        fig=go.Figure(go.Pie(labels=labels,values=vals,
            marker=dict(colors=COLORS[:len(vals)],line=dict(color="#fff",width=2)),
            hovertemplate="%{label}: %{value}<extra></extra>",textposition="inside"))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=270,
                          legend=dict(font=dict(size=10)))
        return fig

    def ch_line():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=labels,y=vals,mode="lines+markers",
            line=dict(color=COBALT,width=2.5),
            marker=dict(color=COBALT,size=6,line=dict(color="#fff",width=2)),
            fill="tozeroy",fillcolor="rgba(22,57,214,.08)"))
        fig.add_trace(go.Scatter(x=labels,y=reg_y,mode="lines",name="Trend",
            line=dict(color=ROSE,width=1.5,dash="dash"),hoverinfo="skip"))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=270,showlegend=False,
                          xaxis=dict(showgrid=True,gridcolor="#f2f4f9"),
                          yaxis=dict(showgrid=True,gridcolor="#f2f4f9"))
        return fig

    def ch_bar():
        colors=[ROSE if f else COBALT for f in anom_mask]
        fig=go.Figure(go.Bar(x=labels,y=vals,marker_color=colors,
            hovertemplate="%{x}: %{y}<extra></extra>"))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=270,
                          xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#f2f4f9"))
        return fig

    def ch_radar():
        fig=go.Figure(go.Scatterpolar(r=vals[:12]+[vals[0]],theta=labels[:12]+[labels[0]],
            fill="toself",fillcolor="rgba(22,57,214,.12)",line=dict(color=COBALT,width=2)))
        fig.update_layout(polar=dict(radialaxis=dict(showticklabels=True,gridcolor="#e8eaf2"),
                                     angularaxis=dict(gridcolor="#e8eaf2")),
                          margin=dict(t=20,b=20,l=20,r=20),height=270,showlegend=False)
        return fig

    def ch_scatter():
        fig=go.Figure(go.Scatter(x=xs,y=vals,mode="markers",
            marker=dict(color=[COLORS[i%len(COLORS)] for i in range(len(vals))],
                        size=9,line=dict(color="#fff",width=1.5)),
            text=labels,hovertemplate="%{text}: %{y}<extra></extra>"))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=270,
                          xaxis=dict(showgrid=True,gridcolor="#f2f4f9"),
                          yaxis=dict(showgrid=True,gridcolor="#f2f4f9"))
        return fig

    def ch_hist():
        fig=go.Figure(go.Histogram(x=vals,nbinsx=min(20,max(5,n//3)),
            marker=dict(color=VIOLET+"cc",line=dict(color=VIOLET,width=1))))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=270,bargap=0.05,
                          xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#f2f4f9"))
        return fig

    def ch_trend():
        fig=go.Figure()
        fig.add_trace(go.Bar(x=labels,y=vals,marker_color=TEAL+"cc",name="Value"))
        fig.add_trace(go.Scatter(x=labels,y=reg_y,mode="lines",name=f"Trend (R²={r2:.3f})",
            line=dict(color=ROSE,width=2.5)))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=270,showlegend=True,
                          legend=dict(font=dict(size=10)),
                          xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#f2f4f9"))
        return fig

    def ch_grouped():
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Actual",x=labels,y=vals,marker_color=COBALT+"cc"))
        fig.add_trace(go.Bar(name="Predicted",x=labels,y=[round(v,2) for v in reg_y],marker_color=TEAL+"cc"))
        fig.update_layout(barmode="group",margin=dict(t=10,b=10,l=10,r=10),height=270,
                          legend=dict(font=dict(size=10)),
                          xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#f2f4f9"))
        return fig

    def ch_hbar():
        fig=go.Figure(go.Bar(y=labels,x=vals,orientation="h",
            marker=dict(color=[COLORS[i%len(COLORS)] for i in range(len(vals))],
                        line=dict(color="#fff",width=1)),
            hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=max(260,n*22),
                          xaxis=dict(showgrid=True,gridcolor="#f2f4f9"),
                          yaxis=dict(showgrid=False,autorange="reversed"))
        return fig

    CHARTS=[("🥧 Pie",ch_pie),("📈 Line",ch_line),("📊 Bar",ch_bar),
            ("🕷️ Radar",ch_radar),("⊹ Scatter",ch_scatter),("📉 Histogram",ch_hist),
            ("📐 Trend",ch_trend),("📚 Grouped Bar",ch_grouped),("↔️ H-Bar",ch_hbar)]

    for row_start in range(0,9,3):
        cols=st.columns(3)
        for i,col in enumerate(cols):
            idx=row_start+i
            if idx<len(CHARTS):
                name,fn=CHARTS[idx]
                with col:
                    st.caption(name)
                    st.plotly_chart(fn(),use_container_width=True,key=f"ch{idx}")

    st.divider()

    # ── ACCURACY & QUALITY CARDS ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Model Accuracy &amp; ✅ Data Quality</div>', unsafe_allow_html=True)
    qc1,qc2=st.columns(2)

    with qc1:
        fav=fa; acc_color=TEAL if fav>=65 else COBALT if fav>=45 else AMBER if fav>=25 else ROSE
        acc_grade_lbl="Good" if fav>=65 else "Moderate" if fav>=45 else "Weak" if fav>=25 else "Poor"
        pb=round(min(75,100-(acc["mape"] if acc else 100)),1)
        mp=round(nan_pct*0.5,1); ap=round(anom_pct*1.0,1)
        st.markdown(f"""
        <div class="gauge-card">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#7a849e;margin-bottom:12px">🎯 MODEL ACCURACY</div>
          <div style="display:flex;align-items:center;gap:18px;margin-bottom:16px">
            <div style="text-align:center">
              <div style="font-size:42px;font-weight:700;color:{acc_color};line-height:1">{fav}%</div>
              <div style="font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:#7a849e;margin-top:2px">Score</div>
            </div>
            <div>
              <div style="font-size:16px;font-weight:700;color:#0a0c12">{acc_grade_lbl}</div>
              <div style="font-size:11px;color:#7a849e">{"Forecasts reasonably reliable." if fav>=65 else "Moderate fit — some noise." if fav>=45 else "Low accuracy — noisy data."}</div>
            </div>
          </div>
          {mbar("🚫","NaN / Unparseable Rows",skipped,nan_pct,ROSE,f"{skipped} rows had no valid numeric value")}
          {mbar("⚠️","Anomalies (>2σ)",len(anom_list),anom_pct,AMBER,", ".join(f"{l}={fmt(v)} (z={z}σ)" for l,v,z in anom_list[:3]))}
          {mbar("❌","Missing Cells",empty_cells,miss_pct,ROSE,f"{empty_cells} empty/null cells across dataset")}
          {mbar("✅","Clean Data Points",n-len(anom_list),round(max(0,100-nan_pct-anom_pct),1),TEAL,"Valid, parseable, within normal range")}
          <div style="background:#f8f9fc;border:1px solid #e8eaf2;border-radius:8px;padding:12px;margin-top:12px">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#7a849e;margin-bottom:8px">Score Penalty Breakdown</div>
            <div style="font-size:12px;color:#374061;display:flex;flex-direction:column;gap:4px">
              <div style="display:flex;justify-content:space-between"><span>Base (MAPE)</span><span style="font-family:monospace">{pb}%</span></div>
              <div style="display:flex;justify-content:space-between;color:{ROSE}"><span>− NaN / missing rows</span><span style="font-family:monospace">−{mp}%</span></div>
              <div style="display:flex;justify-content:space-between;color:{AMBER}"><span>− Anomaly penalty</span><span style="font-family:monospace">−{ap}%</span></div>
              <div style="border-top:1px solid #e8eaf2;margin-top:5px;padding-top:5px;display:flex;justify-content:space-between;font-weight:700"><span>Final Accuracy</span><span style="color:{acc_color}">{fav}%</span></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with qc2:
        cw=round(max(0,100-nan_pct-anom_pct),1); aw=round(min(100-nan_pct,anom_pct),1); nw=round(min(100,nan_pct),1)
        st.markdown(f"""
        <div class="gauge-card">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#7a849e;margin-bottom:12px">✅ DATA QUALITY</div>
          <div style="display:flex;align-items:center;gap:18px;margin-bottom:16px">
            <div style="text-align:center">
              <div style="font-size:42px;font-weight:700;color:{qc};line-height:1">{qs}%</div>
              <div style="font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:#7a849e;margin-top:2px">Score</div>
            </div>
            <div>
              <div style="font-size:16px;font-weight:700;color:#0a0c12">{qg}</div>
              <div style="font-size:11px;color:#7a849e">{qd}</div>
            </div>
          </div>
          <div style="margin-bottom:16px">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px">
              <span style="font-size:11px;font-weight:700;color:#0a0c12">📶 Signal Breakdown</span>
              <span style="font-size:10.5px;color:#7a849e">{total_rows} total rows</span>
            </div>
            <div style="height:20px;border-radius:10px;overflow:hidden;display:flex;background:#e8eaf2">
              <div style="width:{cw}%;background:{TEAL};display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff">{""+str(cw)+"%" if cw>14 else ""}</div>
              <div style="width:{aw}%;background:{AMBER};display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff">{""+str(aw)+"%" if aw>10 else ""}</div>
              <div style="width:{nw}%;background:{ROSE};display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff">{""+str(nw)+"%" if nw>10 else ""}</div>
            </div>
            <div style="display:flex;gap:14px;margin-top:7px">
              <span style="font-size:11px"><b style="color:{TEAL}">{cw}%</b> Clean</span>
              <span style="font-size:11px"><b style="color:{AMBER}">{aw}%</b> Anomalies</span>
              <span style="font-size:11px"><b style="color:{ROSE}">{nw}%</b> NaN</span>
            </div>
          </div>
          {mbar("🚫","NaN Rows",skipped,nan_pct,ROSE,f"{skipped} rows could not be parsed")}
          {mbar("⚠️","Anomalies (>2σ)",len(anom_list),anom_pct,AMBER,"Values deviating strongly from mean")}
          {mbar("❌","Missing Cells",empty_cells,miss_pct,ROSE,f"{empty_cells} empty/null across all columns")}
          {mbar("📐","Spread (CV)",f"{cv}%",min(100,cv),TEAL if cv<20 else AMBER if cv<50 else ROSE,"Low variance — stable" if cv<20 else "Moderate variance" if cv<50 else "High variance")}
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── AI INSIGHTS ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚡ AI Insights</div>', unsafe_allow_html=True)
    insights=[]
    sp=sorted(zip(labels,vals),key=lambda x:-x[1])
    insights.append(("📊","c-b","Dominant Segment",f"<b>{sp[0][0]}</b> leads at {fmt(sp[0][1])}."))
    if len(sp)>=2: insights.append(("📉","c-a","Smallest Segment",f"<b>{sp[-1][0]}</b> is smallest at {fmt(sp[-1][1])}."))
    insights.append(("ℹ️","c-b","Total",f"Sum: <b>{fmt(tot)}</b>{' (~100%)' if 98<=tot<=102 else ''}."))
    tp=round(sp[0][1]/max(1,tot)*100,1)
    insights.append(("⚡","c-a","Concentration",f"Top segment is <b>{tp}%</b>. {'High concentration.' if tp>50 else 'Relatively balanced.'}"))
    if len(vals)>=2:
        dv2=vals[-1]-vals[0]; dp=round(dv2/max(1,abs(vals[0]))*100,1)
        insights.append(("📈" if dv2>0 else "📉","c-t" if dv2>0 else "c-r","Trend Direction",
            f"{'Upward' if dv2>0 else 'Downward'} — <b>{abs(dp)}% {'growth' if dv2>0 else 'decline'}</b>."))
    if len(vals)>=3:
        if abs(r2)>0.5:
            insights.append(("🤖","c-b","ML Regression",f"R²=<b>{r2:.3f}</b>, slope=<b>{round(slope,3)}</b>. {'Strong' if r2>0.85 else 'Moderate'} trend."))
        insights.append(("🔮","c-a","ML Forecast",f"Next 3: <b>[{', '.join(fmt(v) for v in fc3)}]</b>"))
        if anom_list:
            insights.append(("🚨","c-r","Anomaly Detected",f"Outliers (>{z_thresh}σ): <b>{', '.join(l for l,_,_ in anom_list[:4])}</b>"))
        else:
            insights.append(("✅","c-t","No Anomalies","All values within normal range."))
        insights.append(("📐","c-b","Dispersion",f"Std dev: <b>{fmt(round(sd,2))}</b> · CV: <b>{cv}%</b>"))

    ICON_BG={"c-b":"#edf0fd","c-t":"#e8f7f3","c-a":"#fdf4e7","c-r":"#fdf0f1"}
    ig_cols=st.columns(3)
    for i,(ico,cls,title,body) in enumerate(insights):
        with ig_cols[i%3]:
            st.markdown(
                f'<div class="insight-card">'
                f'<div style="width:30px;height:30px;border-radius:8px;background:{ICON_BG.get(cls,"#edf0fd")};'
                f'display:inline-flex;align-items:center;justify-content:center;font-size:14px;margin-right:10px;vertical-align:top;flex-shrink:0">{ico}</div>'
                f'<div style="display:inline-block;vertical-align:top;max-width:calc(100% - 46px)">'
                f'<strong style="font-size:12.5px;color:#0a0c12">{title}</strong><br/>'
                f'<span style="font-size:12px;color:#374061">{body}</span></div></div>',
                unsafe_allow_html=True)

    st.divider()

    # ── AI SUMMARY ────────────────────────────────────────────────────────────
    with st.expander("📋 Overall AI Summary", expanded=True):
        def sb(title_ico, rows_data):
            html=(f'<div class="sum-block"><div class="sum-block-title">{title_ico}</div>'
                  +''.join(f'<div class="sum-line"><span>{l}</span><span class="sum-line-val">{v}</span></div>'
                           for l,v in rows_data)+'</div>')
            return html

        dv3=vals[-1]-vals[0] if len(vals)>=2 else 0
        dp3=round(dv3/max(1,abs(vals[0]))*100,2) if len(vals)>=2 else 0
        c1,c2,c3,c4,c5=st.columns(5)
        c1.markdown(sb("📊 Data Overview",[("Points",n),("Total",fmt(round(tot,2))),("Mean",fmt(round(avg,2))),
                                           ("Median",fmt(round(med,2))),("Std Dev",fmt(round(sd,2))),("Range",fmt(round(vmax-vmin,2)))]),
                    unsafe_allow_html=True)
        c2.markdown(sb("📈 Trend Analysis",[("First",fmt(round(vals[0],2))),("Last",fmt(round(vals[-1],2))),
                                            ("Net Change",("+"+fmt(round(dv3,2)) if dv3>=0 else fmt(round(dv3,2)))),
                                            ("% Change",("+"+str(dp3) if dp3>=0 else str(dp3))+"%"),
                                            ("Slope",str(round(slope,4))),("R²",str(r2))]),
                    unsafe_allow_html=True)
        c3.markdown(sb("🔮 ML Forecast",[("Forecast +1",fmt(fc3[0])),("Forecast +2",fmt(fc3[1])),
                                         ("Forecast +3",fmt(fc3[2])),("Trend","Upward ↑" if slope>0 else "Downward ↓"),
                                         ("R²",str(r2)),("Accuracy",f"{fa}%" if acc else "N/A")]),
                    unsafe_allow_html=True)
        c4.markdown(sb("🎯 Model Accuracy",[("R²",str(acc["r2"]) if acc else "N/A"),
                                            ("MAE",fmt(acc["mae"]) if acc else "N/A"),
                                            ("RMSE",fmt(acc["rmse"]) if acc else "N/A"),
                                            ("MAPE",f"{acc['mape']}%" if acc else "N/A"),
                                            ("Forecast Acc.",f"{fa}%"),("Grade",ag)]),
                    unsafe_allow_html=True)
        c5.markdown(sb("⚠️ Risk & Anomalies",[("Anomalies",len(anom_list)),("Anomaly %",f"{anom_pct}%"),
                                              ("NaN Rows",skipped),("Missing Cells",empty_cells),
                                              ("Quality",f"{qs}%"),("Grade",qg)]),
                    unsafe_allow_html=True)

        verdict=(f"<b>Strong upward trajectory.</b> Consistent positive trend (R²={r2})." if slope>0 and r2>0.7
                 else f"<b>Declining trend.</b> Values decreasing (slope={round(slope,3)})." if slope<0
                 else "<b>Stable or mixed pattern.</b> No strong directional trend. ")
        if anom_list: verdict+=f" <b>{len(anom_list)} anomalies</b> detected — review these data points."
        verdict+=(f" Data quality is <b>{qg}</b> ({qs}%) — suitable for analysis." if qs>=75
                  else f" Data quality is <b>{qg}</b> ({qs}%) — consider cleaning before decisions.")

        tags=[("up","📈 Upward Trend") if slope>0 else ("down","📉 Downward Trend")]
        tags+=[("down",f"🚨 {len(anom_list)} Anomalies") if anom_list else ("up","✅ No Anomalies")]
        tags+=[("neutral",f"🎯 R²={r2}"),("up" if qs>=75 else "down",f"✅ Quality: {qg}")]
        tag_bg={"up":"#e8f7f3","down":"#fdf0f1","neutral":"#edf0fd"}
        tag_cl={"up":"#0d9e7a","down":"#c0313a","neutral":"#1639d6"}
        tag_bc={"up":"rgba(13,158,122,.2)","down":"rgba(192,49,58,.2)","neutral":"rgba(22,57,214,.2)"}
        tag_html="".join(f'<span style="font-size:11px;font-weight:500;padding:4px 12px;border-radius:20px;'
                         f'border:1px solid {tag_bc[t]};margin:2px;display:inline-block;'
                         f'background:{tag_bg[t]};color:{tag_cl[t]}">{lbl}</span>' for t,lbl in tags)
        st.markdown(f'<div class="verdict-box">'
                    f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:#7a849e;margin-bottom:8px">🤖 AI Verdict</div>'
                    f'<div style="font-size:13.5px;color:#1c2033;line-height:1.75">{verdict}</div>'
                    f'<div style="margin-top:12px">{tag_html}</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── DATA SCIENCE WORKBENCH ────────────────────────────────────────────────
    with st.expander("🔬 Data Science Workbench", expanded=False):
        st.markdown('<div style="font-size:20px;font-weight:700;color:#0a0c12">🔬 Data Science Workbench</div>'
                    '<div style="font-size:11px;color:#7a849e;margin-bottom:18px">Advanced statistical analysis, profiling, correlation & distribution</div>',
                    unsafe_allow_html=True)
        ds_tab=st.radio("",["📋 Column Profiling","📐 Deep Statistics","📊 Distributions",
                             "⊹ Scatter Matrix","🧪 Advanced Charts","🔗 Correlation","✅ Data Quality"],
                        horizontal=True, label_visibility="collapsed")
        num_df=df.select_dtypes(include="number")

        if ds_tab=="📋 Column Profiling":
            rows=[]
            for col in df.columns:
                s=df[col]; is_n=pd.api.types.is_numeric_dtype(s)
                ns=pd.to_numeric(s,errors="coerce") if not is_n else s
                rows.append({"Column":col,"Type":"Numeric" if is_n else "Category",
                             "Count":s.count(),"Unique":s.nunique(),
                             "Unique%":f"{round(s.nunique()/max(1,len(s))*100,1)}%",
                             "Missing":s.isna().sum()+(s=="").sum(),
                             "Min":round(float(ns.min()),2) if is_n else "—",
                             "Max":round(float(ns.max()),2) if is_n else "—",
                             "Mean":round(float(ns.mean()),2) if is_n else "—",
                             "Std":round(float(ns.std()),2) if is_n else "—",
                             "Sample":", ".join(str(v) for v in s.dropna().unique()[:3])})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        elif ds_tab=="📐 Deep Statistics":
            if num_df.empty: st.info("No numeric columns found.")
            else:
                dc=st.selectbox("Select column",num_df.columns.tolist())
                s=num_df[dc].dropna()
                if len(s)>=2:
                    q1,q2,q3=float(s.quantile(.25)),float(s.quantile(.5)),float(s.quantile(.75))
                    iqr=q3-q1; skw=float(s.skew()); krt=float(s.kurt())
                    stats_d=[("Count",len(s)),("Mean",round(float(s.mean()),4)),
                             ("Median",round(float(s.median()),4)),("Std Dev",round(float(s.std()),4)),
                             ("Min",round(float(s.min()),4)),("Max",round(float(s.max()),4)),
                             ("Q1",round(q1,4)),("Q3",round(q3,4)),
                             ("IQR",round(iqr,4)),("CV%",round(abs(float(s.std())/float(s.mean())*100) if s.mean()!=0 else 0,1)),
                             ("Skewness",round(skw,4)),("Kurtosis",round(krt,4)),
                             ("Sum",round(float(s.sum()),4)),("Variance",round(float(s.var()),4)),
                             ("Range",round(float(s.max()-s.min()),4)),("Outliers",sum(abs(s-s.mean())>2*s.std()))]
                    cs=st.columns(4)
                    for i,(l,v) in enumerate(stats_d): cs[i%4].metric(l,str(v))

        elif ds_tab=="📊 Distributions":
            if num_df.empty: st.info("No numeric columns.")
            else:
                dc2=st.selectbox("Column",num_df.columns.tolist(),key="dc2")
                dt=st.radio("Type",["Histogram","Box Plot","Area","Density"],horizontal=True)
                s=num_df[dc2].dropna()
                if dt=="Histogram":
                    fig=px.histogram(s,nbins=min(30,max(5,len(s)//5)),color_discrete_sequence=[VIOLET])
                elif dt=="Box Plot":
                    fig=go.Figure(go.Box(y=s,marker_color=COBALT,name=dc2,boxmean="sd"))
                elif dt=="Area":
                    fig=go.Figure(go.Scatter(x=list(range(len(s))),y=s.tolist(),fill="tozeroy",
                                             fillcolor=TEAL+"22",line=dict(color=TEAL,width=2)))
                else:
                    from scipy.stats import gaussian_kde
                    kde=gaussian_kde(s); xr=np.linspace(float(s.min()),float(s.max()),200)
                    fig=go.Figure(go.Scatter(x=xr,y=kde(xr),fill="tozeroy",
                                             fillcolor=VIOLET+"22",line=dict(color=VIOLET,width=2)))
                fig.update_layout(height=300,margin=dict(t=10,b=10,l=10,r=10),showlegend=False)
                st.plotly_chart(fig,use_container_width=True)

        elif ds_tab=="⊹ Scatter Matrix":
            if num_df.shape[1]<2: st.info("Need 2+ numeric columns.")
            else:
                cols_sm=num_df.columns.tolist()[:6]
                fig=px.scatter_matrix(num_df[cols_sm],dimensions=cols_sm,color_discrete_sequence=[COBALT])
                fig.update_traces(marker=dict(size=4,opacity=0.6))
                fig.update_layout(height=500,margin=dict(t=20,b=20,l=20,r=20))
                st.plotly_chart(fig,use_container_width=True)

        elif ds_tab=="🧪 Advanced Charts":
            at=st.radio("Type",["🫧 Bubble","🌊 Waterfall","📏 Pareto","📚 Stacked Bar","📶 Step","🌡️ Heatmap","⊹ Z-Score Scatter"],horizontal=True)
            ac1,ac2,ac3=st.columns(3)
            all_c=df.columns.tolist()
            ax=ac1.selectbox("X/Label",all_c,0,key="ax"); ay=ac2.selectbox("Y/Value",all_c,min(1,len(all_c)-1),key="ay")
            az=ac3.selectbox("Size/Z",all_c,min(2,len(all_c)-1),key="az")
            xv=df[ax].astype(str).tolist()[:40]
            yv=pd.to_numeric(df[ay].astype(str).str.replace(r"[$,%\s]","",regex=True),errors="coerce").fillna(0).tolist()[:40]
            zv=pd.to_numeric(df[az].astype(str).str.replace(r"[$,%\s]","",regex=True),errors="coerce").fillna(0).tolist()[:40]
            N=min(len(xv),len(yv),40)
            if at=="🫧 Bubble":
                mxz=max(abs(v) for v in zv) or 1
                fig=go.Figure(go.Scatter(x=xv[:N],y=yv[:N],mode="markers",
                    marker=dict(size=[max(4,abs(v)/mxz*28) for v in zv[:N]],
                                color=COLORS[:N],opacity=0.7,line=dict(color="#fff",width=1.5)),
                    text=xv[:N],hovertemplate="%{text}: y=%{y}<extra></extra>"))
            elif at=="🌊 Waterfall":
                run2=0; starts2=[]; clrs2=[]
                for v in yv[:N]:
                    starts2.append(run2); run2+=v
                    clrs2.append(TEAL+"bb" if v>=0 else ROSE+"bb")
                fig=go.Figure(go.Bar(x=xv[:N],y=yv[:N],base=starts2,marker_color=clrs2))
            elif at=="📏 Pareto":
                ps=sorted(zip(xv[:N],yv[:N]),key=lambda p:-p[1])
                sl2=[p[0] for p in ps]; sv2=[p[1] for p in ps]
                tp2=sum(sv2) or 1; cum2=0; cp2=[]
                for v in sv2: cum2+=v; cp2.append(round(cum2/tp2*100,1))
                fig=make_subplots(specs=[[{"secondary_y":True}]])
                fig.add_trace(go.Bar(x=sl2,y=sv2,marker_color=COBALT+"bb",name="Value"),secondary_y=False)
                fig.add_trace(go.Scatter(x=sl2,y=cp2,mode="lines+markers",line=dict(color=ROSE,width=2.5),name="Cumulative %"),secondary_y=True)
                fig.update_yaxes(range=[0,105],secondary_y=True)
            elif at=="📚 Stacked Bar":
                fig=go.Figure()
                fig.add_trace(go.Bar(name=ay,x=xv[:N],y=yv[:N],marker_color=COBALT+"bb"))
                fig.add_trace(go.Bar(name=az,x=xv[:N],y=zv[:N],marker_color=TEAL+"bb"))
                fig.update_layout(barmode="stack")
            elif at=="📶 Step":
                fig=go.Figure(go.Scatter(x=xv[:N],y=yv[:N],mode="lines",
                    line=dict(color=VIOLET,width=2.5,shape="hv"),fill="tozeroy",fillcolor=VIOLET+"22"))
            elif at=="🌡️ Heatmap":
                hmc=num_df.columns.tolist()[:6]
                if len(hmc)>=2:
                    cats2=df[ax].astype(str).unique()[:10].tolist()
                    zm=[[float(df[df[ax].astype(str)==cat][c].mean() or 0) if c in df.columns else 0
                          for c in hmc] for cat in cats2]
                    fig=go.Figure(go.Heatmap(z=zm,x=hmc,y=cats2,colorscale="Blues"))
                else: st.info("Need 2+ numeric columns for heatmap."); fig=go.Figure()
            else:
                mu3=np.mean(yv[:N]); sd3=np.std(yv[:N]) or 1
                zs3=[(v-mu3)/sd3 for v in yv[:N]]
                c3a=[COBALT+"cc" if abs(z)<=1 else AMBER+"cc" if abs(z)<=2 else ROSE+"cc" for z in zs3]
                fig=go.Figure(go.Scatter(x=list(range(N)),y=zs3,mode="markers",
                    marker=dict(color=c3a,size=8,line=dict(color="#fff",width=1.5)),
                    text=xv[:N],hovertemplate="%{text}: z=%{y:.2f}<extra></extra>"))
                fig.add_hline(y=2,line_dash="dash",line_color=ROSE,annotation_text="+2σ")
                fig.add_hline(y=-2,line_dash="dash",line_color=ROSE,annotation_text="-2σ")
                fig.add_hline(y=0,line_dash="dot",line_color="#7a849e")
            fig.update_layout(height=350,margin=dict(t=20,b=10,l=10,r=10),showlegend=True,
                              xaxis=dict(showgrid=True,gridcolor="#f2f4f9"),
                              yaxis=dict(showgrid=True,gridcolor="#f2f4f9"))
            st.plotly_chart(fig,use_container_width=True)

        elif ds_tab=="🔗 Correlation":
            if num_df.shape[1]<2: st.info("Need 2+ numeric columns.")
            else:
                corr=num_df.corr()
                fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
                fig.update_layout(height=400,margin=dict(t=20,b=10,l=10,r=10))
                st.plotly_chart(fig,use_container_width=True)
                st.caption("+1 = strong positive, -1 = strong negative, 0 = no correlation")

        else:  # Data Quality Report
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{COBALT},{COBALT}cc);border-radius:12px;'
                f'padding:20px 24px;color:#fff;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between">'
                f'<div><div style="font-size:42px;font-weight:700;line-height:1">{qs}%</div>'
                f'<div style="font-size:11px;opacity:.7;text-transform:uppercase;letter-spacing:.14em;margin-top:4px">Data Quality Score</div></div>'
                f'<div style="font-size:72px;font-weight:700;opacity:.2">{qg[0]}</div></div>',
                unsafe_allow_html=True)
            for ico,lbl,val,pct,color in [
                ("📊","Total Cells",f"{total_cells:,}","",COBALT),
                ("❌","Missing Cells",f"{empty_cells:,}",f"{miss_pct}%",ROSE),
                ("🚫","NaN Rows",f"{skipped:,}",f"{nan_pct}%",ROSE),
                ("⚠️","Anomalies",f"{len(anom_list):,}",f"{anom_pct}%",AMBER),
                ("✅","Completeness",f"{round(completeness,1)}%","",TEAL),
                ("📐","Consistency",f"{round(consistency,1)}%","",TEAL if consistency>=75 else AMBER),
                ("🔵","Uniformity (CV)",f"{cv}%","",TEAL if cv<20 else AMBER if cv<50 else ROSE),
                ("🏆","Verdict",qd,"",qc)]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'padding:10px 0;border-bottom:1px solid #f2f4f9;font-size:12.5px">'
                    f'<span style="color:#374061;font-weight:500">{ico} {lbl}</span>'
                    f'<div style="display:flex;align-items:center;gap:12px">'
                    f'<div style="width:120px;height:8px;background:#e8eaf2;border-radius:4px;overflow:hidden">'
                    f'<div style="height:8px;width:{min(100,float(str(val).replace("%","").replace(",","")) if str(val).replace("%","").replace(",","").replace(".","").isdigit() else 0)}%;background:{color};border-radius:4px"></div></div>'
                    f'<span style="font-family:monospace;font-weight:600;color:{color}">{val}</span>'
                    f'{"<span style=color:#7a849e>"+pct+"</span>" if pct else ""}</div></div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── EXPORT ────────────────────────────────────────────────────────────────
    st.subheader("📥 Export Report")
    report={"source":source_name,"rows":total_rows,"columns":len(file_headers),
            "label_col":label_col,"value_col":value_col,"z_threshold":z_thresh,
            "stats":{"count":n,"total":round(tot,4),"mean":round(avg,4),"median":round(med,4),
                     "std":round(sd,4),"min":round(vmin,4),"max":round(vmax,4),"cv_pct":cv},
            "accuracy":acc,
            "data_quality":{"score":qs,"grade":qg,"verdict":qd},
            "anomalies":[{"label":l,"value":v,"z_score":z} for l,v,z in anom_list],
            "forecast_next_3":fc3}
    st.download_button("⬇️ Download JSON Report",data=json.dumps(report,indent=2),
                       file_name="visiq_report.json",mime="application/json",use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:48px 0;color:#7a849e">
      <div style="font-size:48px;margin-bottom:14px">🔍</div>
      <div style="font-size:17px;font-weight:600;color:#374061;margin-bottom:8px">Ready to analyze your data</div>
      <div style="font-size:13px">Upload a file or type data above, then click <b>Analyze</b></div>
    </div>""", unsafe_allow_html=True)
