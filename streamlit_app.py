import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FinShield AI — AML Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')
CHARTS  = os.path.join(BASE, 'charts')

DRIVE_FILES = {
    'models/xgboost.pkl'                : '1MKV_0mWTSeiISNdyqYx1fmr27S3xWJr3',
    'models/scaler.pkl'                 : '1LmeKRwT2wuM08MliaVSep610mdi-pD8U',
    'models/shap_explainer.pkl'         : '1Z5qGZWH8bci28z0pz9QvFNptF00a7IS9',
    'models/risk_engine_config.pkl'     : '1xfIdKwNU8BbqVs-AMG23wj-5NY9BNhQ4',
    'models/feature_names.pkl'          : '1d3ucIlHUNXDaKej_NvIazNOFXnI8Ifv1',
    'results/ml_comparison_results.csv' : '1dJ01uIANMnXF2tKxNE492qg3PSWzuYgQ',
    'results/shap_feature_importance.csv':'1pZ3AdbDCxIzHhhfOJouyF79qOZAWpjW6',
    'results/xgb_metrics.csv'           : '17A99CF8DOTu10jUKPd6kFLHpaNE8Hrj-',
    'results/lstm_metrics.csv'          : '1klVJgCm9L5CnN1Qh8JuMO0OdeKP1JLyq',
    'results/ae_metrics.csv'            : '1D1at2PYjo-xQ942qN1vCKhASiA7tmkR3',
    'results/cnn_metrics.csv'           : '1IYwKrIU2JCVnDD6uhgUTE5L_wjfumaTD',
    'results/alerts.csv'                : '1IgI13cvi4qWVa-ZFew9ydFBdptdNBA3O',
    'results/final_risk_scores.csv'     : '1lAqxdMvYyE3Kdqd4A8lPD4XcAJlvdh_N',
}

@st.cache_resource(show_spinner="Downloading models from Google Drive...")
def download_and_load():
    try:
        import gdown
        os.makedirs(MODELS, exist_ok=True)
        os.makedirs(RESULTS, exist_ok=True)
        for path, fid in DRIVE_FILES.items():
            full = os.path.join(BASE, path)
            if not os.path.exists(full) or os.path.getsize(full) < 100:
                os.makedirs(os.path.dirname(full), exist_ok=True)
                try:
                    gdown.download(f'https://drive.google.com/uc?id={fid}', full, quiet=True)
                except: pass
    except: pass

    data = {}
    for fname in ['final_risk_scores.csv','alerts.csv','ml_comparison_results.csv','shap_feature_importance.csv','xgb_metrics.csv','lstm_metrics.csv','ae_metrics.csv','cnn_metrics.csv']:
        try: data[fname] = pd.read_csv(os.path.join(RESULTS, fname))
        except: data[fname] = pd.DataFrame()

    model = None
    for mname in ['xgboost.pkl','xgboost_model.pkl']:
        try: model = joblib.load(os.path.join(MODELS, mname)); break
        except: pass

    shap_exp = None
    try:
        with open(os.path.join(MODELS,'shap_explainer.pkl'),'rb') as f:
            shap_exp = pickle.load(f)
    except: pass

    return data, model, shap_exp

download_and_load()
data, model_xgb, shap_exp = download_and_load()

def sf(v, d=4):
    try:
        x = float(v)
        return 0.0 if (x!=x or abs(x)==float('inf')) else round(x,d)
    except: return 0.0

def si(v):
    try: return int(float(v))
    except: return 0

def get_metric(d, *keys):
    for k in keys:
        if k in d and str(d[k]) not in ['nan','inf','']: return sf(d[k])
    return 0.0

risk_df   = data.get('final_risk_scores.csv', pd.DataFrame())
alerts_df = data.get('alerts.csv',            pd.DataFrame())
ml_df     = data.get('ml_comparison_results.csv', pd.DataFrame())
shap_df   = data.get('shap_feature_importance.csv', pd.DataFrame())
xgb_m     = data['xgb_metrics.csv'].iloc[0].to_dict() if not data['xgb_metrics.csv'].empty else {}
lstm_m    = data['lstm_metrics.csv'].iloc[0].to_dict() if not data['lstm_metrics.csv'].empty else {}
ae_m      = data['ae_metrics.csv'].iloc[0].to_dict()   if not data['ae_metrics.csv'].empty else {}
cnn_m     = data['cnn_metrics.csv'].iloc[0].to_dict()  if not data['cnn_metrics.csv'].empty else {}

st.markdown("""
<style>
.stApp { background-color: #030B18; color: #E2EDF8; }
[data-testid="stSidebar"] { background-color: #071428; }
h1,h2,h3 { color: #00D4FF !important; }
div[data-testid="metric-container"] {
    background: #0A1B35; border: 1px solid #1A3050;
    border-radius: 10px; padding: 14px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🛡️ FinShield AI")
st.sidebar.markdown("**AML Intelligence Platform**")
st.sidebar.markdown("---")

if model_xgb:
    st.sidebar.success("✅ XGBoost Loaded")
else:
    st.sidebar.error("❌ XGBoost Not Found")

st.sidebar.markdown(f"**Transactions:** {len(risk_df):,}")
st.sidebar.markdown(f"**Alerts:** {len(alerts_df):,}")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Data Understanding",
    "🔍 EDA",
    "⚙️ Preprocessing",
    "🧪 Feature Engineering",
    "🤖 ML Comparison",
    "🌲 XGBoost",
    "🧠 LSTM",
    "🔄 Autoencoder",
    "📡 CNN-1D",
    "🕸️ Graph Network",
    "💡 SHAP",
    "⚠️ Risk Engine",
    "🔔 Alert Queue",
    "▶️ Live Predict",
])

if page == "🏠 Overview":
    st.title("🛡️ FinShield AI — AML Detection")
    st.markdown("Multi-model ensemble on IBM 5M transaction AML dataset")
    st.markdown("---")
    total = len(risk_df)
    fraud = si(risk_df['Is Laundering'].sum()) if not risk_df.empty else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Cases",        f"{fraud:,}")
    c3.metric("Alerts",             f"{len(alerts_df):,}")
    c4.metric("Models Trained",     "7 ML + 3 DL")
    st.markdown("---")
    st.subheader("Model Performance")
    perf = pd.DataFrame({
        'Model'  : ['XGBoost','LSTM','Autoencoder','CNN-1D'],
        'AUC-ROC': [get_metric(xgb_m,'auc_roc','AUC-ROC'),
                    get_metric(lstm_m,'auc_roc'),
                    get_metric(ae_m,'auc_roc'),
                    get_metric(cnn_m,'auc_roc')],
        'AUC-PR' : [get_metric(xgb_m,'auc_pr','AUC-PR'),
                    get_metric(lstm_m,'auc_pr'),
                    get_metric(ae_m,'auc_pr'),
                    get_metric(cnn_m,'auc_pr')],
        'Recall' : [get_metric(xgb_m,'recall','Recall'),
                    get_metric(lstm_m,'recall'),
                    get_metric(ae_m,'recall'),
                    get_metric(cnn_m,'recall')],
    })
    st.dataframe(perf, use_container_width=True)
    st.markdown("---")
    st.subheader("Charts")
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if f.endswith('.png')][:6]
        cols = st.columns(3)
        for i,img in enumerate(imgs):
            with cols[i%3]:
                st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "📊 Data Understanding":
    st.title("📊 Data Understanding & Cleaning")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Rows",      "5,078,336")
    c2.metric("Fraud Rows",      "5,177",    "0.102%")
    c3.metric("Imbalance",       "979:1")
    c4.metric("Original Columns","11")
    st.markdown("---")
    st.subheader("8 Cleaning Steps")
    steps = [
        ("Column Renaming",         "Account→Sender Account, Account.1→Receiver Account"),
        ("Timestamp Parsing",       "Extracted Hour, Day, DayOfWeek, IsWeekend"),
        ("Missing Value Check",     "Zero missing values confirmed"),
        ("Payment Format OHE",      "7 categories → 7 binary columns"),
        ("Currency Encoding",       "15 currencies → Label Encoded"),
        ("Account Encoding",        "515,080 accounts → Label Encoded"),
        ("Transaction Flags",       "is_cross_bank, is_self_transfer, is_currency_mismatch"),
        ("Outlier Detection",       "4 methods → outlier_score 0-4"),
    ]
    for title, desc in steps:
        with st.expander(f"**{title}**"):
            st.write(desc)

elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fraud Rate",    "0.102%")
    c2.metric("Peak Hour",     "12:00",  "0.174%")
    c3.metric("Riskiest",      "ACH",    "0.746%")
    c4.metric("Riskiest Day",  "Sunday", "0.311%")
    st.markdown("---")
    st.subheader("Payment Format Fraud Rates")
    fmt = pd.DataFrame({
        'Format'    : ['ACH','Bitcoin','Cash','Cheque','Credit Card','Reinvestment','Wire'],
        'Fraud Rate': ['0.746%','0.038%','0.022%','0.017%','0.016%','0.000%','0.000%'],
        'Risk'      : ['🔴 HIGH','🟡','🟢','🟢','🟢','✅ SAFE','✅ SAFE'],
    })
    st.dataframe(fmt, use_container_width=True)
    st.markdown("---")
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if any(k in f.lower() for k in ['eda','class','fraud'])]
        for img in imgs[:4]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Train Rows",  "5.27M",    "after SMOTE")
    c2.metric("Test Rows",   "1.015M",   "100% real")
    c3.metric("SMOTE Ratio", "3.3:1",    "from 979:1")
    c4.metric("Synthetic",   "1,213,415","fraud rows added")
    st.markdown("---")
    steps = [
        ("Log Transform",    "Amount Paid: $0→$1T compressed to 0→27.68"),
        ("Train/Test Split", "80/20 stratified · random_state=42"),
        ("RobustScaler",     "Fitted on train only — avoids data leakage"),
        ("SMOTE",            "Applied on train only · sampling_strategy=0.3"),
    ]
    for t,d in steps:
        with st.expander(f"**{t}**"):
            st.write(d)

elif page == "🧪 Feature Engineering":
    st.title("🧪 Feature Engineering")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Features Created", "79")
    c2.metric("Top Feature",      "86.54%", "is_sending_to_mule")
    c3.metric("Mule Score=4",     "50%",    "fraud rate")
    c4.metric("Groups",           "8")
    st.markdown("---")
    groups = pd.DataFrame({
        'Group'       : ['Velocity','Amount Ratio','Structuring','Network',
                         'Time Pattern','Currency','Mule Detection','Sending to Mule'],
        'Key Feature' : ['sender_txn_count_per_day','sender_amount_zscore',
                         'is_just_below_threshold','is_first_time_transfer',
                         'is_weekend_large_amount','is_same_currency',
                         'mule_score_4','is_sending_to_mule'],
        'Fraud Rate'  : ['43% more','0.253 vs 0.000','0.31%','72.9%',
                         '6×','100%','50%','86.54%'],
    })
    st.dataframe(groups, use_container_width=True)

elif page == "🤖 ML Comparison":
    st.title("🤖 ML Model Comparison")
    if not ml_df.empty:
        sort_by = st.selectbox("Sort by", ['AUC-ROC','AUC-PR','Recall'])
        col_map = {'AUC-ROC':['AUC-ROC','auc_roc'],'AUC-PR':['AUC-PR','auc_pr'],'Recall':['Recall','recall']}
        for c in col_map[sort_by]:
            if c in ml_df.columns:
                ml_df = ml_df.sort_values(c, ascending=False)
                break
        st.dataframe(ml_df, use_container_width=True)
    else:
        st.warning("ml_comparison_results.csv not found")
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if 'ml_model' in f.lower() or 'confusion' in f.lower()]
        for img in imgs[:4]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "🌲 XGBoost":
    st.title("🌲 XGBoost Deep Dive")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC-ROC",      f"{get_metric(xgb_m,'auc_roc','AUC-ROC'):.4f}")
    c2.metric("AUC-PR",       f"{get_metric(xgb_m,'auc_pr','AUC-PR'):.4f}")
    c3.metric("Recall",       f"{get_metric(xgb_m,'recall','Recall'):.4f}")
    c4.metric("Fraud Caught", str(si(xgb_m.get('fraud_caught',xgb_m.get('true_positives',0)))))
    st.markdown("---")
    params = pd.DataFrame({'Parameter':['n_estimators','max_depth','learning_rate','scale_pos_weight','eval_metric'],
                           'Value':['300','6','0.05','3.33','aucpr']})
    st.dataframe(params, use_container_width=True)
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if any(k in f.lower() for k in ['confusion_roc','pr_threshold','feature_importance'])]
        for img in imgs[:4]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "🧠 LSTM":
    st.title("🧠 LSTM Deep Learning")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC-ROC",    f"{get_metric(lstm_m,'auc_roc'):.4f}")
    c2.metric("AUC-PR",     f"{get_metric(lstm_m,'auc_pr'):.4f}")
    c3.metric("Recall",     f"{get_metric(lstm_m,'recall'):.4f}")
    c4.metric("Seq Length", "10 transactions")
    st.info("Reads 10 consecutive transactions per account — detects temporal patterns invisible to XGBoost")
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if 'lstm' in f.lower()]
        for img in imgs[:4]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "🔄 Autoencoder":
    st.title("🔄 Autoencoder — Unsupervised")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC-ROC",  f"{get_metric(ae_m,'auc_roc'):.4f}")
    c2.metric("AUC-PR",   f"{get_metric(ae_m,'auc_pr'):.4f}")
    c3.metric("Recall",   f"{get_metric(ae_m,'recall'):.4f}")
    c4.metric("Training", "Normal Only")
    st.success("Trained on NORMAL transactions only — detects fraud by reconstruction failure. Zero fraud labels used.")
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if 'ae_' in f.lower()]
        for img in imgs[:4]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "📡 CNN-1D":
    st.title("📡 CNN-1D Multi-Scale")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC-ROC",  f"{get_metric(cnn_m,'auc_roc'):.4f}")
    c2.metric("AUC-PR",   f"{get_metric(cnn_m,'auc_pr'):.4f}")
    c3.metric("Recall",   f"{get_metric(cnn_m,'recall'):.4f}")
    c4.metric("Kernels",  "3 · 5 · 7")
    st.info("3 parallel branches detect patterns at different time scales simultaneously")
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if 'cnn' in f.lower()]
        for img in imgs[:4]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "🕸️ Graph Network":
    st.title("🕸️ Graph Network Analysis")
    c1,c2,c3 = st.columns(3)
    c1.metric("Approach",  "Focused Subgraph")
    c2.metric("Library",   "NetworkX DiGraph")
    c3.metric("Features",  "8 created")
    st.markdown("---")
    feats = pd.DataFrame({
        'Feature'      : ['graph_in_degree','graph_out_degree','graph_pagerank',
                          'is_in_cycle','graph_fan_pattern','graph_mule_score'],
        'Means'        : ['Money collector','Money distributor','Network centrality',
                          'Circular flow','0=Normal 3=Mule','Mule likelihood 0-6'],
    })
    st.dataframe(feats, use_container_width=True)
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if 'graph' in f.lower() or 'circular' in f.lower()]
        for img in imgs[:4]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "💡 SHAP":
    st.title("💡 SHAP Explainability")
    if not shap_df.empty:
        fc = next((c for c in shap_df.columns if 'feature' in c.lower()), shap_df.columns[0])
        vc = next((c for c in shap_df.columns if 'shap' in c.lower() or 'mean' in c.lower()), shap_df.columns[1])
        st.subheader("Top 20 Features")
        st.bar_chart(shap_df.head(20).set_index(fc)[vc])
        st.dataframe(shap_df.head(20), use_container_width=True)
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if 'shap' in f.lower()]
        for img in imgs[:6]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "⚠️ Risk Engine":
    st.title("⚠️ Risk Scoring Engine")
    total = len(risk_df)
    fraud = si(risk_df['Is Laundering'].sum()) if not risk_df.empty else 0
    alerts_count = len(alerts_df)
    captured = si(alerts_df['Is Laundering'].sum()) if not alerts_df.empty and 'Is Laundering' in alerts_df.columns else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Scored",   f"{total:,}")
    c2.metric("Alerts",         f"{alerts_count:,}")
    c3.metric("Captured",       f"{captured:,}/{fraud:,}")
    c4.metric("Capture Rate",   f"{round(captured/fraud*100,2) if fraud>0 else 0}%")
    st.markdown("---")
    weights = pd.DataFrame({
        'Model'  : ['XGBoost','LSTM','Autoencoder','CNN-1D','Graph','Composite'],
        'Weight' : ['50%','12%','10%','10%','10%','8%'],
    })
    st.subheader("Ensemble Weights")
    st.dataframe(weights, use_container_width=True)
    if not risk_df.empty and 'risk_level' in risk_df.columns:
        st.subheader("Risk Level Distribution")
        st.bar_chart(risk_df['risk_level'].value_counts())
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if 'risk_engine' in f.lower()]
        for img in imgs:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "🔔 Alert Queue":
    st.title("🔔 Alert Queue")
    if alerts_df.empty:
        st.error("alerts.csv not found")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Alerts", f"{len(alerts_df):,}")
        fraud_in = si(alerts_df['Is Laundering'].sum()) if 'Is Laundering' in alerts_df.columns else 0
        c2.metric("Fraud in Alerts", f"{fraud_in:,}")
        if 'risk_level' in alerts_df.columns:
            c3.metric("Critical", f"{(alerts_df['risk_level']=='CRITICAL').sum():,}")
        st.markdown("---")
        level = st.selectbox("Filter by Level", ["All","CRITICAL","HIGH","MEDIUM","LOW"])
        df2 = alerts_df.copy()
        if level != "All" and 'risk_level' in df2.columns:
            df2 = df2[df2['risk_level']==level]
        st.dataframe(df2.head(50), use_container_width=True)
        st.caption(f"Showing 50 of {len(df2):,} alerts")

elif page == "▶️ Live Predict":
    st.title("▶️ Live Transaction Predict")
    if model_xgb is None:
        st.error("XGBoost model not loaded")
    else:
        XGB_FEATURES = list(model_xgb.feature_names_in_) if hasattr(model_xgb,'feature_names_in_') else []
        DEFAULTS = {
            'Hour':14,'Day':15,'DayOfWeek':2,'IsWeekend':0,
            'Sender Bank ID':11,'Receiver Bank ID':33,
            'is_cross_bank':1,'is_self_transfer':0,
            'Amount_Paid_Log':10.5,'Amount_Received_Log':10.4,
            'is_amount_outlier_iqr':0,'outlier_score':0,
            'txn_count_per_sender_hour':1,'txn_count_per_sender_day':5,
            'fmt_ACH':1,'fmt_Bitcoin':0,'fmt_Cash':0,'fmt_Cheque':0,
            'fmt_Wire':0,'Payment_Currency_Encoded':0,
        }
        payload = {}
        feat_list = XGB_FEATURES or list(DEFAULTS.keys())
        rows = [feat_list[i:i+4] for i in range(0,len(feat_list),4)]
        for row in rows:
            cols = st.columns(len(row))
            for j,feat in enumerate(row):
                with cols[j]:
                    payload[feat] = st.number_input(
                        feat, value=float(DEFAULTS.get(feat,0)), step=0.001, key=feat
                    )
        if st.button("▶ Analyse Transaction", type="primary"):
            try:
                inp = pd.DataFrame([{f:float(payload.get(f,0)) for f in feat_list}])
                if XGB_FEATURES:
                    for f in XGB_FEATURES:
                        if f not in inp.columns: inp[f]=0.0
                    inp = inp[XGB_FEATURES]
                prob  = float(model_xgb.predict_proba(inp)[0][1])
                score = prob*100
                level = 'CRITICAL' if score>=80 else 'HIGH' if score>=60 else 'MEDIUM' if score>=40 else 'LOW'
                col1,col2,col3 = st.columns(3)
                col1.metric("Risk Score",       f"{score:.2f}/100")
                col2.metric("Risk Level",       level)
                col3.metric("Fraud Probability",f"{prob*100:.4f}%")
                if level in ['HIGH','CRITICAL']:
                    st.error(f"🚨 {level} RISK — Escalate for review!")
                elif level == 'MEDIUM':
                    st.warning("🟡 MEDIUM RISK — Monitor closely")
               
