import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="FinShield AI", page_icon="🛡️", layout="wide")

BASE    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')
CHARTS  = os.path.join(BASE, 'charts')

@st.cache_data
def load_csv(fname):
    try:
        return pd.read_csv(os.path.join(RESULTS, fname))
    except:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    for name in ['xgboost.pkl','xgboost_model.pkl']:
        try:
            return joblib.load(os.path.join(MODELS, name))
        except:
            pass
    return None

risk_df   = load_csv('final_risk_scores.csv')
alerts_df = load_csv('alerts.csv')
ml_df     = load_csv('ml_comparison_results.csv')
shap_df   = load_csv('shap_feature_importance.csv')
xgb_m     = load_csv('xgb_metrics.csv')
model_xgb = load_model()

st.sidebar.title("🛡️ FinShield AI")
st.sidebar.caption("AML Intelligence Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Data Understanding",
    "EDA",
    "Preprocessing",
    "Feature Engineering",
    "ML Comparison",
    "XGBoost",
    "LSTM",
    "Autoencoder",
    "CNN-1D",
    "Graph Network",
    "SHAP",
    "Risk Engine",
    "Alert Queue",
    "Live Predict",
])

if page == "Overview":
    st.title("🛡️ FinShield AI — AML Detection")
    st.markdown("Multi-model ensemble on IBM 5M transaction AML dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions", f"{len(risk_df):,}" if not risk_df.empty else "5,078,336")
    c2.metric("Fraud Cases",  "5,177")
    c3.metric("Alerts",       f"{len(alerts_df):,}" if not alerts_df.empty else "0")
    c4.metric("Models",       "7 ML + 3 DL")
    st.markdown("---")
    if os.path.exists(CHARTS):
        imgs = [f for f in os.listdir(CHARTS) if f.endswith('.png')]
        cols = st.columns(3)
        for i,img in enumerate(imgs[:6]):
            with cols[i%3]:
                st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "Data Understanding":
    st.title("📊 Data Understanding")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Rows",   "5,078,336")
    c2.metric("Fraud Rows",   "5,177")
    c3.metric("Imbalance",    "979:1")
    c4.metric("Columns",      "11 → 32+")
    st.markdown("---")
    st.subheader("Original 11 Columns")
    df = pd.DataFrame({
        'Column' :['Timestamp','From Bank','Account','To Bank','Account.1',
                   'Amount Received','Receiving Currency','Amount Paid',
                   'Payment Currency','Payment Format','Is Laundering'],
        'Action' :['→ Hour,Day,DayOfWeek,IsWeekend','→ Sender Bank ID',
                   '→ Sender Account (encoded)','→ Receiver Bank ID',
                   '→ Receiver Account (encoded)','→ log transform',
                   '→ Label Encoded','→ log transform + outlier flags',
                   '→ Label Encoded','→ 7 OHE columns','TARGET'],
    })
    st.dataframe(df, use_container_width=True)

elif page == "EDA":
    st.title("🔍 Exploratory Analysis")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fraud Rate",  "0.102%")
    c2.metric("Peak Hour",   "12:00")
    c3.metric("Riskiest",    "ACH 0.746%")
    c4.metric("Safe Formats","Wire + Reinvestment 0%")
    st.markdown("---")
    st.subheader("Format Fraud Rates")
    st.dataframe(pd.DataFrame({
        'Format'    :['ACH','Bitcoin','Cash','Cheque','Credit Card','Reinvestment','Wire'],
        'Fraud Rate':['0.746%','0.038%','0.022%','0.017%','0.016%','0.000%','0.000%'],
    }), use_container_width=True)
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'class_balance' in f][:2]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "Preprocessing":
    st.title("⚙️ Preprocessing")
    st.dataframe(pd.DataFrame({
        'Step'  :['Log Transform','Train/Test Split','RobustScaler','SMOTE'],
        'Detail':['Amount $0→$1T compressed to 0→27.68',
                  '80/20 stratified · random_state=42',
                  'Fitted on train only — no leakage',
                  'Train only · sampling_strategy=0.3'],
    }), use_container_width=True)

elif page == "Feature Engineering":
    st.title("🧪 Feature Engineering")
    c1,c2,c3 = st.columns(3)
    c1.metric("Features",    "79")
    c2.metric("Top Feature", "86.54%")
    c3.metric("Groups",      "8")
    st.dataframe(pd.DataFrame({
        'Group'       :['Velocity','Amount Ratio','Network','Mule','Sending to Mule'],
        'Key Feature' :['sender_txn_count_per_day','sender_amount_zscore',
                        'is_first_time_transfer','mule_score_4','is_sending_to_mule'],
        'Fraud Rate'  :['43% higher','0.253 vs 0.000','72.9%','50%','86.54%'],
    }), use_container_width=True)

elif page == "ML Comparison":
    st.title("🤖 ML Comparison")
    if not ml_df.empty:
        st.dataframe(ml_df, use_container_width=True)
    else:
        st.warning("ml_comparison_results.csv not found in results/")
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'ml_model' in f]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "XGBoost":
    st.title("🌲 XGBoost")
    if not xgb_m.empty:
        row = xgb_m.iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{float(row.get('auc_roc',row.get('AUC-ROC',0))):.4f}")
        c2.metric("AUC-PR",  f"{float(row.get('auc_pr',row.get('AUC-PR',0))):.4f}")
        c3.metric("Recall",  f"{float(row.get('recall',row.get('Recall',0))):.4f}")
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if any(k in f for k in ['confusion_roc','pr_threshold'])]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "LSTM":
    st.title("🧠 LSTM")
    lstm_m = load_csv('lstm_metrics.csv')
    if not lstm_m.empty:
        row = lstm_m.iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{float(row.get('auc_roc',0)):.4f}")
        c2.metric("AUC-PR",  f"{float(row.get('auc_pr',0)):.4f}")
        c3.metric("Recall",  f"{float(row.get('recall',0)):.4f}")
    st.info("Reads 10 consecutive transactions per account in time order")
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'lstm' in f.lower()]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "Autoencoder":
    st.title("🔄 Autoencoder")
    ae_m = load_csv('ae_metrics.csv')
    if not ae_m.empty:
        row = ae_m.iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{float(row.get('auc_roc',0)):.4f}")
        c2.metric("AUC-PR",  f"{float(row.get('auc_pr',0)):.4f}")
        c3.metric("Recall",  f"{float(row.get('recall',0)):.4f}")
    st.success("Trained on NORMAL only — zero fraud labels used")
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'ae_' in f.lower()]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "CNN-1D":
    st.title("📡 CNN-1D")
    cnn_m = load_csv('cnn_metrics.csv')
    if not cnn_m.empty:
        row = cnn_m.iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{float(row.get('auc_roc',0)):.4f}")
        c2.metric("AUC-PR",  f"{float(row.get('auc_pr',0)):.4f}")
        c3.metric("Recall",  f"{float(row.get('recall',0)):.4f}")
    st.info("3 parallel branches detect patterns at kernels 3, 5, 7")
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'cnn' in f.lower()]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "Graph Network":
    st.title("🕸️ Graph Network")
    st.dataframe(pd.DataFrame({
        'Feature'  :['graph_in_degree','graph_out_degree','graph_pagerank','is_in_cycle','graph_mule_score'],
        'Meaning'  :['Money collector','Money distributor','Network hub','Circular flow','Mule likelihood'],
    }), use_container_width=True)
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'graph' in f.lower() or 'circular' in f.lower()]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "SHAP":
    st.title("💡 SHAP Explainability")
    if not shap_df.empty:
        fc = shap_df.columns[0]
        vc = shap_df.columns[1]
        st.bar_chart(shap_df.head(15).set_index(fc)[vc])
        st.dataframe(shap_df.head(20), use_container_width=True)
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'shap' in f.lower()]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "Risk Engine":
    st.title("⚠️ Risk Engine")
    if not risk_df.empty:
        fraud = int(risk_df['Is Laundering'].sum())
        captured = int(alerts_df['Is Laundering'].sum()) if not alerts_df.empty and 'Is Laundering' in alerts_df.columns else 0
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Scored", f"{len(risk_df):,}")
        c2.metric("Captured",     f"{captured}/{fraud}")
        c3.metric("Capture Rate", f"{round(captured/fraud*100,2) if fraud>0 else 0}%")
        if 'risk_level' in risk_df.columns:
            st.bar_chart(risk_df['risk_level'].value_counts())
    if os.path.exists(CHARTS):
        for img in [f for f in os.listdir(CHARTS) if 'risk_engine' in f.lower()]:
            st.image(os.path.join(CHARTS,img), use_column_width=True)

elif page == "Alert Queue":
    st.title("🔔 Alert Queue")
    if alerts_df.empty:
        st.error("alerts.csv not found")
    else:
        level = st.selectbox("Filter", ["All","CRITICAL","HIGH","MEDIUM","LOW"])
        df2 = alerts_df.copy()
        if level != "All" and 'risk_level' in df2.columns:
            df2 = df2[df2['risk_level']==level]
        st.dataframe(df2.head(100), use_container_width=True)

elif page == "Live Predict":
    st.title("▶️ Live Predict")
    if model_xgb is None:
        st.error("XGBoost not loaded")
        st.info("Add models/xgboost.pkl to your repo")
    else:
        XGB_FEATURES = list(model_xgb.feature_names_in_) if hasattr(model_xgb,'feature_names_in_') else []
        st.subheader("Enter transaction features")
        payload = {}
        DEFAULTS = {'Hour':14,'Day':15,'DayOfWeek':2,'IsWeekend':0,
                    'Sender Bank ID':11,'Receiver Bank ID':33,
                    'is_cross_bank':1,'Amount_Paid_Log':10.5,
                    'fmt_ACH':1,'outlier_score':0}
        feat_list = XGB_FEATURES or list(DEFAULTS.keys())
        rows = [feat_list[i:i+4] for i in range(0,len(feat_list),4)]
        for row in rows:
            cols = st.columns(len(row))
            for j,feat in enumerate(row):
                with cols[j]:
                    payload[feat] = st.number_input(feat, value=float(DEFAULTS.get(feat,0)), step=0.001, key=feat)
        if st.button("Analyse", type="primary"):
            try:
                inp = pd.DataFrame([{f:float(payload.get(f,0)) for f in feat_list}])
                if XGB_FEATURES:
                    for f in XGB_FEATURES:
                        if f not in inp.columns:
                            inp[f] = 0.0
                    inp = inp[XGB_FEATURES]
                prob  = float(model_xgb.predict_proba(inp)[0][1])
                score = prob * 100
                level = 'CRITICAL' if score>=80 else 'HIGH' if score>=60 else 'MEDIUM' if score>=40 else 'LOW'
                c1,c2,c3 = st.columns(3)
                c1.metric("Risk Score",  f"{score:.2f}/100")
                c2.metric("Risk Level",  level)
                c3.metric("Fraud Prob",  f"{prob*100:.4f}%")
                if level in ['HIGH','CRITICAL']:
                    st.error(f"🚨 {level} — Escalate!")
                elif level == 'MEDIUM':
                    st.warning("🟡 Monitor")
                else:
                    st.success("🟢 Normal")
            except Exception as e:
                st.error(f"Error: {e}")
