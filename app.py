"""
Streamlit demo: score a credit card customer for default risk.
Run (after train.py):  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳")
st.title("💳 Credit Card Default Predictor")
st.caption("PyCaret AutoML · UCI Default of Credit Card Clients · Group 8")


@st.cache_resource
def get_model():
    return load_model("outputs/credit_default_model")


model = get_model()
PAY_HELP = "-2/-1 = paid duly, 0 = revolving, 1..8 = months delayed"

tab1, tab2 = st.tabs(["Single customer", "Batch CSV"])

with tab1:
    c1, c2 = st.columns(2)
    limit = c1.number_input("Credit limit (NT$)", 10000, 1000000, 150000, step=10000)
    age = c2.number_input("Age", 21, 79, 35)
    sex = c1.selectbox("Sex", [1, 2], format_func=lambda x: {1: "Male", 2: "Female"}[x])
    edu = c2.selectbox("Education", [1, 2, 3, 4],
                       format_func=lambda x: {1: "Graduate school", 2: "University",
                                              3: "High school", 4: "Other"}[x])
    mar = c1.selectbox("Marital status", [1, 2, 3],
                       format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x])

    st.subheader("Last 6 months (1 = most recent)")
    pay, bill, amt = {}, {}, {}
    for i in range(1, 7):
        a, b, c = st.columns(3)
        pay[i] = a.number_input(f"PAY_{i} status", -2, 8, 0, help=PAY_HELP, key=f"p{i}")
        bill[i] = b.number_input(f"BILL_AMT{i}", -200000, 1000000, 50000, step=1000, key=f"b{i}")
        amt[i] = c.number_input(f"PAY_AMT{i}", 0, 1000000, 2000, step=500, key=f"a{i}")

    if st.button("Predict risk", type="primary"):
        row = {"LIMIT_BAL": limit, "SEX": sex, "EDUCATION": edu, "MARRIAGE": mar, "AGE": age}
        row.update({f"PAY_{i}": pay[i] for i in range(1, 7)})
        row.update({f"BILL_AMT{i}": bill[i] for i in range(1, 7)})
        row.update({f"PAY_AMT{i}": amt[i] for i in range(1, 7)})
        out = predict_model(model, data=pd.DataFrame([row]), raw_score=True)
        p = float(out["prediction_score_1"].iloc[0])
        st.metric("Probability of default next month", f"{p:.1%}")
        if p >= 0.5:
            st.error("HIGH RISK – flag for review / reduce limit")
        elif p >= 0.3:
            st.warning("MEDIUM RISK – monitor")
        else:
            st.success("LOW RISK")

with tab2:
    f = st.file_uploader("Upload CSV with the same 23 feature columns", type="csv")
    if f is None:
        st.info("Try outputs/unseen_sample.csv produced by train.py")
    else:
        df = pd.read_csv(f)
        feats = df.drop(columns=["default"], errors="ignore")
        out = predict_model(model, data=feats, raw_score=True)
        df["default_prob"] = out["prediction_score_1"].round(3)
        df["prediction"] = out["prediction_label"]
        st.write(f"{len(df)} customers scored · {int(df['prediction'].sum())} predicted defaults")
        st.dataframe(df.sort_values("default_prob", ascending=False), use_container_width=True)
        st.download_button("Download results", df.to_csv(index=False), "scored.csv")
