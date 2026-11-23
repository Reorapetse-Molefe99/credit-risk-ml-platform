import json
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app import DECISION_COLORS, GRADE_COLORS, api_url, cfg, get_model, get_portfolio, sidebar_info

from credit_risk.predict import explain, score_frame

st.set_page_config(page_title="Customer Scoring", page_icon="🧾", layout="wide")
sidebar_info()
st.title("Customer Scoring")

portfolio, source = get_portfolio()
model = get_model()
id_col, target = cfg["data"]["id_col"], cfg["data"]["target"]


def score(app_df: pd.DataFrame) -> tuple[dict, str]:
    """Score one application via the API when available, else in-process."""
    url = api_url()
    if url:
        payload = {k: (None if pd.isna(v) else v) for k, v in app_df.iloc[0].to_dict().items()}
        payload = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in payload.items()}
        req = urllib.request.Request(f"{url}/predict?explain_top=0", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=10).read()), "API"
    row = score_frame(model, app_df, cfg).iloc[0]
    return {"probability_of_default": float(row.probability_of_default), "risk_grade": row.risk_grade,
            "decision": row.decision}, "in-process model"


def show_result(app_df: pd.DataFrame, actual=None) -> None:
    res, via = score(app_df)
    pd_ = res["probability_of_default"]
    pct = (portfolio["probability_of_default"] < pd_).mean()
    c1, c2 = st.columns([2, 3])
    with c1:
        steps, lo = [], 0.0
        for g, hi in cfg["risk_grades"].items():
            steps.append({"range": [lo * 100, min(hi, 0.4) * 100], "color": GRADE_COLORS[g]})
            lo = hi
        fig = go.Figure(go.Indicator(mode="gauge+number", value=pd_ * 100, number={"suffix": "%", "valueformat": ".1f"},
                                     title={"text": "Probability of default"},
                                     gauge={"axis": {"range": [0, 40]}, "steps": steps, "bar": {"color": "#333"}}))
        fig.update_layout(height=260, margin=dict(t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        m = st.columns(3)
        m[0].metric("Risk grade", res["risk_grade"])
        m[1].markdown(f"**Decision**<br><span style='font-size:1.6rem;color:{DECISION_COLORS[res['decision']]}'>"
                      f"{res['decision'].upper()}</span>", unsafe_allow_html=True)
        m[2].metric("Riskier than", f"{pct:.0%}", help="Share of the portfolio with a lower PD")
        if actual is not None:
            st.caption(f"Actual outcome: **{'defaulted' if actual == 1 else 'did not default'}**")
        st.caption(f"Scored via {via}")
    with c2:
        st.markdown("**Main drivers (reason codes)**")
        contrib = explain(model, app_df, top=10)
        if contrib is None:
            st.info("Reason codes are available for LightGBM, XGBoost and logistic-regression models.")
        else:
            contrib = contrib.iloc[::-1]
            fig = px.bar(contrib, x="contribution", y="feature", orientation="h", color="direction",
                         color_discrete_map={"increases risk": "#d73027", "decreases risk": "#1a9850"},
                         hover_data={"value": True})
            fig.update_layout(height=360, margin=dict(t=10), xaxis_title="Contribution to log-odds of default",
                              yaxis_title=None, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")


tab_existing, tab_new = st.tabs(["Existing applicant", "New application"])

with tab_existing:
    if id_col not in portfolio:
        st.info("No applicant IDs in this portfolio.")
    else:
        ids = portfolio[id_col].astype(int)
        if "applicant_id" not in st.session_state:
            st.session_state.applicant_id = int(ids.iloc[0])
        c1, c2, c3 = st.columns([2, 1, 1])
        if c2.button("Random applicant"):
            st.session_state.applicant_id = int(ids.sample(1).iloc[0])
        if c3.button("Random defaulter") and target in portfolio:
            st.session_state.applicant_id = int(portfolio.loc[portfolio[target] == 1, id_col].sample(1).iloc[0])
        chosen = c1.number_input(f"Applicant ID ({id_col})", value=st.session_state.applicant_id, step=1)
        match = portfolio[portfolio[id_col] == chosen]
        if match.empty:
            st.warning("ID not in the scored portfolio.")
        else:
            row = match.iloc[[0]]
            info = row[[c for c in ["NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                                    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"] if c in row]]
            st.dataframe(info, hide_index=True, width="stretch")
            show_result(row.drop(columns=[target, "probability_of_default", "risk_grade", "decision"],
                                 errors="ignore"),
                        actual=int(row[target].iloc[0]) if target in row else None)

with tab_new:
    with st.form("new_application"):
        c = st.columns(3)
        contract = c[0].selectbox("Contract type", ["Cash loans", "Revolving loans"])
        income = c[0].number_input("Annual income", min_value=1.0, value=180000.0, step=10000.0)
        credit = c[0].number_input("Loan amount", min_value=1.0, value=500000.0, step=10000.0)
        annuity = c[0].number_input("Annual repayment (annuity)", min_value=0.0, value=27000.0, step=1000.0)
        goods = c[0].number_input("Goods price", min_value=0.0, value=450000.0, step=10000.0)
        age = c[1].slider("Age (years)", 18, 75, 38)
        employed = c[1].checkbox("Currently employed", True)
        years_emp = c[1].slider("Years in current job", 0.0, 40.0, 5.0, 0.5, disabled=not employed)
        income_type = c[1].selectbox("Income type", ["Working", "Commercial associate", "Pensioner", "State servant",
                                                      "Unemployed", "Student", "Businessman"])
        education = c[1].selectbox("Education", ["Secondary / secondary special", "Higher education",
                                                 "Incomplete higher", "Lower secondary", "Academic degree"])
        family = c[2].selectbox("Family status", ["Married", "Single / not married", "Civil marriage",
                                                  "Separated", "Widow"])
        housing = c[2].selectbox("Housing", ["House / apartment", "With parents", "Municipal apartment",
                                             "Rented apartment", "Office apartment", "Co-op apartment"])
        fam_members = c[2].number_input("Family members", 1, 20, 2)
        region = c[2].selectbox("Region rating (1 best – 3 worst)", [1, 2, 3], index=1)
        ext = {}
        for i, default in ((1, 0.5), (2, 0.55), (3, 0.5)):
            known = c[2].checkbox(f"External score {i} available", value=i != 1)
            ext[f"EXT_SOURCE_{i}"] = c[2].slider(f"External score {i}", 0.0, 1.0, default, 0.01) if known else np.nan
        submitted = st.form_submit_button("Score application", type="primary")
    if submitted:
        app_df = pd.DataFrame([{
            "NAME_CONTRACT_TYPE": contract, "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods, "DAYS_BIRTH": -age * 365.25,
            "DAYS_EMPLOYED": -years_emp * 365.25 if employed else 365243, "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education, "NAME_FAMILY_STATUS": family, "NAME_HOUSING_TYPE": housing,
            "CNT_FAM_MEMBERS": float(fam_members), "REGION_RATING_CLIENT": float(region),
            "REGION_RATING_CLIENT_W_CITY": float(region), **ext,
        }])
        st.caption(f"Credit-to-income {credit / income:.2f} · repayment-to-income {annuity / income:.1%}. "
                   "Fields not captured here are treated as missing and imputed.")
        show_result(app_df)
