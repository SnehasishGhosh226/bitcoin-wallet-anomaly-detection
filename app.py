from pathlib import Path
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Anomaly Intelligence",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "results" / "metrics"
PLOTS = ROOT / "results" / "plots"

PROJECT_TITLE = "Bitcoin Wallet Behaviour Clustering for Suspicious Pattern Identification"
PROBLEM_CODE = "SIH26146"

# ------------------------------------------------------------
# DESIGN
# ------------------------------------------------------------
st.markdown("""
<style>
.block-container {max-width: 1480px; padding-top: 1.6rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {border-right: 1px solid #263950;}
[data-testid="stMetric"] {
    background: #111d2e; border: 1px solid #263950;
    border-radius: 13px; padding: 16px 18px;
}
[data-testid="stMetricLabel"] {color: #b7c7d9;}
[data-testid="stMetricValue"] {color: #f4f8fc;}
.hero {
    background: linear-gradient(120deg,#102238,#183b56 65%,#1b4b61);
    border: 1px solid #31536d; border-radius: 18px;
    padding: 30px 32px; margin-bottom: 22px;
}
.hero-kicker {color:#8bd3dd; text-transform:uppercase; letter-spacing:.13em;
    font-size:.76rem; font-weight:700; margin-bottom:10px;}
.hero h1 {color:white; font-size:clamp(1.7rem,3vw,2.5rem); margin:0 0 12px;}
.hero p {color:#d0dfeb; max-width:900px; margin-bottom:0;}
.panel {background:#111d2e; border:1px solid #263950; border-radius:13px;
    padding:17px 19px; min-height:115px;}
.panel h4 {color:#edf4fb; margin:0 0 8px; font-size:1rem;}
.panel p {color:#b8c7d8; margin:0; font-size:.92rem;}
.kicker {color:#8bd3dd; text-transform:uppercase; letter-spacing:.1em;
    font-size:.72rem; font-weight:700;}
.notice {border-left:4px solid #d5a85b; background:rgba(120,88,35,.16);
    padding:13px 16px; border-radius:7px; color:#e8d8b8; margin:12px 0;}
hr {border-color:#263950;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(filename):
    path = METRICS / filename
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        return {"_error": str(exc), "_filename": filename}

@st.cache_data(show_spinner=False)
def load_image(filename):
    path = PLOTS / filename
    return path.read_bytes() if path.is_file() else None

def is_error(data):
    return isinstance(data, dict) and "_error" in data

def normalise_evaluation(data):
    if data is None or is_error(data):
        return data
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    return data.rename(columns={
        "F1 Score": "F1_score",
        "F1-score": "F1_score",
        "F1": "F1_score",
        "Flagged Transactions": "Flagged_total",
        "Flagged": "Flagged_total",
        "Flagged Count": "Flagged_total",
    })

def show_missing(filename, data=None):
    if is_error(data):
        st.error(f"Could not read {filename}: {data['_error']}")
    else:
        st.error(f"Required file not found: results/metrics/{filename}")

def model_columns(data):
    if data is None or is_error(data):
        return []
    return [c for c in data.columns if str(c).startswith(("IF_", "LOF_"))]

def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    })

DATA = {
    "evaluation": normalise_evaluation(load_csv("model_evaluation.csv")),
    "enriched_evaluation": normalise_evaluation(load_csv("enriched_model_evaluation.csv")),
    "agreement": load_csv("model_agreement.csv"),
    "comparison": load_csv("original_vs_enriched_comparison.csv"),
    "false_positives": load_csv("false_positive_summary.csv"),
    "original_flags": load_csv("transaction_anomaly_flags.csv"),
    "enriched_flags": load_csv("enriched_transaction_anomaly_flags.csv"),
    "pca": load_csv("pca_variance_comparison.csv"),
}

# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## ₿ BITCOIN ANALYTICS")
    st.caption("UNSUPERVISED LEARNING PROJECT")
    st.divider()
    page = st.radio("NAVIGATION", [
        "Overview", "Model Performance", "Feature Comparison",
        "Visual Analytics", "Transaction Explorer", "Methodology"
    ])
    st.divider()
    st.markdown("**PROJECT INFORMATION**")
    st.caption(f"Problem code: {PROBLEM_CODE}")
    st.caption("Dataset: Elliptic Bitcoin Dataset")
    st.caption("Methods: Isolation Forest · LOF")
    st.caption("Visualisation: PCA")
    st.caption("Feature sets: 165 and 167")
    st.divider()
    st.caption("Research prototype · Precomputed results")

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
if page == "Overview":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-kicker">Unsupervised learning · {PROBLEM_CODE}</div>
      <h1>Bitcoin Anomaly Intelligence</h1>
      <p>{PROJECT_TITLE}</p>
      <p style="margin-top:12px">An experimental comparison of Isolation Forest and
      Local Outlier Factor, with PCA visualisation and transaction-network
      feature enrichment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="kicker">EXPERIMENT AT A GLANCE</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Transactions sampled", "10,000")
    b.metric("Original features", "165")
    c.metric("Enriched features", "167")
    d.metric("Model configurations", "6")
    st.divider()

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.subheader("Research objective")
        st.write(
            "Investigate whether unsupervised anomaly-detection methods identify "
            "unusual transaction patterns, compare the observations they flag, "
            "and assess how transaction-network connectivity features change results."
        )
        st.subheader("Methods used")
        st.markdown("""
        - **Isolation Forest:** identifies observations relatively easy to isolate.
        - **Local Outlier Factor:** compares local density with neighbouring observations.
        - **PCA:** projects high-dimensional features into two dimensions for inspection.
        """)
    with right:
        st.subheader("Experimental feature sets")
        st.dataframe(pd.DataFrame({
            "Feature set": ["Original", "Graph-enriched"],
            "Feature count": [165, 167],
            "Additional variables": ["None", "Incoming and outgoing connections"],
        }), hide_index=True, use_container_width=True)
        st.subheader("Dataset")
        st.write("203,769 transactions, 165 numerical features, class labels and 234,355 transaction relationships.")

    st.markdown("""
    <div class="notice"><strong>Responsible interpretation:</strong> An anomaly is
    an unusual observation, not proof of illicit activity. This is transaction-level
    analysis; the dataset does not directly establish wallet ownership or identity.</div>
    """, unsafe_allow_html=True)

    st.subheader("Dashboard sections")
    cols = st.columns(3)
    descriptions = [
        ("01 · Compare", "Inspect precision, recall and F1-score across model configurations."),
        ("02 · Investigate", "Examine how transaction connections affect anomaly flags."),
        ("03 · Interpret", "Review PCA projections, agreement measures and limitations."),
    ]
    for col, (title, body) in zip(cols, descriptions):
        with col:
            st.markdown(f'<div class="panel"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Post-hoc evaluation against known licit and illicit labels. Unknown labels are excluded.")
    original = DATA["evaluation"]
    enriched = DATA["enriched_evaluation"]

    if original is None or is_error(original):
        show_missing("model_evaluation.csv", original)
    if enriched is None or is_error(enriched):
        show_missing("enriched_model_evaluation.csv", enriched)

    if original is not None and enriched is not None and not is_error(original) and not is_error(enriched):
        choice = st.radio("Feature set", ["Original features", "Enriched features"], horizontal=True)
        selected = original if choice == "Original features" else enriched
        required = ["Model", "Precision", "Recall", "F1_score"]
        missing = [x for x in required if x not in selected.columns]

        if missing:
            st.error("Missing evaluation columns: " + ", ".join(missing))
            st.caption("Available: " + ", ".join(selected.columns))
        else:
            st.subheader(f"{choice} · Evaluation results")
            fmt = {c: "{:.4f}" for c in ["Precision", "Recall", "F1_score"]}
            st.dataframe(selected.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)

            best = selected.loc[selected["F1_score"].idxmax()]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best F1-score", f"{best['F1_score']:.4f}")
            m2.metric("Precision", f"{best['Precision']:.4f}")
            m3.metric("Recall", f"{best['Recall']:.4f}")
            if "Flagged_total" in selected.columns:
                m4.metric("Flagged transactions", f"{int(best['Flagged_total']):,}")
            else:
                m4.metric("Best configuration", str(best["Model"]))

            st.info(f"Highest F1-score in this feature set: **{best['Model']}** ({best['F1_score']:.4f}). This is the best tested configuration, not evidence of production-ready detection.")
            st.subheader("Metric comparison")
            st.bar_chart(selected.set_index("Model")[["Precision", "Recall", "F1_score"]], height=340)
            if "Flagged_total" in selected.columns:
                st.subheader("Flagged observations by configuration")
                st.bar_chart(selected.set_index("Model")[["Flagged_total"]], height=240)

    st.markdown("""
    <div class="notice">Anomaly flags are treated as predicted illicit labels only
    for post-hoc evaluation. Unknown transactions are excluded from the metrics.</div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# FEATURE COMPARISON
# ------------------------------------------------------------
elif page == "Feature Comparison":
    st.title("Original vs Enriched Features")
    st.caption("Investigate whether transaction-network connectivity features change model decisions.")

    st.subheader("Additional transaction-network features")
    x, y = st.columns(2)
    with x:
        st.markdown('<div class="panel"><h4>Incoming connections</h4><p>Number of incoming transaction relationships recorded for a transaction.</p></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="panel"><h4>Outgoing connections</h4><p>Number of outgoing transaction relationships recorded for a transaction.</p></div>', unsafe_allow_html=True)
    st.caption("These are transaction-network measures, not wallet ownership indicators or a complete graph-learning model.")
    st.divider()

    comparison = DATA["comparison"]
    if comparison is None or is_error(comparison):
        show_missing("original_vs_enriched_comparison.csv", comparison)
    else:
        required = ["Model", "Jaccard_similarity"]
        missing = [c for c in required if c not in comparison.columns]
        if missing:
            st.error("Comparison file is missing: " + ", ".join(missing))
        else:
            st.subheader("Anomaly decision consistency")
            st.dataframe(comparison.style.format({"Jaccard_similarity": "{:.4f}"}), use_container_width=True, hide_index=True)
            st.bar_chart(comparison.set_index("Model")[["Jaccard_similarity"]], height=300)
            st.caption("Jaccard similarity measures overlap between original and enriched flagged transactions. Values closer to 1 indicate greater overlap.")
            st.metric("Mean Jaccard similarity", f"{comparison['Jaccard_similarity'].mean():.4f}")

    st.divider()
    st.subheader("PCA explained variance")
    pca = DATA["pca"]
    if pca is None or is_error(pca):
        show_missing("pca_variance_comparison.csv", pca)
    else:
        st.dataframe(pca, use_container_width=True, hide_index=True)
        st.write("The first two principal components explain approximately 17.25% of variance for the original features and 17.03% for the enriched features. Enrichment changed some anomaly decisions but did not improve this two-component variance summary.")

# ------------------------------------------------------------
# VISUAL ANALYTICS
# ------------------------------------------------------------
elif page == "Visual Analytics":
    st.title("Visual Analytics")
    st.caption("Explore the saved anomaly-detection and PCA figures.")
    plots = {
        "Original anomaly detection comparison": "anomaly_detection_comparison.png",
        "Original PCA — transaction labels": "pca_transaction_labels.png",
        "Enriched PCA — transaction features": "pca_enriched_features.png",
    }
    selected_plot = st.selectbox("Choose a visualisation", list(plots))
    image = load_image(plots[selected_plot])
    if image is None:
        st.error(f"Plot not found: results/plots/{plots[selected_plot]}")
    else:
        st.subheader(selected_plot)
        st.image(image, use_container_width=True)

    st.divider()
    st.subheader("PCA interpretation")
    st.write("PCA compresses the feature space into two principal components. These components capture only part of the total variance. Visible overlap between licit, illicit and unknown transactions does not establish that the original high-dimensional data is indistinguishable.")
    a, b, c = st.columns(3)
    a.metric("Original PCA variance", "17.25%")
    b.metric("Enriched PCA variance", "17.03%")
    c.metric("Difference", "−0.22 pp")

# ------------------------------------------------------------
# TRANSACTION EXPLORER
# ------------------------------------------------------------
elif page == "Transaction Explorer":
    st.title("Transaction Explorer")
    st.caption("Filter and inspect saved transaction-level anomaly flags. These are precomputed results.")
    choice = st.radio("Result set", ["Enriched features", "Original features"], horizontal=True)
    if choice == "Enriched features":
        data = DATA["enriched_flags"]
        filename = "enriched_transaction_anomaly_flags.csv"
    else:
        data = DATA["original_flags"]
        filename = "transaction_anomaly_flags.csv"

    if data is None or is_error(data):
        show_missing(filename, data)
    else:
        data = data.copy()
        flags = model_columns(data)
        if not flags:
            st.error("No model flag columns found. Expected names beginning with IF_ or LOF_.")
            st.caption("Available columns: " + ", ".join(data.columns))
        else:
            c1, c2, c3 = st.columns([1.1, 1, 1.2])
            with c1:
                model = st.selectbox("Model configuration", flags)
            with c2:
                status = st.selectbox("Transaction status", ["All", "Flagged anomalies", "Not flagged"])
            with c3:
                search = st.text_input("Search transaction ID", placeholder="Enter transaction ID")

            flag_values = to_bool(data[model])
            data["_flag_value"] = flag_values
            filtered = data.copy()
            if status == "Flagged anomalies":
                filtered = filtered[filtered["_flag_value"] == True]
            elif status == "Not flagged":
                filtered = filtered[filtered["_flag_value"] == False]

            if search.strip():
                id_col = next((c for c in filtered.columns if c.lower() in ["txid", "tx_id", "transaction_id"]), None)
                if id_col is None:
                    st.warning("No transaction ID column was found in this result file.")
                else:
                    filtered = filtered[filtered[id_col].astype(str).str.contains(search.strip(), case=False, na=False)]

            a, b, c = st.columns(3)
            a.metric("Transactions in file", f"{len(data):,}")
            b.metric("Flagged by selected model", f"{int(flag_values.fillna(False).sum()):,}")
            c.metric("Rows after filters", f"{len(filtered):,}")

            display = filtered.drop(columns=["_flag_value"], errors="ignore")
            st.dataframe(display, use_container_width=True, hide_index=True, height=430)
            st.download_button(
                "Download filtered transactions (CSV)",
                data=display.to_csv(index=False).encode("utf-8"),
                file_name=f"{choice.lower().replace(' ', '_')}_filtered_transactions.csv",
                mime="text/csv",
            )
            st.caption("This explorer displays saved model flags. It does not run new predictions or independently verify illicit activity.")

# ------------------------------------------------------------
# METHODOLOGY
# ------------------------------------------------------------
elif page == "Methodology":
    st.title("Methodology and Responsible Interpretation")
    st.caption("Dataset, preprocessing, experiments and limits of inference.")

    with st.expander("Dataset and preparation", expanded=True):
        st.markdown("""
        - Elliptic Bitcoin Dataset: 203,769 transactions, 165 numerical features and 234,355 transaction relationships.
        - Transaction features were joined to labels using transaction IDs.
        - Missing values and duplicate transaction IDs were checked.
        - A reproducible sample of 10,000 transactions was used (`random_state=42`).
        - Original numerical features already appeared standardised, so no additional scaling was applied to them.
        """)

    with st.expander("Graph-connectivity feature enrichment"):
        st.markdown("""
        - Derived incoming and outgoing connection counts from the edge list.
        - Applied `log1p` transformation and standardisation to these two features.
        - Combined them with the original 165 features to create a 167-feature dataset.
        - Total connection count was excluded because it is redundant with incoming plus outgoing counts.
        """)
        st.caption("These variables describe transaction relationships; they do not establish wallet ownership.")

    with st.expander("Anomaly-detection configurations"):
        st.markdown("""
        **Isolation Forest**
        - 200 estimators; contamination settings of 2% and 5%.

        **Local Outlier Factor**
        - 20 and 50 neighbours; contamination settings of 2% and 5%.
        """)

    with st.expander("Evaluation"):
        st.markdown("""
        - Precision, recall and F1-score using known labels only.
        - Jaccard similarity between model outputs.
        - Original versus enriched feature comparison.
        - PCA visualisation and false-positive implications.
        - Labels were not used as model inputs or to fit the unsupervised methods.
        """)

    st.subheader("Limitations")
    for heading, detail in [
        ("Incomplete labels", "Unknown transactions are excluded from label-based evaluation."),
        ("Sampling", "Experiments use a reproducible sample of 10,000 transactions."),
        ("Parameter sensitivity", "Different algorithms and parameter settings flag different observations."),
        ("PCA representation", "The first two components explain approximately 17% of variance."),
        ("Transaction-level scope", "The dataset does not directly identify wallet owners."),
        ("False positives", "An anomaly flag is not proof of illicit activity."),
        ("Preprocessing reuse", "A future prediction pipeline must reuse fitted preprocessing parameters for new data."),
        ("Dashboard scope", "The application displays precomputed results; it is not a live monitoring system."),
    ]:
        st.markdown(f"**{heading}:** {detail}")

    st.subheader("Ethical considerations")
    st.write("False-positive anomaly flags can cause unnecessary scrutiny, wasted investigative resources and unfair suspicion. Model outputs should be treated as indicators for further review rather than definitive judgments. No transaction or wallet should be labelled criminal solely because an unsupervised model identifies it as unusual.")

    st.markdown("""
    <div class="notice"><strong>Research prototype:</strong> This dashboard presents
    precomputed notebook results. It is not a live transaction-monitoring or
    automated enforcement system.</div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.caption(f"{PROJECT_TITLE} · {PROBLEM_CODE} · Unsupervised Learning Project")
