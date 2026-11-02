
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bitcoin Anomaly Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "results" / "metrics"
PLOTS = ROOT / "results" / "plots"


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stMetric"] {
        background: #172235;
        border: 1px solid #293a52;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #b8c7da;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #14243b, #203c5c);
        border: 1px solid #35506e;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #d1deed;
        font-size: 1rem;
    }

    .section-note {
        color: #8b98a9;
        font-size: 0.9rem;
    }

    .warning-note {
        padding: 15px;
        border-radius: 10px;
        background: #382d1c;
        border: 1px solid #796037;
        color: #f4dfb7;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# 3. DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_csv(filename):
    path = METRICS / filename

    if path.exists():
        return pd.read_csv(path)

    return None


# Load all project results
evaluation = load_csv("model_evaluation.csv")
enriched_evaluation = load_csv("enriched_model_evaluation.csv")
agreement = load_csv("model_agreement.csv")
comparison = load_csv("original_vs_enriched_comparison.csv")
false_positives = load_csv("false_positive_summary.csv")
original_flags = load_csv("transaction_anomaly_flags.csv")
enriched_flags = load_csv("enriched_transaction_anomaly_flags.csv")
pca_comparison = load_csv("pca_variance_comparison.csv")


# Temporary diagnostic output
if evaluation is not None:
    evaluation = evaluation.rename(columns={
        "F1 Score": "F1_score",
        "Flagged Transactions": "Flagged_total"
    })

if enriched_evaluation is not None:
    enriched_evaluation = enriched_evaluation.rename(columns={
        "F1 Score": "F1_score",
        "Flagged Transactions": "Flagged_total"
    })

# Helper function for missing result files
def show_missing_file(filename):
    st.error(
        f"Required result file not found: {filename}"
    )


def metric_value(dataframe, column):
    if dataframe is not None and column in dataframe.columns:
        return dataframe[column].sum()

    return None


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("BITCOIN ANALYTICS")
st.sidebar.caption("Unsupervised Learning Project")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Model Performance",
        "Feature Comparison",
        "Visual Analytics",
        "Transaction Explorer",
        "Methodology"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("**PROJECT INFORMATION**")
st.sidebar.caption("Dataset: Elliptic Bitcoin Dataset")
st.sidebar.caption("Problem Code: SIH26146")
st.sidebar.caption("Methods: Isolation Forest and LOF")
st.sidebar.caption("Visualisation: PCA")


# --------------------------------------------------
# 5. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>Bitcoin Anomaly Intelligence</h1>
        <p>
        Bitcoin Wallet Behaviour Clustering for Suspicious
        Pattern Identification
        </p>
        <p>
        An unsupervised learning study comparing Isolation Forest,
        Local Outlier Factor and PCA-based visual analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions analysed", "10,000")
    c2.metric("Original features", "165")
    c3.metric("Enriched features", "167")
    c4.metric("Model configurations", "6")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Research Objective")

        st.write("""
        Identify unusual transaction patterns using unsupervised
        anomaly-detection methods, compare their outputs and
        examine their consistency against available dataset labels.
        """)

        st.subheader("Algorithms")

        st.markdown("""
        - **Isolation Forest:** Identifies observations that are
          relatively easy to isolate.
        - **Local Outlier Factor:** Identifies observations with
          unusually low local density.
        - **PCA:** Projects high-dimensional features into
          two dimensions for visual inspection.
        """)

    with right:
        st.subheader("Experimental Design")

        st.markdown("""
        | Feature set | Number of features |
        |---|---:|
        | Original | 165 |
        | Enriched | 167 |
        """)

        st.markdown("""
        **Additional network features**

        - Incoming transaction connections
        - Outgoing transaction connections
        """)

    st.markdown("""
    <div class="warning-note">
    <b>Responsible interpretation:</b> An anomaly is an unusual
    observation, not proof of illicit activity. The dataset supports
    transaction-level analysis; it does not directly establish
    individual wallet ownership or identity.
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# 6. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance")
    st.caption("Evaluation against known licit and illicit labels.")

    if evaluation is None or enriched_evaluation is None:
        show_missing_file("model_evaluation.csv or enriched_model_evaluation.csv")

    else:
        feature_choice = st.radio(
            "Select feature set",
            ["Original features", "Enriched features"],
            horizontal=True
        )

        selected = (
            evaluation
            if feature_choice == "Original features"
            else enriched_evaluation
        )

        st.subheader(f"{feature_choice} — Evaluation Metrics")

        st.dataframe(
            selected.style.format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1_score": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Standardise metric column names
        selected = selected.rename(columns={
            "F1": "F1_score",
            "F1 Score": "F1_score",
            "F1-score": "F1_score"
        })

        required_columns = ["Model", "Precision", "Recall", "F1_score"]

        if all(col in selected.columns for col in required_columns):
            st.subheader("Model Metric Comparison")

            chart_data = selected.set_index("Model")[
                ["Precision", "Recall", "F1_score"]
            ]
            st.bar_chart(chart_data)

            best = selected.loc[selected["F1_score"].idxmax()]
            st.info(
                f"Highest F1 score in this feature set: "
                f"{best['Model']} — {best['F1_score']:.4f}"
            )
            st.caption(
                "Metrics exclude transactions labelled unknown. "
                "The anomaly flag is treated as a predicted illicit label "
                "only for this post-hoc evaluation."
            )
        else:
            st.error(
                "Required evaluation columns are missing. "
                f"Available columns: {selected.columns.tolist()}"
            )


# --------------------------------------------------
# 7. FEATURE COMPARISON
# --------------------------------------------------

elif page == "Feature Comparison":

    st.title("Original vs Enriched Features")

    st.write("""
    This section examines how adding transaction-network connectivity
    features changes the anomaly decisions made by each model.
    """)

    if comparison is not None:

        st.subheader("Anomaly Decision Consistency")

        st.dataframe(
            comparison.style.format({
                "Jaccard_similarity": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Jaccard Similarity")

        chart = comparison.set_index("Model")[
            ["Jaccard_similarity"]
        ]

        st.bar_chart(chart)

        st.caption(
            "A value closer to 1 indicates greater overlap between "
            "the original and enriched model's flagged transactions."
        )

    else:
        show_missing_file("original_vs_enriched_comparison.csv")

    st.divider()

    if pca_comparison is not None:

        st.subheader("PCA Explained Variance")

        st.dataframe(
            pca_comparison,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            "The original feature set explains 17.25% of variance "
            "in the first two components, compared with 17.03% "
            "for the enriched feature set."
        )


# --------------------------------------------------
# 8. VISUAL ANALYTICS
# --------------------------------------------------

elif page == "Visual Analytics":

    st.title("Visual Analytics")

    plot_information = {
        "Original Anomaly Detection Comparison":
            "anomaly_detection_comparison.png",

        "Original PCA — Transaction Labels":
            "pca_transaction_labels.png",

        "Enriched PCA — Transaction Features":
            "pca_enriched_features.png"
    }

    selected_plot = st.selectbox(
        "Choose visualisation",
        list(plot_information.keys())
    )

    plot_path = PLOTS / plot_information[selected_plot]

    if plot_path.exists():

        st.subheader(selected_plot)

        st.image(
            str(plot_path),
            use_container_width=True
        )

    else:
        st.error(f"Plot not found: {plot_path.name}")

    st.divider()

    st.subheader("PCA Interpretation")

    st.write("""
    PCA compresses the feature space into two principal components.
    These components capture only a portion of the total variance.

    Therefore, visible overlap between licit, illicit and unknown
    transactions does not establish that the original high-dimensional
    data is indistinguishable.
    """)


# --------------------------------------------------
# 9. TRANSACTION EXPLORER
# --------------------------------------------------

elif page == "Transaction Explorer":

    st.title("Transaction Explorer")

    st.write("""
    Explore the saved transaction-level anomaly flags and
    download the experimental results.
    """)

    feature_choice = st.radio(
        "Select result set",
        ["Enriched features", "Original features"],
        horizontal=True
    )

    selected_data = (
        enriched_flags
        if feature_choice == "Enriched features"
        else original_flags
    )

    if selected_data is None:

        st.error("The selected transaction result file was not found.")

    else:

        st.subheader("Filter Transactions")

        model_columns = [
            col for col in selected_data.columns
            if col.startswith("IF_") or col.startswith("LOF_")
        ]

        selected_model = st.selectbox(
            "Choose anomaly detection model",
            model_columns
        )

        anomaly_filter = st.selectbox(
            "Transaction status",
            ["All transactions", "Flagged anomalies", "Not flagged"]
        )

        filtered = selected_data.copy()

        if anomaly_filter == "Flagged anomalies":
            filtered = filtered[filtered[selected_model] == True]

        elif anomaly_filter == "Not flagged":
            filtered = filtered[filtered[selected_model] == False]

        st.metric("Transactions displayed", len(filtered))

        search = st.text_input("Search transaction ID")

        if search.strip() and "txId" in filtered.columns:
            filtered = filtered[
                filtered["txId"].astype(str).str.contains(
                    search.strip(),
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download filtered transactions",
            data=csv_data,
            file_name="filtered_transaction_results.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# 10. METHODOLOGY
# --------------------------------------------------

elif page == "Methodology":

    st.title("Methodology and Responsible Interpretation")

    st.subheader("Dataset")

    st.write("""
    The Elliptic Bitcoin Dataset contains transaction-level
    features, transaction classes and transaction relationships.
    """)

    st.subheader("Data Preparation")

    st.markdown("""
    - Loaded transaction features, labels and edge-list data.
    - Checked missing values and duplicate transaction IDs.
    - Selected 165 numerical transaction features.
    - Derived incoming and outgoing network connection counts.
    - Applied log transformation and standardisation to the
      two additional network features.
    """)

    st.subheader("Anomaly Detection")

    st.markdown("""
    **Isolation Forest**
    - 200 estimators.
    - Contamination settings of 2% and 5%.

    **Local Outlier Factor**
    - Neighbour settings of 20 and 50.
    - Contamination settings of 2% and 5%.
    """)

    st.subheader("Evaluation")

    st.markdown("""
    - Precision, recall and F1 score.
    - Jaccard similarity between model outputs.
    - Comparison of original and enriched feature sets.
    - PCA visualisation.
    - False-positive implications.
    """)

    st.subheader("Limitations")

    st.markdown("""
    1. The study uses a reproducible 10,000-transaction sample.
    2. The first two PCA components explain only a small proportion
       of total variance.
    3. The models flag unusual transactions, not confirmed crimes.
    4. Dataset labels are incomplete; unknown transactions are
       excluded from label-based evaluation.
    5. Network connectivity describes transaction relationships,
       not verified wallet ownership.
    6. Low precision and recall limit practical detection claims.
    """)

    st.subheader("Ethical Considerations")

    st.write("""
    False-positive anomaly flags can cause unnecessary scrutiny,
    wasted investigative resources and unfair suspicion.
    Model outputs should be treated as indicators for further
    review rather than definitive judgments.
    """)


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Bitcoin Wallet Behaviour Clustering for Suspicious Pattern "
    "Identification | SIH26146 | Unsupervised Learning Project"
)
