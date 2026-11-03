# Bitcoin Wallet Behaviour Clustering for Suspicious Pattern Identification

**SIH Problem Code:** SIH26146

## 1. Project Overview

This project investigates unsupervised anomaly detection in the Elliptic Bitcoin Dataset. It applies Isolation Forest and Local Outlier Factor (LOF) to identify statistically unusual Bitcoin transactions and compares their results across different parameter settings.

Principal Component Analysis (PCA) is used to visualise transaction features in two dimensions. Available licit, illicit, and unknown labels are kept separate from model inputs and used only for post-hoc evaluation.

The project also investigates whether adding transaction-network connectivity features changes anomaly detection results.

**Important scope note:** The Elliptic dataset represents Bitcoin transactions and their relationships, not a direct dataset of individual wallet owners. The analysis is therefore transaction-level; it does not establish the behaviour or identity of particular wallet owners. Graph-connectivity features describe transaction relationships and should not be interpreted as wallet ownership.

## 2. Objectives

### Objective 1 — Data Preparation

Prepare the selected data source (Elliptic Bitcoin Dataset), identify relevant features, handle missing values, encode categorical variables where required, and scale numerical variables before applying the selected unsupervised learning methods.

### Objective 2 — Anomaly Detection and Comparison

Apply Isolation Forest and Local Outlier Factor to the prepared features, compare observations flagged by each method across multiple parameter settings, and use PCA visualization plus available labels or manual review to assess detection consistency.

### Objective 3 — Evaluation and Responsible Interpretation

Evaluate detection consistency across methods and parameter settings using available labels or manual review, examine false-positive implications, and document limitations, possible bias, and ethical considerations required for responsible anomaly interpretation in practice.

## 3. Dataset

The project uses the **Elliptic Bitcoin Dataset**, which contains:

- Transaction features: 203,769 transactions and 165 numerical features, in addition to transaction ID and time step.
- Transaction classes: licit, illicit, and unknown.
- Transaction edges: 234,355 relationships between transactions.
- Time steps: 1–49.

### Dataset sources

- [Elliptic Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- [Elliptic dataset announcement](https://www.elliptic.co/insights/elliptic-dataset-cryptocurrency-financial-crime/)

The raw dataset files are not included in this repository. Download them from the source above and place the extracted CSV files in:

`data/raw/elliptic/`

The raw data directory is excluded from Git tracking.

## 4. Methodology

### 4.1 Data preparation

- Loaded and inspected the transaction features, class labels, and edge list.
- Merged transaction features with class labels using transaction IDs.
- Checked missing values and duplicate transaction IDs.
- Kept labels separate from the model feature matrix.
- Selected 165 numerical features for the original anomaly detection experiments.
- Used a reproducible random sample of 10,000 transactions (`random_state=42`).
- The original numerical features already appeared standardised, so no additional scaling was applied to them.

### 4.2 Original anomaly detection

Two unsupervised methods were compared.

**Isolation Forest**
- 200 estimators, 2% contamination.
- 200 estimators, 5% contamination.

**Local Outlier Factor**
- 20 neighbours, 2% contamination.
- 20 neighbours, 5% contamination.
- 50 neighbours, 2% contamination.
- 50 neighbours, 5% contamination.

### 4.3 Transaction-network feature enrichment

The Elliptic edge list was used to derive two additional transaction-network connectivity features:

- `incoming_connections`: number of incoming transaction relationships.
- `outgoing_connections`: number of outgoing transaction relationships.

A logarithmic transformation (`log1p`) and standardisation were applied to these two features. They were then combined with the original 165 numerical features to create a 167-feature enriched dataset.

The total connection count was not included because it is redundant with the incoming and outgoing counts.

The graph-enriched dataset was evaluated using the same Isolation Forest and LOF configurations as the original dataset.

These features describe transaction connectivity, not wallet ownership or a complete graph-based representation of Bitcoin behaviour.

### 4.4 PCA visualisation

PCA reduced the feature matrices to two principal components for visualisation.

- Original features: approximately **17.25%** of variance explained by the first two components.
- Graph-enriched features: approximately **17.03%** of variance explained by the first two components.

The two-dimensional projections provide a visual summary but cannot represent all the structure in the original feature space.

### 4.5 Evaluation

- Compared anomaly flags using Jaccard similarity.
- Evaluated flagged observations against available labels.
- Calculated precision, recall, and F1-score using known labels only.
- Examined flagged licit and unknown transactions to discuss false-positive implications and uncertainty.
- Compared original and graph-enriched anomaly flags to investigate the effect of additional connectivity features.

Labels were not used as model inputs or to train the anomaly detection algorithms.

## 5. Key Results

All results below are based on the reproducible sample of 10,000 transactions. Label-based evaluation excludes transactions with unknown labels.

### 5.1 Original feature evaluation

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Isolation Forest (2%) | 0.0000 | 0.000 | 0.0000 |
| Isolation Forest (5%) | 0.0000 | 0.000 | 0.0000 |
| LOF (20, 2%) | 0.0781 | 0.020 | 0.0318 |
| LOF (20, 5%) | 0.0452 | 0.036 | 0.0401 |
| LOF (50, 2%) | 0.0571 | 0.016 | 0.0250 |
| LOF (50, 5%) | 0.0437 | 0.032 | 0.0370 |

LOF with 20 neighbours and 5% contamination achieved the highest recall and F1-score among the original configurations. It detected 9 of the 250 labelled illicit transactions in the sample.

Neither tested Isolation Forest configuration flagged a labelled illicit transaction in this sample.

### 5.2 Agreement between original methods

Jaccard similarity showed low overlap between Isolation Forest and LOF anomaly flags, approximately 0.042–0.076 for the tested comparisons.

Agreement between LOF configurations ranged from approximately 0.308 to 0.481. This indicates that anomaly detection results depend on the selected algorithm and parameters.

### 5.3 Graph-enriched feature evaluation

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Isolation Forest (2%) | 0.0000 | 0.000 | 0.0000 |
| Isolation Forest (5%) | 0.0000 | 0.000 | 0.0000 |
| LOF (20, 2%) | 0.0533 | 0.016 | 0.0246 |
| LOF (20, 5%) | 0.0461 | 0.040 | 0.0428 |
| LOF (50, 2%) | 0.0575 | 0.020 | 0.0297 |
| LOF (50, 5%) | 0.0332 | 0.028 | 0.0304 |

LOF with 20 neighbours and 5% contamination achieved the highest F1-score among the enriched configurations. It detected 10 of the 250 labelled illicit transactions.

The enriched results show a small improvement in recall and F1-score for this configuration compared with the original feature experiment. This does not establish that graph enrichment improves detection in general.

### 5.4 Original versus enriched anomaly flags

| Model | Original vs. enriched Jaccard similarity |
|---|---:|
| Isolation Forest (2%) | 0.8265 |
| Isolation Forest (5%) | 0.7794 |
| LOF (20, 2%) | 0.6878 |
| LOF (20, 5%) | 0.7271 |
| LOF (50, 2%) | 0.7167 |
| LOF (50, 5%) | 0.7544 |

Most anomaly flags remained consistent after adding transaction-network connectivity features, although some transactions changed classification.

Isolation Forest showed higher agreement between original and enriched feature configurations than LOF in these experiments.

### 5.5 PCA comparison

- Original feature set: 17.25% explained variance.
- Graph-enriched feature set: 17.03% explained variance.
- Difference: approximately -0.22 percentage points.

Adding connectivity features changed some anomaly detection decisions but did not improve the variance represented by the first two principal components.

### 5.6 False-positive observations

In the original experiment, LOF with 20 neighbours and 5% contamination flagged 190 transactions labelled licit and 9 labelled illicit.

In the enriched experiment, the same configuration flagged 190 transactions labelled licit and 10 labelled illicit.

These observations demonstrate why an anomaly flag should be treated as an indicator for further review, not proof of illicit activity. A flagged transaction labelled licit is a potential false alarm relative to the available label, not proof of wrongdoing by any wallet owner.

## 6. Streamlit Dashboard

The project includes a Streamlit dashboard for exploring the completed analysis.

The dashboard provides:

- Project overview and summary metrics.
- Original and graph-enriched model performance.
- Comparison of anomaly detection configurations.
- Feature-enrichment comparison.
- PCA and anomaly detection visualisations.
- Transaction anomaly flag exploration.
- Methodology and responsible interpretation information.

### Run the dashboard locally

From the project root directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard currently displays precomputed results generated by the analysis notebook. It does not independently retrain the models or provide live predictions for newly uploaded transactions.

## 7. Project Structure

```text
bitcoin-wallet-anomaly-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── raw/
│       └── elliptic/              # Local dataset files; not tracked by Git
│
├── notebooks/
│   └── bitcoin_anomaly_detection.ipynb
│
└── results/
    ├── metrics/
    │   ├── model_evaluation.csv
    │   ├── model_agreement.csv
    │   ├── transaction_anomaly_flags.csv
    │   ├── false_positive_summary.csv
    │   ├── enriched_model_evaluation.csv
    │   ├── enriched_transaction_anomaly_flags.csv
    │   ├── original_vs_enriched_comparison.csv
    │   └── pca_variance_comparison.csv
    │
    └── plots/
        ├── anomaly_detection_comparison.png
        ├── pca_transaction_labels.png
        └── pca_enriched_features.png
```

## 8. Installation and Reproducibility

### Requirements

- Python
- Jupyter Notebook
- VS Code with the Python and Jupyter extensions

### Clone the repository

```bash
git clone https://github.com/SnehasishGhosh226/bitcoin-wallet-anomaly-detection.git
cd bitcoin-wallet-anomaly-detection
```

### Create and activate a virtual environment on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Prepare the dataset

Download the Elliptic dataset and place the three CSV files in:

`data/raw/elliptic/`

### Run the analysis

Open `notebooks/bitcoin_anomaly_detection.ipynb` in VS Code, select the project Python environment as the notebook kernel, and run the notebook cells from top to bottom.

### Run the dashboard

```bash
streamlit run app.py
```

For reproducibility, the experiments use a fixed random seed (`random_state=42`) and a sample of 10,000 transactions.

## 9. Limitations and Responsible Interpretation

- **Incomplete labels:** Many transactions have unknown labels. Label-based evaluation covers only transactions with known labels.
- **Transaction-level scope:** The analysis identifies unusual transaction patterns; it does not directly establish wallet ownership or criminal behaviour.
- **Feature interpretation:** Some dataset features have limited transparency, which restricts the interpretation of why a transaction is flagged.
- **Sampling:** The model comparison uses a reproducible sample of 10,000 transactions rather than the complete dataset.
- **Parameter sensitivity:** Different algorithms and parameter settings identify different observations.
- **PCA limitations:** The two-dimensional PCA projections explain approximately 17% of the variance, so they cannot represent all structure in the original feature space.
- **Graph information:** Incoming and outgoing transaction-connection counts provide limited network information. They do not constitute a complete graph-based model or establish wallet identity.
- **False positives:** A labelled licit transaction flagged as anomalous is a potential false alarm relative to the available label. It should not be treated as proof of wrongdoing.
- **Scaling and generalisation:** The graph-feature scaler was fitted on the exploratory sample. A future prediction pipeline must retain and reuse fitted preprocessing parameters rather than refitting them independently on new data.
- **Temporal and dataset limitations:** Existing feature standardisation and the construction of some dataset features may limit claims about temporal generalisation and real-world deployment.
- **Dashboard scope:** The current Streamlit dashboard visualises precomputed experiment results and is not a live transaction-monitoring system.

Anomaly detection results should support further investigation, not automatic accusations, enforcement decisions, or conclusions about criminal intent.

## 10. Project Outcome

Produce a reproducible anomaly-detection study using documented preprocessing, Isolation Forest and LOF comparisons, multiple parameter settings, PCA visualization, and label- or review-based evaluation where available. The final submission should identify consistent anomaly patterns, limitations, false-positive risks, and responsible interpretation for the selected application domain.

## 11. Author

**GitHub Repository:** [bitcoin-wallet-anomaly-detection](https://github.com/SnehasishGhosh226/bitcoin-wallet-anomaly-detection)
