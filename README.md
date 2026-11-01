# Bitcoin Wallet Behaviour Clustering for Suspicious Pattern Identification

**SIH Problem Code:** SIH26146

## 1. Project Overview

This project investigates unsupervised anomaly detection in the Elliptic Bitcoin Dataset. It applies Isolation Forest and Local Outlier Factor (LOF) to identify statistically unusual Bitcoin transactions and compares their results across different parameter settings.

Principal Component Analysis (PCA) is used to visualise transaction features in two dimensions. Available licit, illicit, and unknown labels are kept separate from model inputs and used only for post-hoc evaluation.

**Important scope note:** The Elliptic dataset represents Bitcoin transactions and their relationships, not a direct dataset of individual wallet owners. The analysis is therefore transaction-level; it does not establish the behaviour or identity of particular wallet owners.

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

### Data preparation

- Loaded and inspected the transaction features, class labels, and edge list.
- Merged transaction features with class labels using transaction IDs.
- Checked missing values and duplicate transaction IDs.
- Kept labels separate from the model feature matrix.
- Selected 165 numerical features for anomaly detection.
- Used a reproducible random sample of 10,000 transactions (`random_state=42`).
- The numerical features already appeared standardised, so no additional scaling was applied.

### Anomaly detection

Two unsupervised methods were compared:

**Isolation Forest**
- 200 estimators, 2% contamination.
- 200 estimators, 5% contamination.

**Local Outlier Factor**
- 20 neighbours, 2% contamination.
- 20 neighbours, 5% contamination.
- 50 neighbours, 2% contamination.
- 50 neighbours, 5% contamination.

### PCA visualisation

PCA reduced the 165-dimensional feature matrix to two principal components for visualisation. The first two components explained approximately **17.25%** of the variance.

### Evaluation

- Compared anomaly flags using Jaccard similarity.
- Evaluated flagged observations against available labels.
- Calculated precision, recall, and F1-score using known labels only.
- Examined flagged licit and unknown transactions to discuss false-positive implications and uncertainty.

Labels were not used as model inputs or to train the anomaly detection algorithms.

## 5. Key Results

### Label-based evaluation

Results below are based on the 10,000-transaction sample and its known labels.

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Isolation Forest (2%) | 0.0000 | 0.000 | 0.0000 |
| Isolation Forest (5%) | 0.0000 | 0.000 | 0.0000 |
| LOF (20, 2%) | 0.0781 | 0.020 | 0.0318 |
| LOF (20, 5%) | 0.0452 | 0.036 | 0.0401 |
| LOF (50, 2%) | 0.0571 | 0.016 | 0.0250 |
| LOF (50, 5%) | 0.0437 | 0.032 | 0.0370 |

LOF with 20 neighbours and 5% contamination achieved the highest recall and F1-score among the tested configurations. It detected 9 of the 250 labelled illicit transactions in the sample. Its recall was nevertheless low.

Neither tested Isolation Forest configuration flagged a labelled illicit transaction in this sample.

### Agreement between methods

Jaccard similarity showed low overlap between Isolation Forest and LOF anomaly flags, approximately 0.042–0.076 for the tested comparisons.

Agreement between LOF configurations ranged from approximately 0.308 to 0.481. This indicates that anomaly detection results depend on the selected algorithm and parameters.

### False-positive observations

The LOF configuration with 20 neighbours and 5% contamination flagged 190 transactions labelled licit and 9 labelled illicit. Isolation Forest at 5% contamination flagged 409 transactions labelled licit and no labelled illicit transactions.

These results demonstrate why an anomaly flag should be treated as an indicator for further review, not proof of illicit activity.

## 6. Project Structure

```text
bitcoin-wallet-anomaly-detection/
│
├── data/
│   └── raw/
│       └── elliptic/              # Local dataset files; not tracked by Git
│
├── notebooks/
│   └── bitcoin_anomaly_detection.ipynb
│
├── results/
│   ├── metrics/
│   │   ├── model_evaluation.csv
│   │   ├── model_agreement.csv
│   │   ├── transaction_anomaly_flags.csv
│   │   └── false_positive_summary.csv
│   │
│   └── plots/
│       ├── anomaly_detection_comparison.png
│       └── pca_transaction_labels.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 7. Installation and Reproducibility

### Requirements

- Python
- Jupyter Notebook
- VS Code with the Python and Jupyter extensions

### Install dependencies

Clone the repository:

```bash
git clone https://github.com/SnehasishGhosh226/bitcoin-wallet-anomaly-detection.git
cd bitcoin-wallet-anomaly-detection
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the Elliptic dataset and place the three CSV files in `data/raw/elliptic/`.

Open `notebooks/bitcoin_anomaly_detection.ipynb` in VS Code, select the project Python environment as the notebook kernel, and run the notebook cells from top to bottom.

## 8. Limitations and Responsible Interpretation

- **Incomplete labels:** Many transactions have unknown labels. Label-based evaluation covers only transactions with known labels.
- **Transaction-level scope:** The analysis identifies unusual transaction patterns; it does not directly establish wallet ownership or criminal behaviour.
- **Feature interpretation:** Some dataset features have limited transparency, which restricts the interpretation of why a transaction is flagged.
- **Sampling:** The model comparison uses a reproducible sample of 10,000 transactions rather than the complete dataset.
- **Parameter sensitivity:** Different algorithms and parameter settings identify different observations.
- **PCA limitations:** The two-dimensional PCA projection explains only 17.25% of the variance, so it cannot represent all structure in the original feature space.
- **Graph information:** Although the dataset includes transaction relationships, the anomaly detection models in this implementation use the numerical transaction features rather than explicitly modelling the graph structure.
- **False positives:** A labelled licit transaction flagged as anomalous is a potential false alarm relative to the available label. It should not be treated as proof of wrongdoing.
- **Temporal and dataset limitations:** Existing feature standardisation and the construction of some dataset features may limit claims about temporal generalisation and real-world deployment.

Anomaly detection results should support further investigation, not automatic accusations, enforcement decisions, or conclusions about criminal intent.

## 9. Project Outcome

Produce a reproducible anomaly-detection study using documented preprocessing, Isolation Forest and LOF comparisons, multiple parameter settings, PCA visualization, and label- or review-based evaluation where available. The final submission should identify consistent anomaly patterns, limitations, false-positive risks, and responsible interpretation for the selected application domain.

## 10. Author

**GitHub Repository:** [bitcoin-wallet-anomaly-detection](https://github.com/SnehasishGhosh226/bitcoin-wallet-anomaly-detection)