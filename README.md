# Credit-Card-Fraud-Detection-Hybrid
Hybrid Fraud Detection model using Random Forest (SMOTE) and Isolation Forest.
# Credit Card Fraud Detection - Hybrid Ensemble Model 🛡️

This project implements a robust **Hybrid Machine Learning Model** designed to detect fraudulent credit card transactions with high accuracy. 

By combining **Random Forest** (Supervised Learning) and **Isolation Forest** (Unsupervised Anomaly Detection), this approach addresses the severe class imbalance problem inherent in fraud datasets, achieving a significantly higher recall rate compared to single-model approaches.

## 🚀 Key Features

* **Hybrid Architecture:** Combines the predictive power of Random Forest with the anomaly detection capabilities of Isolation Forest.
* **Imbalance Handling:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training data.
* **Threshold Tuning:** Implements a custom decision threshold (`0.22`) to maximize sensitivity (Recall) for catching fraud cases.
* **Weighted Soft Voting:** Final decision is based on a weighted average (80% RF / 20% IF) to reduce false negatives.

## 📊 Model Performance

Traditional models often struggle with "imbalanced data," missing many fraud cases. This hybrid approach prioritizes catching thieves (High Recall) while keeping false alarms manageable.

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **84.46%** | The model successfully detected **125** out of **148** total fraud cases. |
| **False Positives** | **72** | Only 72 legitimate transactions were flagged incorrectly (out of 85,000+). |
| **Missed Fraud (FN)** | **23** | Only 23 fraud cases went undetected. |
| **Improvement** | **~3x** | Compared to standard Isolation Forest (which caught only ~43 cases), this model is significantly more effective. |

## 🛠️ Tech Stack

* **Python 3.x**
* **Scikit-Learn** (RandomForest, IsolationForest, Metrics)
* **Imbalanced-Learn** (SMOTE)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Data Visualization)

## 📂 Dataset

This project uses the famous **Credit Card Fraud Detection** dataset.
Due to GitHub's file size limits, the dataset is not included in this repository.

👉 **[Download the Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

**Instructions:**
1.  Download `creditcard.csv` from the link above.
2.  Place the file in the same directory as the script.
3.  Run the Python script.

## 🧠 Methodology (How it Works)

1.  **Data Preprocessing:** The `Amount` feature is scaled using `StandardScaler`.
2.  **SMOTE Application:** Synthetic samples are generated for the minority class (Fraud) to prevent the model from being biased towards the majority class.
3.  **Random Forest (The Expert):** Trained on the balanced dataset. It outputs a probability score.
    * *Optimization:* A custom threshold of `0.22` is applied instead of the standard `0.5` to catch subtle fraud signals.
4.  **Isolation Forest (The Anomaly Detector):** Runs on the original data to identify outliers based on density.
5.  **Ensemble Decision:** * Formula: `(RandomForest_Decision * 0.80) + (IsolationForest_Decision * 0.20)`
    * If the weighted score > 0.5, the transaction is flagged as **FRAUD**.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
