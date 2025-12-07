# Telco Customer Churn Prediction (KH6001CMD)

## Project Overview
This project is a machine learning pipeline designed to predict customer churn in the telecommunications sector. It was developed as part of the **KH6001CMD Machine Learning** module coursework.

The solution formulates the problem as a binary classification task. It uniquely combines **unsupervised learning** (K-Means clustering for feature engineering) with **supervised learning** algorithms to improve predictive performance.

## 📂 Dataset
The dataset used for this project is the **Telco Customer Churn** dataset provided by Blastchar.

**[Link to Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**

*Note: The `WA_Fn-UseC_-Telco-Customer-Churn.csv` file should be placed in the root directory of this repository to run the notebook.*

## 🛠️ Methodology
The pipeline consists of the following stages:

1.  **Data Preprocessing:**
    * **Leakage-Free Design:** Imputation of missing values is integrated directly into the training pipeline (using `SimpleImputer`) to prevent data leakage.
    * Duplicate and outlier inspection.
    * One-Hot Encoding for categorical variables.
2.  **Unsupervised Feature Engineering:**
    * Applied **K-Means Clustering** to numerical features to identify latent customer subgroups.
    * **Elbow Method** was used to statistically justify the choice of $K=2$.
    * The resulting cluster labels were added as a new feature for the supervised models.
3.  **Supervised Modelling:**
    * **Logistic Regression** (Baseline & Tuned)
    * **k-Nearest Neighbours (KNN)**
    * **Random Forest** (Baseline & Tuned)
4.  **Evaluation:**
    * **Optimization Metric:** Models were tuned using **F1-Score** to address class imbalance.
    * **Business Impact:** Analysis focuses on Recall (minimizing False Negatives) to estimate revenue protection.

## 📊 Key Results
The models were evaluated on a stratified test set (20% hold-out).

| Model | Accuracy | F1-Score | Recall |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Tuned)** | **80.55%** | **0.605** | **0.561** |
| Logistic Regression (Baseline) | 80.41% | 0.602 | 0.558 |
| Random Forest (Tuned) | 79.84% | 0.574 | 0.511 |
| Random Forest (Baseline) | 79.28% | 0.581 | 0.527 |
| k-Nearest Neighbours | 76.37% | 0.553 | 0.551 |

*Note: Metrics may vary slightly based on the final run.*

The **Tuned Logistic Regression** model achieved the best balance of Accuracy and F1-Score. The decision to optimize for F1-Score improved the model's ability to detect churners compared to a standard accuracy-based approach.

## 💻 Installation & Usage

### Prerequisites
To run this notebook, you need Python installed along with the following libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

You can install the dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
