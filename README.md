# 🏦 Credit Risk Assessment Model

## 📌 Project Overview
This project builds a robust machine learning pipeline to predict loan defaults, specifically addressing the challenge of **highly imbalanced financial datasets**. 

Unlike standard accuracy-focused models, this system prioritizes a **"Safety-First" decision framework**. It is optimized to minimize financial risk by aggressively identifying potential defaulters, even at the cost of rejecting some safe applicants.

## 🚀 Key Features
* **Imbalanced Data Handling:** Utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset and prevent model bias toward the majority class.
* **Model Benchmarking:** Compared Logistic Regression, Decision Trees, and Random Forest, with **Random Forest** selected as the champion model.
* **Threshold Optimization:** Shifted the decision threshold from the standard `0.5` to `0.65` to maximize the capture of high-risk loans.
* **Business-Centric Metrics:** Optimized for **F2-Score** (0.77) and **Recall** rather than simple accuracy.

## 📊 Performance & Results
The model was evaluated based on its ability to prevent capital loss (i.e., minimizing False Negatives).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.84** | Strong ability to distinguish between defaulters and non-defaulters. |
| **Defaulter Recall** | **86.8%** | Captures nearly 9 out of 10 actual defaulters. |
| **F2-Score** | **0.77** | Balances precision and recall with a heavier weight on recall. |

### 📉 Business Impact (Confusion Matrix Analysis)
By tuning the probability threshold to **0.65**, the model achieved a critical reduction in risk:
* **Baseline (Threshold 0.5):** 14 False Negatives (High financial risk).
* **Optimized (Threshold 0.65):** **5 False Negatives** (Risk reduced by ~64%).

> *Trade-off: This strategy intentionally accepts a higher rejection rate to ensure capital safety, demonstrating a practical approach to credit risk management.*

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Sampling:** Imbalanced-learn (SMOTE)
* **Visualization:** Matplotlib, Seaborn

## 💻 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/HarshitChhabra28/Credit-Risk-Assessment-Model.git](https://github.com/HarshitChhabra28/Credit-Risk-Assessment-Model.git)
    cd Credit-Risk-Assessment-Model
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook**
    ```bash
    jupyter notebook "data cleaning.ipynb"
    ```

## 📂 Project Structure
