# 🥇 #1 Kaggle Competition: Credit Card Fraud Detection

This project builds a machine learning model to detect fraudulent transactions, winning **1st place** in a Kaggle competition.  
The solution uses custom feature engineering and a high-performance **CatBoostClassifier** to achieve top results.

---

## 🛠️ Project Structure

- **data/**
  - `train.csv` — Training data
  - `test.csv` — Test data for final predictions
  - `sample_submission.csv` — Submission file format
- **submission.csv** — Final predictions (generated after model training)

---

## 🚀 Approach

1. **Feature Engineering**
   - Extracted transaction **hour** from `unix_time`.
   - Combined `first` and `last` names into a single `full_name` feature.
   - Derived **age** from `dob`.
   - Dropped unnecessary fields (e.g., location coordinates, raw timestamps).

2. **Modeling**
   - Model: **CatBoostClassifier**
   - Parameters: `iterations=5000`, `depth=6`, `learning_rate=0.7`
   - Handled categorical features: `category`, `gender`, `city`, `state`, `job`, `merchant`, `full_name`, `street`

3. **Training Strategy**
   - Trained on the full training set to maximize available data.
   - No validation split used in final submission (to achieve highest possible score).

4. **Prediction**
   - Predictions are generated on the test dataset.
   - Final predictions are saved in `submission.csv`.

---

## 🧩 How to Run

1. Install dependencies:
   ```bash
   pip install pandas catboost scikit-learn
