# 🚀 Kaggle Credit Card Fraud — Booster Benchmark

A machine learning benchmarking experiment using the Kaggle Playground Series credit card fraud dataset. This project compares **XGBoost, LightGBM, and CatBoost** to evaluate predictive performance, training speed, and boosting iterations.

---

## 🎯 Objective

Benchmark three popular gradient boosting algorithms on a tabular binary classification task using a consistent train-validation split.

The goal is to identify:
- Which model achieves the best validation ROC-AUC?
- Which model trains the fastest?
- How do boosting iterations and training time compare?
- Is the performance difference between models meaningful?

---

## 📊 Dataset

**Competition:** Kaggle Playground Series — Season 6, Episode 11  
**Task:** Binary Classification  
**Target:** `is_fraud`  
**Evaluation Metric:** ROC-AUC

🔗 [Kaggle Competition — Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)

---

## 🤖 Models Compared

| Model | Description |
|---|---|
| CatBoost | Gradient boosting optimized for categorical features |
| LightGBM | Efficient gradient boosting using histogram-based learning |
| XGBoost | Regularized gradient boosting with optimized tree construction |

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib

---

## 🔬 Methodology

1. Load and inspect the competition dataset.
2. Separate features and target variable.
3. Perform an **80/20 stratified train-validation split**.
4. Handle missing values and encode categorical features.
5. Train all three boosting models using comparable settings.
6. Use early stopping to determine the best boosting iteration.
7. Evaluate models using validation ROC-AUC.
8. Compare model performance, training time, and best iterations.

**Random State:** 42  
**Validation Strategy:** Stratified Holdout  
**Early Stopping:** Enabled

---

## 📈 Benchmark Results

| Rank | Model | Validation ROC-AUC | Best Iteration | Training Time (sec) |
|---|---|---:|---:|---:|
| 🥇 1 | CatBoost | **0.944426** | 440 | 24.14 |
| 🥈 2 | LightGBM | 0.943152 | 169 | **7.99** |
| 🥉 3 | XGBoost | 0.942245 | 148 | 15.65 |

### Performance Differences

| Comparison | ROC-AUC Difference |
|---|---:|
| CatBoost vs. LightGBM | 0.001274 |
| CatBoost vs. XGBoost | 0.002181 |
| LightGBM vs. XGBoost | 0.000907 |

---

## 🏆 Key Findings

- **Best Model:** CatBoost achieved the highest validation ROC-AUC of **0.944426**.
- **Fastest Model:** LightGBM completed training in approximately **7.99 seconds**.
- **CatBoost:** Achieved the highest score but required approximately 24.14 seconds of training.
- **LightGBM:** Delivered competitive performance while training approximately 3 times faster than CatBoost.
- **XGBoost:** Achieved a ROC-AUC of 0.942245 with a training time of approximately 15.65 seconds.

The differences in ROC-AUC are relatively small. CatBoost performed best on this particular validation split, but additional cross-validation or repeated experiments would be needed to determine whether its advantage is consistent and statistically meaningful.

---

## 📉 Booster Comparison

The benchmark shows that all three models achieve similar validation performance, with CatBoost slightly ahead.

![Booster Comparison](booster_comparison.png)

---

## 💡 What I Learned

- How to benchmark multiple gradient boosting algorithms fairly.
- How stratified train-validation splitting preserves class proportions.
- How early stopping helps determine the optimal number of boosting iterations.
- How to compare models using ROC-AUC.
- How to evaluate the trade-off between predictive performance and computational efficiency.
- Why small validation score differences should not automatically be considered meaningful.
- Why repeated validation or cross-validation is important for reliable model selection.

---

## 📂 Project Structure

```text
Predicting_Credit_Card_Fraud/
│
├── Predicting_Credit_Card_Fraud.ipynb
├── README.md
└── .gitignore
```

---

## 🔮 Next Steps

- Perform cross-validation to evaluate model stability.
- Tune hyperparameters for the best-performing booster.
- Explore feature engineering.
- Compare validation results against the Kaggle leaderboard.
- Generate predictions for the competition test set.
- Prepare a submission and evaluate leaderboard performance.

---

## 👨‍💻 Learning Journey

This project is part of my hands-on Machine Learning practice, focusing on algorithm understanding, experimentation, model evaluation, and Kaggle competitions.

**Learning by doing. Experimenting. Improving.** 🚀
