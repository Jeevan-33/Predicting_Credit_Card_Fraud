# 🚀 Predicting Credit Card Fraud — Machine Learning Experiments

A hands-on machine learning project using the Kaggle Playground Series credit card fraud dataset. This repository explores gradient boosting, model benchmarking, hyperparameter optimization, cross-validation, and model evaluation.

## 🎯 Project Objectives

- Compare popular gradient boosting algorithms.
- Evaluate model performance using ROC-AUC.
- Understand training time and early stopping.
- Optimize CatBoost hyperparameters using Optuna.
- Measure the difference between cross-validation and held-out test performance.
- Understand model selection bias.

## 📊 Dataset

- **Competition:** Kaggle Playground Series — Season 6, Episode 11
- **Task:** Binary Classification
- **Target:** `is_fraud`
- **Evaluation Metric:** ROC-AUC

[Kaggle Competition — Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- Optuna
- Matplotlib
- Jupyter Notebook

---

## Experiment 1: Gradient Boosting Benchmark

Compared three gradient boosting algorithms using an 80/20 stratified train-validation split.

### Models

- CatBoost
- LightGBM
- XGBoost

### Benchmark Results

| Rank | Model | Validation ROC-AUC | Best Iteration | Training Time (sec) |
|---|---|---:|---:|---:|
| 🥇 1 | CatBoost | **0.944426** | 440 | 24.14 |
| 🥈 2 | LightGBM | 0.943152 | 169 | **7.99** |
| 🥉 3 | XGBoost | 0.942245 | 148 | 15.65 |

### Key Findings

- CatBoost achieved the highest validation ROC-AUC of **0.944426**.
- LightGBM was the fastest, completing training in approximately **7.99 seconds**.
- The differences between the three ROC-AUC scores were relatively small.
- Additional validation would be needed to establish whether the performance differences are consistent.

---

## Experiment 2: CatBoost Hyperparameter Optimization with Optuna

The second experiment focuses on optimizing CatBoost using Optuna.

### Experimental Setup

| Setting | Value |
|---|---|
| Final holdout test size | 20% |
| Development data | 80% |
| Cross-validation | 5-fold Stratified CV |
| Optimization library | Optuna |
| Number of trials | 40 |
| Optimization objective | Maximize ROC-AUC |
| Pruning | Median Pruner |
| Early stopping | 100 rounds |

### Hyperparameters Tuned

- Learning rate
- Tree depth
- L2 leaf regularization
- Random strength
- Bagging temperature

### Evaluation Strategy

1. Reserve 20% of the original training data as a final test set.
2. Use the remaining 80% for hyperparameter optimization.
3. Run 5-fold stratified cross-validation for each Optuna trial.
4. Apply early stopping and pruning during optimization.
5. Record the best cross-validation ROC-AUC and parameters.
6. Compare the default-parameter CV score against the optimized CV score.
7. Refit the selected model on the development data.
8. Evaluate the final model once on the untouched test set.

### Optuna Results

| Metric | Result |
|---|---:|
| Default CatBoost CV ROC-AUC | Pending |
| Best Optuna CV ROC-AUC | Pending |
| Final Test ROC-AUC | Pending |
| CV - Test Gap | Pending |

### Best Hyperparameters

Results will be added after completing the Optuna experiment.

### Understanding Selection Bias

The difference between the best cross-validation score and the final held-out test score is:

**Selection gap = Best CV ROC-AUC − Final Test ROC-AUC**

A positive gap means the selected model performed better during cross-validation than on the final test set.

This gap provides an estimate of how much the model-selection process may have benefited from validation noise.

---

## 📂 Project Structure

```text
Predicting_Credit_Card_Fraud/
│
├── Predicting_Credit_Card_Fraud.ipynb
├── Optuna_hyperparameter_tuning.ipynb
├── README.md
└── .gitignore
```

## 💡 Key Learnings

- Comparing gradient boosting algorithms.
- Building stratified train-validation splits.
- Understanding early stopping.
- Using ROC-AUC for binary classification.
- Hyperparameter optimization with Optuna.
- Applying pruning to reduce unnecessary trial computation.
- Using 5-fold stratified cross-validation.
- Keeping a final test set separate from model selection.
- Understanding the difference between validation performance and final test performance.
- Recognizing potential selection bias.

## 🔮 Future Improvements

- Explore additional feature engineering.
- Compare different boosting algorithms after optimization.
- Analyze feature importance.
- Experiment with different cross-validation strategies.
- Generate Kaggle competition predictions.
- Improve leaderboard performance.

---

## 👨‍💻 About This Project

This repository is part of my practical Machine Learning learning journey, focusing on implementing algorithms, conducting experiments, evaluating models, and developing skills through Kaggle competitions.

**Learning by doing. Experimenting. Improving.** 🚀
