# Credit Card Fraud Detection | Kaggle Playground Series S6E11

## 📌 Project Overview

This project focuses on predicting fraudulent credit card transactions using machine learning models on the **Kaggle Playground Series Season 6, Episode 11** dataset.

The goal is to build, evaluate, compare, and improve classification models for a highly imbalanced binary classification problem. Experiments include model benchmarking, hyperparameter optimization, out-of-fold (OOF) predictions, adversarial validation, and final Kaggle submission.

The project follows a structured machine learning workflow with an emphasis on validation strategy, model performance, experimentation, and reproducibility.

---

## 🏆 Kaggle Results

| Metric | Score |
|---|---:|
| **Public Leaderboard ROC-AUC** | **0.94709** |
| Competition | Playground Series S6E11 |
| Kaggle Username | `jeevandevang` |
| Kaggle Display Name | Jeevan Devang |
| Submission Status | Submitted successfully |

### Kaggle Submission

My prediction file was successfully submitted to Kaggle and received a public leaderboard score of:

**0.94709**

[View Kaggle Competition Submissions](https://www.kaggle.com/competitions/playground-series-s6e11/submissions)

---

## 🎯 Objectives

- Understand and explore a tabular classification dataset.
- Establish baseline performance using multiple gradient boosting models.
- Compare CatBoost, LightGBM, and XGBoost.
- Perform hyperparameter optimization using Optuna.
- Evaluate model performance using stratified cross-validation.
- Generate out-of-fold predictions.
- Perform adversarial validation to investigate train-test distribution differences.
- Generate and validate the final Kaggle submission.
- Track experiments and document results.

---

## 📊 Dataset

The dataset comes from the Kaggle Playground Series Season 6, Episode 11 competition.

**Problem type:** Binary classification  
**Target variable:** `is_fraud`  
**Evaluation metric:** ROC-AUC

The target represents whether a transaction is fraudulent.

The dataset contains anonymized/tabular transaction features and a binary target. The objective is to predict the probability of fraud rather than simply classify transactions into 0 or 1.

Dataset source:

[Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11)

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- LightGBM
- XGBoost
- Optuna
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab
- Git & GitHub

---

## 🧪 Experiments

### Experiment 1 — Gradient Boosting Model Benchmark

Compared three gradient boosting algorithms using baseline/default-style configurations.

| Model | Validation ROC-AUC | Best Iteration | Training Time |
|---|---:|---:|---:|
| CatBoost | **0.944426** | 440 | 24.14 sec |
| LightGBM | 0.943152 | 169 | 7.99 sec |
| XGBoost | 0.942245 | 148 | 15.65 sec |

### Observations

- CatBoost achieved the highest validation ROC-AUC among the three benchmark models.
- LightGBM trained the fastest.
- XGBoost also performed competitively.
- CatBoost was selected for further experimentation.

> Note: The benchmark was conducted on a dataset/sample configuration different from some of the later experiments. Therefore, these results should not be interpreted as a perfectly controlled comparison across all project experiments.

---

### Experiment 2 — Hyperparameter Optimization with Optuna

Used Optuna to tune CatBoost hyperparameters.

#### Parameters explored

- Depth
- Learning rate
- L2 regularization
- Random strength
- Bagging temperature
- Border count

#### Optimization configuration

| Setting | Value |
|---|---|
| Model | CatBoost |
| Optimization framework | Optuna |
| Trials completed | 15 |
| Cross-validation | 3-fold Stratified CV |
| Tuning data | 20% subset of development data |
| Early stopping | Enabled |
| Pruning | Enabled |

The original plan was to run 40 trials with 5-fold stratified cross-validation. The actual experiment was completed using 15 trials and 3-fold CV on a subset of the development data.

#### Results

| Evaluation | ROC-AUC |
|---|---:|
| Default CatBoost CV | **0.945966** |
| Best Optuna CV | 0.941015 |
| Holdout Score | 0.944562 |
| CV–Holdout Difference | -0.003547 |

#### Best Hyperparameters

```python
{
    "depth": 4,
    "learning_rate": 0.1044004075,
    "l2_leaf_reg": 9.4156101644,
    "random_strength": 3.5695556312,
    "bagging_temperature": 0.1029224715,
    "border_count": 249
}
```

#### Observations

- Hyperparameter tuning did not outperform the default CatBoost baseline in this experiment.
- The default configuration achieved a CV ROC-AUC of 0.945966.
- The best Optuna trial achieved 0.941015.
- The experiment highlighted the importance of a consistent evaluation setup when comparing hyperparameter configurations.

**Validation note:** The holdout set was used as an evaluation set for early stopping. Therefore, the holdout score should not be considered a completely untouched independent test estimate.

---

### Experiment 3 — Out-of-Fold (OOF) Predictions

Implemented 5-fold Stratified Cross-Validation using CatBoost to generate out-of-fold predictions.

#### Configuration

| Parameter | Value |
|---|---|
| Model | CatBoost Classifier |
| Number of folds | 5 |
| Stratification | StratifiedKFold |
| Random seed | 42 |
| Iterations | 1500 |
| Learning rate | 0.05 |
| Depth | 6 |
| Loss function | Logloss |
| Evaluation metric | AUC |

#### Fold Results

| Fold | ROC-AUC |
|---|---:|
| Fold 1 | 0.944425 |
| Fold 2 | 0.947394 |
| Fold 3 | 0.950738 |
| Fold 4 | 0.946150 |
| Fold 5 | 0.941625 |
| **Mean** | **0.946067** |
| **Standard Deviation** | **0.003034** |

#### Overall OOF Score

**Overall OOF ROC-AUC: 0.946022**

OOF predictions were generated by ensuring that each training observation received a prediction from a model that had not trained on that observation.

This provides a more robust estimate of model performance than a single validation split.

---

### Experiment 4 — Adversarial Validation

Performed adversarial validation to investigate whether the training and test datasets have noticeably different distributions.

The approach was to distinguish training samples from test samples using a classifier.

#### Result

| Metric | Score |
|---|---:|
| Adversarial Validation ROC-AUC | **0.498377** |

#### Interpretation

An adversarial validation ROC-AUC close to 0.50 indicates that the classifier was unable to meaningfully distinguish the training data from the test data.

This suggests that the train and test datasets have similar distributions under this validation setup.

There is no strong evidence of a significant train-test distribution shift based on this experiment.

---

### Experiment 5 — Final Kaggle Submission

Generated fraud probabilities using CatBoost and prepared the submission file according to the competition's required format.

#### Submission Validation

- Correct column names: `id`, `is_fraud`
- Correct number of rows
- IDs match the sample submission
- Correct ID ordering
- No missing predictions
- Predictions are within the range `[0, 1]`
- Predictions are probabilities generated using `predict_proba`

#### Submission Result

| Metric | Result |
|---|---:|
| Public Leaderboard ROC-AUC | **0.94709** |
| Submission | Successful |
| Kaggle Status | Public score received |

My final public leaderboard score of **0.94709** is slightly higher than the overall OOF ROC-AUC of **0.946022**.

This provides a useful comparison between cross-validation performance and leaderboard performance.

---

## 📈 Performance Summary

| Experiment | Model / Method | ROC-AUC |
|---|---|---:|
| Benchmark | CatBoost | 0.944426 |
| Benchmark | LightGBM | 0.943152 |
| Benchmark | XGBoost | 0.942245 |
| Optuna | Default CatBoost CV | 0.945966 |
| Optuna | Best tuned CV | 0.941015 |
| Optuna | Holdout | 0.944562 |
| OOF | 5-Fold CatBoost | 0.946022 |
| Adversarial Validation | Train vs Test | 0.498377 |
| **Kaggle Public Leaderboard** | **Final Submission** | **0.94709** |

---

## 🧠 Key Learnings

- Gradient boosting models are strong choices for tabular classification problems.
- CatBoost provided the best score in the initial model benchmark.
- Training speed and model performance are not always directly correlated.
- Hyperparameter optimization does not guarantee improvement over default parameters.
- Consistent cross-validation methodology is essential when comparing experiments.
- Stratified cross-validation is useful for imbalanced classification problems.
- OOF predictions provide a more robust model evaluation.
- Adversarial validation can help identify train-test distribution shift.
- Correct submission formatting and validation are important parts of a Kaggle workflow.
- Public leaderboard results should be interpreted alongside cross-validation results.

---

## 📁 Project Structure

```text
Predicting_Credit_Card_Fraud/
│
|
│── Compare_Boosters.ipynb
│── Optuna_Hyperparameter_Tuning.ipynb
│── OOF_Adversarial_Submission.ipynb
│
├── README.md
├── submission.csv
└── xgboost_vs_lgb_vs_cat.png
```

*The directory structure may evolve as additional experiments and scripts are added.*

---

## 🚀 Future Improvements

- Complete a larger Optuna search with 40+ trials.
- Use 5-fold stratified CV consistently for hyperparameter tuning.
- Tune CatBoost using the full development dataset.
- Compare tuned CatBoost against LightGBM and XGBoost.
- Explore feature engineering and feature interactions.
- Investigate class imbalance strategies.
- Experiment with ensemble models.
- Blend OOF predictions from multiple boosting models.
- Analyze feature importance and model explainability.
- Improve reproducibility by organizing experiments and configurations.
- Track public leaderboard performance across submissions.

---

## 👨‍💻 Author

**Jeevan Devang**

- Kaggle: [jeevandevang](https://www.kaggle.com/jeevandevang)
- GitHub: [Jeevan-33](https://github.com/Jeevan-33)

---

## ⭐ Acknowledgements

Thanks to Kaggle for providing the Playground Series competition and dataset.

This project is part of my machine learning learning journey, focusing on hands-on practice, experimentation, model validation, and improving my understanding of real-world tabular machine learning workflows.
