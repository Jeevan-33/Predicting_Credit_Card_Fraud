🚀 Predicting Credit Card Fraud — Machine Learning Experiments
A hands-on machine learning project using the Kaggle Playground Series credit card fraud dataset. This repository explores gradient boosting, model benchmarking, hyperparameter optimization, cross-validation, adversarial validation, and model evaluation.
🎯 Project Objectives
- Compare popular gradient boosting algorithms.
- Evaluate model performance using ROC-AUC.
- Understand training time and early stopping.
- Optimize CatBoost hyperparameters using Optuna.
- Measure differences between cross-validation and held-out test performance.
- Understand model selection bias and limitations of validation estimates.
- Generate out-of-fold (OOF) predictions.
- Investigate train-test distribution differences using adversarial validation.
- Prepare and validate Kaggle submissions.
📊 Dataset
- Competition: Kaggle Playground Series — Season 6, Episode 11
- Task: Binary Classification
- Target: is_fraud
- Evaluation Metric: ROC-AUC
- ID Column: id
Kaggle Competition — Predicting Credit Card Fraud

The benchmark notebook used a smaller training-file version with 157,030 rows, while the OOF and Optuna notebooks used a training file with 699,960 rows. Results across these experiments should therefore not be treated as perfectly directly comparable.
🛠️ Tech Stack
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
Experiment 1: Gradient Boosting Benchmark
Compared three gradient boosting algorithms using an 80/20 stratified train-validation split.

Dataset used in this notebook: 157,030 rows in the loaded training file.
Models
- CatBoost
- LightGBM
- XGBoost
Benchmark Results
Rank	Model	Validation ROC-AUC	Best Iteration	Training Time (sec)
🥇 1	CatBoost	0.944426	440	24.14
🥈 2	LightGBM	0.943152	169	7.99
🥉 3	XGBoost	0.942245	148	15.65


Key Findings
- CatBoost achieved the highest validation ROC-AUC of 0.944426.
- LightGBM was the fastest, completing training in approximately 7.99 seconds.
- The AUC difference between CatBoost and LightGBM was 0.001274.
- The AUC difference between CatBoost and XGBoost was 0.002181.
- These results come from one validation split; further validation would be needed to establish whether the ranking is stable.
Experiment 2: CatBoost Hyperparameter Optimization with Optuna
This experiment used a stratified 80/20 development/test split on the 699,960-row training file. The development portion contained 559,968 rows, and the holdout portion contained 139,992 rows.
Experimental Setup (as actually run)
Setting	Value
Holdout test size	20%
Development data	80%
Optuna tuning subset	20% of development data
Tuning CV	3-fold Stratified CV
Trials run	15
Optimization library	Optuna
Optimization objective	Maximize ROC-AUC
Pruning	Median Pruner
Early stopping during tuning	50 rounds
Default baseline CV	5-fold Stratified CV
Default baseline early stopping	100 rounds


Note: The notebook was initially planned for 40 trials and 5-fold CV, but the executed code used 15 trials and 3-fold CV on a 20% subset of the development data.
Hyperparameters Tuned
- Depth
- Learning rate
- L2 leaf regularization
- Random strength
- Bagging temperature
- Border count
Default CatBoost 5-Fold CV Results
Fold	ROC-AUC
Fold 1	0.945313
Fold 2	0.949048
Fold 3	0.948976
Fold 4	0.938589
Fold 5	0.947905
Mean	0.945966
Std (population)	0.003928


Optuna Results
Metric	ROC-AUC
Default CatBoost 5-fold CV	0.945966
Best Optuna tuning CV (3-fold subset)	0.941015
Holdout score reported by notebook	0.944562
Best CV − reported holdout	-0.003547


The Optuna search did not outperform the default baseline in this run. The best tuning score was approximately 0.004951 lower than the default 5-fold CV score. Note that these CV scores use different evaluation setups (full development data for the default baseline versus a smaller subset and 3 folds for Optuna), so they are not a perfectly controlled comparison.
Best Hyperparameters
{
    "depth": 4,
    "learning_rate": 0.1044004075,
    "l2_leaf_reg": 9.4156101644,
    "random_strength": 3.5695556312,
    "bagging_temperature": 0.1029224715,
    "border_count": 249
}
Holdout Evaluation Caveat
The notebook used the holdout test set as CatBoost's eval_set with early stopping, then reported the score on that same set. The holdout therefore influenced the selected iteration (best iteration: 899) and is not a fully untouched final evaluation.

The reported score of 0.944562 should be treated as a holdout score affected by early-stopping selection, not as an unbiased final test estimate.

The notebook also retrained a model on the development data using 899 iterations, but the reported score was generated from the model that had used the holdout set for early stopping.
Experiment 3: Out-of-Fold (OOF) Validation
Evaluated a default CatBoost classifier using 5-fold stratified cross-validation on the 699,960-row training file and generated out-of-fold predictions for all training rows.
Experimental Setup
Setting	Value
Model	CatBoost Classifier
Iterations	1,500
Learning Rate	0.05
Depth	6
Loss Function	Logloss
Evaluation Metric	AUC
Random Seed	42
Cross-validation	5-fold StratifiedKFold


Fold Results
Fold	ROC-AUC
Fold 1	0.944425
Fold 2	0.947394
Fold 3	0.950738
Fold 4	0.946150
Fold 5	0.941625
Fold Mean	0.946067
Fold Standard Deviation	0.003034


Overall OOF Score
Overall OOF ROC-AUC: 0.946022

The overall OOF ROC-AUC was calculated using combined out-of-fold predictions across all training rows. It can differ slightly from the arithmetic mean of the fold AUCs.
Key Findings
- Overall OOF ROC-AUC: 0.946022
- Five-fold mean ROC-AUC: 0.946067
- Fold standard deviation: 0.003034
- The fold scores were relatively consistent across the five validation splits.
Experiment 4: Adversarial Validation
Adversarial validation was used to investigate whether the original training dataset and Kaggle test dataset have distinguishable feature distributions.

A CatBoost classifier was trained to distinguish training rows (0) from test rows (1).
Experimental Setup
- Combined training and test feature rows.
- Training labels set to 0; test labels set to 1.
- Stratified 80/20 adversarial train-validation split.
- CatBoost classifier.
- ROC-AUC evaluation.
Results
Metric	AUC
Adversarial Validation AUC	0.498377


Interpretation
The adversarial AUC is very close to 0.50. In this experiment, the model was unable to distinguish the competition training and test rows better than chance using the selected features and setup.

This suggests there is no strong detectable train-test distribution difference under this adversarial validation experiment.
Experiment 5: Kaggle Submission Preparation
Trained a CatBoost model on the labelled training data and generated probability predictions for the competition test data.
Submission Validation
The generated submission.csv was checked against sample_submission.csv.
- Correct shape: 299,984 rows × 2 columns
- Correct columns: id, is_fraud
- ID order: verified
- Missing predictions: none
- Predictions in the valid probability range [0, 1]
Submission checker result: Passed
Submission Status
Item	Status
submission.csv generated	Yes
Submission validation	Passed
Kaggle API authentication	Failed / not configured
Submission uploaded to Kaggle	No
Public leaderboard score	Not available


The Kaggle CLI returned an authentication message asking for API credentials. No public leaderboard score was obtained in this run.
📂 Project Structure
Predicting_Credit_Card_Fraud/
│
├── Predicting_Credit_Card_Fraud.ipynb
├── Optuna_hyperparameter_tuning.ipynb
├── OOF_Adversarial_Submission.ipynb
├── xgboost_vs_lgb_vs_cat.png
├── README.md
└── .gitignore
💡 Key Learnings
- Comparing gradient boosting algorithms.
- Building stratified train-validation splits.
- Understanding early stopping.
- Using ROC-AUC for binary classification.
- Hyperparameter optimization with Optuna.
- Applying pruning to reduce unnecessary trial computation.
- Using stratified cross-validation.
- Generating out-of-fold predictions.
- Evaluating fold mean and standard deviation.
- Keeping a final test set separate from model selection.
- Understanding the difference between validation performance and final test performance.
- Recognizing potential selection bias.
- Using adversarial validation to investigate train-test distribution differences.
- Creating and validating Kaggle submission files.
- Understanding that a test set used for early stopping is no longer fully untouched.
- Troubleshooting Kaggle API authentication.
🔮 Future Improvements
- Rerun Optuna with the originally planned 40 trials and 5-fold CV.
- Tune using the full development dataset or clearly document a tuning subset.
- Use a validation set within the development data for early stopping.
- Keep the final test set untouched until the final evaluation.
- Compare tuned and default models under identical CV conditions.
- Analyze feature importance.
- Explore additional feature engineering.
- Complete Kaggle API authentication and submit one validated baseline.
- Compare the public leaderboard score with OOF performance.
- Explore other cross-validation strategies if justified.
👨‍💻 About This Project
This repository is part of my practical Machine Learning learning journey, focusing on implementing algorithms, conducting experiments, evaluating models, and developing skills through Kaggle competitions.

Learning by doing. Experimenting. Improving. 🚀
