# Machine Learning: Sampling, Cross-Validation and Model Evaluation

A hands-on exploration of fundamental machine learning concepts using Python and Scikit-learn, with a focus on class imbalance, sampling strategies, cross-validation, and model evaluation.

## Project Overview

In classification problems, especially those involving medical or biological data, the distribution of classes can be imbalanced. A model may achieve high accuracy while performing poorly at identifying the minority class.

This project explores how sampling strategies and proper validation techniques help us understand and evaluate classification models more effectively.

A synthetic dataset representing healthy and disease-positive patients was generated for this experiment.

## Dataset

The dataset was generated using Scikit-learn's `make_classification()` function.

- **Total samples:** 2,000
- **Features:** 5 synthetic biomarker measurements
- **Class 0:** Healthy (1,596 samples, approximately 80%)
- **Class 1:** Disease-positive (404 samples, approximately 20%)

The dataset was intentionally generated with an 80:20 class distribution to study the effects of class imbalance.

## What I Explored

### 1. Random Sampling vs. Stratified Sampling

Compared random train-test splitting with stratified splitting to understand how each approach distributes class proportions across training and testing datasets.

Used `train_test_split()` with the `stratify` parameter to preserve the original class distribution.

### 2. Stratified K-Fold Cross-Validation

Implemented 5-fold Stratified Cross-Validation using `StratifiedKFold`.

This ensures that each fold maintains approximately the same class distribution as the original dataset.

### 3. Feature Scaling and Scikit-learn Pipelines

- Applied `StandardScaler` to standardize the features.
- Used a Scikit-learn Pipeline to combine scaling and Logistic Regression.
- Ensured that scaling was fitted independently within each training fold, helping prevent data leakage during cross-validation.

### 4. Model Evaluation

Trained a Logistic Regression classifier and evaluated its performance using:

- Confusion Matrix
- Precision
- Recall
- F1-score
- Accuracy

## Results and Observations

The Logistic Regression model achieved approximately **82% accuracy** on the test set.

However, its performance differed significantly between the two classes:

| Metric | Healthy (Class 0) | Disease (Class 1) |
|---|---:|---:|
| Precision | 83% | 69% |
| Recall | 97% | 22% |
| F1-score | 90% | 34% |

The model identified most healthy cases but struggled to identify disease-positive cases.

This demonstrates why accuracy alone can be misleading when evaluating models trained on imbalanced datasets.

The mean F1-score across the 5 cross-validation folds was approximately **0.273**, with a standard deviation of approximately **0.055**.

## Key Learning Outcomes

- Understood the difference between random and stratified sampling.
- Learned how class imbalance affects classification performance.
- Explored Stratified K-Fold Cross-Validation.
- Understood the importance of precision, recall, F1-score, and confusion matrices.
- Learned why feature scaling must be performed correctly during cross-validation.
- Understood how Scikit-learn Pipelines help prevent data leakage.
- Recognized why high accuracy does not necessarily indicate good minority-class detection.

## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Repository Contents

- `Class_Imbalance,_Stratified_Sampling_&_Cross_Validation.ipynb` — Notebook containing the experiments, implementations, visualizations, and model evaluation.

## Note

The dataset used in this project is synthetically generated for educational purposes. It does not contain actual patient data, and the model is not intended for clinical use.

## Conclusion

This project helped me understand that building a classification model is not just about achieving high accuracy. Choosing appropriate sampling strategies, validating models correctly, preventing data leakage, and interpreting evaluation metrics are equally important parts of the machine learning workflow.
