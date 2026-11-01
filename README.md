# Customer Churn Prediction

Predicts whether a telecom customer will churn, using classical machine learning.

## Dataset
Telco Customer Churn (Kaggle): 7,043 customers, 20 features.
Target: Churn (Yes/No). About 26% of customers churn, so the data is imbalanced.

## Approach
1. Cleaned the data (fixed blank TotalCharges values, dropped customerID)
2. EDA: histograms, box plots, correlation heatmap, count plots
3. Label-encoded categorical features and saved the encoders
4. Stratified 80/20 train/test split; SMOTE applied on the training data only
5. Trained and compared Decision Tree, Random Forest and Logistic Regression
6. Evaluated on the held-out test set with precision, recall, F1 and confusion matrix

## Results (test set, churn class)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree (max_depth=5) | 0.769 | 0.55 | 0.65 | 0.60 |
| Random Forest | 0.779 | 0.58 | 0.58 | 0.58 |
| Logistic Regression | 0.752 | 0.52 | 0.72 | 0.60 |

**Best model: Logistic Regression.** It has the highest churn recall (0.72).
Recall was prioritised because missing a customer who churns costs more
than a false alarm. Random Forest has the best accuracy, but accuracy is
misleading on imbalanced data.

## Files
- `churn_prediction.ipynb`: full notebook (EDA, preprocessing, models)
- `best_model.pkl`: trained Logistic Regression pipeline (with scaler)
- `encoders.pkl`: label encoders for categorical columns
- `feature_names.pkl`: column order used in training

## Tech
Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn

## How to run
1. Clone the repo and install: `pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn`
2. Open `churn_prediction.ipynb` in Jupyter or Google Colab and run all cells.

## Next steps
- PyTorch neural network and comparison with the baseline
- FastAPI endpoint for real-time predictions
