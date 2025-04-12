# Early Detection of Depression and Anxiety from Text

## Problem Description:
This project uses Natural Language Processing (NLP) to detect early signs of depression and anxiety from text data. The system classifies text as either "Depressed," "Anxious," or "Healthy."

## How to Use:
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app with: `streamlit run src/app.py`
4. Enter a text sample into the app for analysis.

## Dataset:
- Reddit Mental Health Dataset (available from Kaggle)

## Model Performance:
We evaluated three traditional machine learning classifiers:

| Model               | Accuracy | F1 Score | Training Time (s) |
|---------------------|----------|----------|-------------------|
| Logistic Regression | 0.669    | 0.668    | 5.68              |
| SVM                 | 0.724    | 0.726    | 1109.67           |
| Random Forest       | 0.704    | 0.704    | 248.30            |

## Key Findings:
- SVM achieved the highest accuracy (72.4%) and F1 score (72.6%) but with significantly longer training time
- Logistic Regression had the fastest training time (<6 seconds) while maintaining reasonable accuracy
- Random Forest provided a good balance between performance and training time

## Evaluation Metrics:
- Accuracy: Overall correctness of the model
- Precision: Measure of exactness (quality)
- Recall: Measure of completeness (quantity)
- F1-Score: Harmonic mean of precision and recall
