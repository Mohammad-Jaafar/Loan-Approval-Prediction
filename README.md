# Loan Approval Prediction

This project builds a **machine learning model** to predict whether a
loan application will be approved based on applicant and financial
features.\
It demonstrates how **binary classification**, **data preprocessing**,
and **imbalanced data handling** techniques are applied to real-world
loan datasets.

------------------------------------------------------------------------

## Overview

The goal of this project is to:

-   Load and preprocess a loan approval dataset.
-   Handle missing values and encode categorical attributes.
-   Address **class imbalance** using **SMOTE**.
-   Train two classification models:
    -   **Logistic Regression**
    -   **Decision Tree Classifier**
-   Evaluate performance using **precision, recall, F1-score**, and
    **confusion matrix**.

------------------------------------------------------------------------

## Features

-   Loading and cleaning the dataset
-   Encoding categorical features using **Label Encoding**
-   Managing dataset imbalance with **SMOTE**
-   Scaling numeric features with **StandardScaler**
-   Training two machine-learning models
-   Model evaluation using standard classification metrics

------------------------------------------------------------------------

## Technologies Used

-   **Python 3.9+**
-   **Pandas**
-   **NumPy**
-   **Scikit-learn**
-   **Imbalanced-learn (SMOTE)**

------------------------------------------------------------------------

## Project Structure

    Loan-Approval-Prediction/
    │
    ├── loan_approval_prediction.ipynb
    ├── loan_approval_dataset.csv
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mohammad-Jaafar/Loan-Approval-Prediction.git
   ```
2. Open the notebook in Jupyter or Google Colab.
3. Run all cells step-by-step to reproduce the results.

------------------------------------------------------------------------

## Results & Evaluation

### Logistic Regression

                  precision    recall  f1-score   support

               0       0.94      0.91      0.92       536
               1       0.85      0.90      0.88       318

        accuracy                           0.91       854
       macro avg       0.90      0.90      0.90       854
    weighted avg       0.91      0.91      0.91       854

------------------------------------------------------------------------

### Decision Tree

                  precision    recall  f1-score   support

               0       0.98      0.98      0.98       536
               1       0.97      0.97      0.97       318

        accuracy                           0.98       854
       macro avg       0.97      0.98      0.97       854
    weighted avg       0.98      0.98      0.98       854

------------------------------------------------------------------------

## Author

**Mohammad Jaafar**\
mhdjaafar24@gmail.com\
[LinkedIn](https://www.linkedin.com/in/mohammad-jaafar-)\
[HuggingFace](https://huggingface.co/Mhdjaafar)\
[GitHub](https://github.com/Mohammad-Jaafar)

------------------------------------------------------------------------

*If you find this project helpful, please consider giving it a star on
GitHub!*
