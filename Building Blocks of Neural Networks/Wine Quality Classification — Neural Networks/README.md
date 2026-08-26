# Wine Quality Prediction: Linear Regression

## About the Project
This project is about predicting the quality of Portuguese white wine using linear regression built from scratch with NumPy, no scikit-learn.
The goal is to learn what combination of physicochemical properties (like alcohol, acidity, and density) best predicts a wine's quality score.

> Dataset from: Cortez et al., *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, Elsevier, 2009.

## Dataset
The dataset contains 4898 white wines with 11 features each:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

Target: **Quality** (integer score, roughly 3 to 9)

Dataset loaded directly from the [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv).

## What I Learned
- Loading and exploring real-world data with NumPy
- Plotting feature distributions and scatter plots to understand the data
- Calculating correlation coefficients to identify the most predictive features
- Implementing **Ordinary Least Squares (OLS)** from scratch using the closed-form formula
- Matrix operations: transposition, multiplication, and inversion with NumPy
- Train/test splitting: properly evaluating a model on unseen data
- Adding a **bias term** by appending a column of ones to the feature matrix
- Evaluating predictions using correlation and scatter plots

## The OLS Formula
Rather than using a library, weights are computed analytically:

$$\vec{w} = (X^T X)^{-1} X^T \vec{y}$$

This minimises the sum of squared errors between predicted and true quality scores.

## Results
| Model | Correlation (r) |
|---|---|
| OLS without bias | ~0.52 |
| OLS with bias term | 0.529 |

Most predictive single feature: **alcohol** (r ≈ +0.44)

The model captures a real signal in the data, but wine quality is inherently noisy. It is a human judgement score and the relationship with physicochemical properties is not perfectly linear.

## Libraries Used
- Python
- NumPy
- matplotlib
