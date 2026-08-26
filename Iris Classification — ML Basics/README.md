# Iris Classification — ML Basics

## About the Project
This project is about multiclass classification using the classic Iris flower dataset.
The goal is to classify three types of iris flowers (Setosa, Versicolor, Virginica)
based on their sepal and petal measurements.

Note: We used only 2 features (sepal length and sepal width) for visualization purposes.

## Dataset
The Iris dataset contains 150 flowers with 4 features each:
- Sepal length
- Sepal width
- Petal length
- Petal width

Dataset loaded using sklearn's built-in `load_iris()`.

## What I Learned
- Loading and exploring a dataset with pandas
- Building a design matrix (features) and target array (labels)
- Label encoding — converting text class names to numbers
- Train/test splitting — how to properly divide data for evaluation
- Logistic Regression — first classifier
- Evaluation metrics — accuracy, precision, recall, f1-score
- Visualizing decision boundaries
- K-Nearest Neighbors (KNN) — bonus classifier

## Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 78% |
| KNN (k=5) | 83% |

KNN outperformed Logistic Regression on this dataset.
Accuracy could be further improved by using all 4 features.

## Libraries Used
- Python
- NumPy
- pandas
- scikit-learn
- matplotlib
- mlxtend
