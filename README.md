# CSCI 4734 Assignment 1

This repository contains my work for Assignment 1 in CSCI 4734. The work is
implemented in [`CSCI4734_2026F_A1.ipynb`](./CSCI4734_2026F_A1.ipynb) and uses
Python, NumPy, pandas, Matplotlib, and scikit-learn.

## Contents

| File | Description |
| --- | --- |
| [`CSCI4734_2026F_A1.ipynb`](./CSCI4734_2026F_A1.ipynb) | Notebook containing all preprocessing, visualizations, model implementations, training, evaluation, and predictions |
| [`turboaz.csv`](./turboaz.csv) | Azerbaijani car listings used for the linear-regression task |
| [`exams.csv`](./exams.csv) | Exam scores and admission labels used for the logistic-regression task |
| [`CSCI4734_2026F_A1.pdf`](./CSCI4734_2026F_A1.pdf) | PDF version of the assignment notebook |

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook or JupyterLab

The notebook imports the following main classes and functions:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
```

## Running the notebook

1. Clone or download this repository.
2. Install the required packages, for example:

   ```bash
   python -m pip install numpy pandas matplotlib scikit-learn jupyter
   ```

3. Start Jupyter from the repository directory:

   ```bash
   jupyter notebook
   ```

4. Open `CSCI4734_2026F_A1.ipynb` and run the cells from top to bottom.

The CSV files are loaded using relative paths, so the notebook should be run
with the repository as the working directory.

## Part 1: Linear regression

### Goal

Predict a car's price from:

- `Yurush` — mileage
- `Buraxilish ili` — manufacturing year
- `Qiymet` — listing price

### Data preparation

The notebook:

1. Loads `turboaz.csv` with pandas.
2. Removes spaces and the `km` suffix from mileage values and converts them
   to floating-point numbers.
3. Removes the `AZN` and `$` price suffixes.
4. Converts dollar-denominated prices to AZN using a factor of `1.7`.
5. Creates scatter plots for mileage versus price, year versus price, and a
   three-dimensional plot containing mileage, year, and price.
6. Keeps copies of the original-scale data for evaluation and prediction.
7. Standardizes mileage, year, and price with z-score normalization for the
   from-scratch implementation.

### From-scratch implementation

The notebook implements:

- A linear hypothesis function, `h_linear(X, T) = X @ T`
- The mean squared cost function
- A bias column and parameter vector initialized to zero
- Batch gradient descent with:
  - Learning rate: `0.001`
  - Iterations: `10,000`
- A cost-versus-iteration plot
- Two-dimensional and three-dimensional prediction visualizations

The cost decreases from approximately `0.4996` at iteration 0 to
approximately `0.1062` at iteration 9,000.

### Results

The from-scratch model achieved:

- Mean squared error: `39,560,353.49`
- R² score: `0.787531`

Example price predictions from the from-scratch model:

| Mileage | Year | Predicted price |
| ---: | ---: | ---: |
| 240,000 km | 2000 | `15,841.44` |
| 415,558 km | 1996 | `5,426.07` |

### scikit-learn implementation

The notebook also trains `sklearn.linear_model.LinearRegression` using the
original, unstandardized mileage and year features. Its results are:

- Mean squared error: `39,556,044.58`
- R² score: `0.787554`

The library model predicts approximately:

- `15,820.54` for a car with 240,000 km from 2000
- `5,453.69` for a car with 415,558 km from 1996

### Normal equation

The normal-equation solution is also calculated on the standardized data:

```text
T = (XᵀX)⁻¹XᵀY
```

The resulting parameters are approximately:

```text
[ 1.69603914e-14, -9.33437483e-02, 8.30867464e-01 ]
```

These values are close to the parameters learned by gradient descent:

```text
[ 1.67602282e-14, -9.83868080e-02, 8.25824265e-01 ]
```

## Part 2: Logistic regression

### Goal

Predict whether a student is admitted using:

- `exam_1` — first exam score
- `exam_2` — second exam score
- `admitted` — binary admission label

### Data preparation

The notebook:

1. Loads `exams.csv` with pandas.
2. Separates the two exam scores from the admission labels.
3. Keeps copies of the original scores for later predictions.
4. Applies min-max scaling to both exam-score features.
5. Plots admitted and non-admitted students using different colors.

### From-scratch implementation

The notebook implements:

- The sigmoid hypothesis function
- Binary cross-entropy cost
- A bias column and zero-initialized parameter vector
- Batch gradient descent with:
  - Learning rate: `0.1`
  - Iterations: `10,000`
- A cost-versus-iteration plot
- A decision-boundary plot
- A classification threshold of `0.5`

The cost decreases from approximately `0.6931` at iteration 0 to
approximately `0.2237` at iteration 9,000.

The from-scratch model achieves an accuracy of `0.89`.

Example probabilities:

| Exam 1 | Exam 2 | Predicted admission probability |
| ---: | ---: | ---: |
| 55 | 70 | `0.553489` |
| 40 | 60 | `0.043984` |

Using a threshold of `0.5`, the first example is classified as admitted and
the second as not admitted.

### scikit-learn implementation

The notebook also trains `sklearn.linear_model.LogisticRegression` on the
original exam scores. This implementation also achieves an accuracy of
`0.89`.

The library model returns the following class probabilities for the same two
test students:

```text
[[0.42967107, 0.57032893],
 [0.99186143, 0.00813857]]
```

Each row is ordered as `[probability of not admitted, probability of
admitted]`.

## Summary

This assignment compares manually implemented optimization with
scikit-learn implementations:

- Linear regression was used for continuous car-price prediction.
- Logistic regression was used for binary student-admission prediction.
- Gradient descent, cost functions, normalization, plotting, model evaluation,
  and example predictions were completed in the notebook.
- The manually implemented and library models produced very similar evaluation
  results for both tasks.
