# 📘 SML: Regression and Classification Fundamentals

> **Tagline:** Implementation and visualization of core Machine Learning algorithms — Linear Regression, Gradient Descent, and Logistic Regression — built from scratch using Python and NumPy.

**GitHub About Section (suggestion):**  
Learn and implement the mathematical foundations of Supervised Machine Learning. This project demonstrates Linear and Logistic Regression, cost functions, and optimization using gradient descent — without using pre-built ML libraries.

---

## 🚀 Overview

This repository explores the **fundamental building blocks of Machine Learning** — regression and classification — implemented from first principles.  

The goal is to develop a strong intuition behind:
- **Cost functions**  
- **Gradient descent optimization**  
- **Linear regression (single & multiple variable)**  
- **Logistic regression for classification tasks**

Unlike high-level libraries (e.g., scikit-learn or TensorFlow), these implementations rely only on **NumPy**, enabling complete control and mathematical transparency.

---

## 📂 Repository Structure

```
SML_Regression-Classification/
│
├── Cost_Function.ipynb              # Visualizing and computing cost functions for regression
├── Gradient-DescentLR.ipynb         # Implementing gradient descent for linear regression
├── Multiple-VariableLR.ipynb        # Extending linear regression to multiple variables
├── Logistic_Regression.ipynb        # Binary classification using logistic regression
│
├── lab_utils_common.py              # Shared helper functions (plotting, vectorization)
├── lab_utils_uni.py                 # Single variable regression helpers
├── utils.py                         # General math utility functions
│
└── README.md                        # Documentation (this file)
```

---

## 🧠 Concepts Covered

| Topic | Description |
|--------|-------------|
| **Linear Regression** | Predicting continuous values using one or more features. |
| **Gradient Descent** | Iterative optimization algorithm minimizing cost function. |
| **Vectorization** | Performance optimization using NumPy operations instead of loops. |
| **Cost Function** | Mean Squared Error (MSE) implementation from scratch. |
| **Multiple Variable Regression** | Extending linear regression to multiple features. |
| **Logistic Regression** | Binary classification using sigmoid activation. |

---

## ⚙️ Mathematical Foundation

### 🧩 Linear Regression Cost Function:
\[
J(θ) = \frac{1}{2m} \sum_{i=1}^{m} (h_θ(x^{(i)}) - y^{(i)})^2
\]

### ⚙️ Gradient Descent Update Rule:
\[
θ_j := θ_j - α \frac{∂}{∂θ_j}J(θ)
\]

Where:
- \( α \) = learning rate  
- \( m \) = number of training examples  
- \( h_θ(x) \) = predicted value  

---

## 🧮 Example Implementation

```python
import numpy as np

# Hypothesis function
def compute_model_output(X, w, b):
    return np.dot(X, w) + b

# Cost function
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    return cost / (2 * m)

# Gradient descent
def gradient_descent(X, y, w, b, alpha, iterations):
    m = X.shape[0]
    for i in range(iterations):
        dj_dw = (1/m) * np.dot(X.T, (np.dot(X, w) + b - y))
        dj_db = (1/m) * np.sum(np.dot(X, w) + b - y)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b
```

---

## 📊 Visualizations & Insights

- **Cost Function Surface Plot:** Visualizing how the cost changes with parameters.  
- **Convergence Curve:** Shows how gradient descent minimizes cost over iterations.  
- **Decision Boundary (Logistic Regression):** Visualization of classification thresholds.  
- **Feature Scaling Effect:** Demonstrates why normalization is critical for convergence speed.

*(All visualizations are included in the Jupyter notebooks.)*

---

## 💡 Key Learnings

- Understanding of **how ML models learn parameters from data** through optimization.  
- Implementation of **Gradient Descent manually** using vectorized operations.  
- Insight into **how learning rate affects convergence**.  
- Comparison between **single-variable vs multi-variable regression**.  
- Foundation for building more advanced ML algorithms (SVMs, Neural Networks).

---

## 🧰 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Language** | Python |
| **Libraries** | NumPy, Matplotlib |
| **Environment** | Jupyter Notebook |
| **Concepts** | Linear Regression, Logistic Regression, Gradient Descent, Vectorization |

---

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Shubham91999/SML_Regression-Classification.git
   cd SML_Regression-Classification
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib jupyter
   ```

3. Launch notebooks:
   ```bash
   jupyter notebook
   ```
   Then open any `.ipynb` file (e.g., `Gradient-DescentLR.ipynb`) to explore.

---

## 📈 Example Results

| Task | Model | Key Metric | Observation |
|------|--------|-------------|--------------|
| Linear Regression (1 var) | Custom implementation | Converges to global minimum | Verified against analytical solution |
| Linear Regression (multi var) | Gradient Descent | Low MSE after 1500 iterations | Shows benefit of feature scaling |
| Logistic Regression | Sigmoid + Gradient Descent | >90% accuracy on toy dataset | Demonstrates binary classification learning |

---

## 🧠 Ideal Use Case

This repository is perfect for:
- **Interview preparation** (understanding ML math fundamentals).  
- **Academic learning** (visualizing cost and convergence).  
- **Teaching / demos** (conceptual clarity on regression and optimization).  

It’s not meant for large-scale datasets or production ML — it’s an **educational deep dive** into algorithmic mechanics.

---

## 🧑‍🚀 Author

**Shubham Kulkarni**  
Machine Learning Engineer | Data Science & AI Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/shubham91999) • [GitHub](https://github.com/Shubham91999)

---

## 🪙 License

This project is released under the **MIT License** — you’re free to use it for learning or research.

---

⭐ *If you find this project insightful, consider giving it a star!* 🌟
