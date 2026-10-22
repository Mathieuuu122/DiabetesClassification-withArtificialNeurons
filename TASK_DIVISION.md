# Pima Indians Diabetes ML Project - Task Distribution

This document outlines the collaborative work division for our machine learning project across all four development phases. Both team members have balanced technical responsibilities in every phase.

---

## Phase 1: Data Audit, Cleaning & Preprocessing

### Member 1
* **Data Loading & EDA:** Load the dataset, check dimensions (768 rows, 9 columns), and run summary statistics (`df.describe()`).
* **Feature Scaling:** Implement feature normalization using Scikit-Learn's `StandardScaler` to standardize numerical inputs.
* **Visualizations:** Create distribution plots or feature correlation heatmaps using Matplotlib/Seaborn.

### KUIATE MESSADO MATHIEU
* **Handling Zero-Values:** Identify biologically impossible zero-values (Glucose, Blood Pressure, BMI, etc.) and write the code for median imputation.
* **Class Distribution Check:** Analyze target balance (`Outcome` column) and check for class imbalance issues.
* **Data Splitting:** Write the stratification code to split the dataset into Train (60%), Validation (20%), and Test (20%) sets.

---

## Phase 2: Baseline Model ($M0$) & Single Neuron ($M1$)

### Member 1
* **Baseline Model ($M0$):** Train a Scikit-learn classifier (like Logistic Regression) on the training set.
* **Baseline Evaluation:** Evaluate $M0$ on the validation set and compute baseline accuracy and confusion matrix.

### KUIATE MESSADO MATHIEU
* **PyTorch Setup:** Convert DataFrames into PyTorch Tensors and instantiate `TensorDataset` and `DataLoader`.
* **Single Neuron Architecture ($M1$):** Build the single-layer linear model using PyTorch (`nn.Linear` and `nn.Sigmoid`).
* **Training & Loss Loop ($M1$):** Code the training loop with Binary Cross-Entropy loss (`BCELoss`) and an optimizer (Adam/SGD) to train the neuron.

---

## Phase 3: Multi-Layer Perceptron ($M2$) & Optimization

### Member 1
* **MLP Architecture ($M2$):** Build the multi-layer neural network with hidden layers and non-linear activation functions (e.g., ReLU).
* **Forward/Backward Passes:** Set up the forward propagation and backpropagation pipeline for $M2$.

### KUIATE MESSADO MATHIEU
* **Hyperparameter Tuning:** Test different learning rates, batch sizes, and hidden unit dimensions.
* **Regularization:** Implement dropout layers or weight decay to prevent overfitting.
* **Model Comparison Check:** Run validation evaluations to compare $M0$, $M1$, and $M2$ side-by-side.

---

## Phase 4: Final Evaluation, Comparison & Documentation

### Member 1
* **Test Set Evaluation:** Run the final unseen test set through the trained models ($M0$, $M1$, and $M2$).
* **Core Metrics Calculation:** Compute final Precision, Recall, F1-Score, and Accuracy matrices.

### KUIATE MESSADO MATHIEU
* **Comparative Visualizations:** Generate loss curves, ROC/AUC curves, and performance comparison bar charts.
* **Repository & Code Polish:** Write the final model summary and usage instructions for the GitHub repository `README.md`.
