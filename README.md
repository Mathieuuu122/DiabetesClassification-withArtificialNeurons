


# Diabetes Classification with Artificial Neurons
**From a Single Neuron to a PyTorch Neural Network**

## 📌 Project Overview
This project is an educational machine learning exercise designed to build, train, and compare transparent, reproducible binary classifiers on historical tabular records[cite: 1, 3]. The main objective is to evaluate whether a small multilayer perceptron (MLP) outperforms a single artificial neuron when using identical input features, data partitions, and evaluation criteria[cite: 3].

## ⚙️ Tech Stack & Platform
* **Platform:** Google Colab[cite: 1]
* **Framework:** PyTorch[cite: 1]
* **Environment:** Python (Pandas, Scikit-Learn, Matplotlib/Seaborn)

## 📊 Dataset
* **Name:** Pima Indians Diabetes Database[cite: 3]
* **Source:** Kaggle / UCI Machine Learning Repository[cite: 3]
* **Features (8):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age[cite: 4]
* **Target:** Outcome ($0 = \text{negative}, 1 = \text{positive}$)[cite: 4]

## 🚀 Project Roadmap & Structure
1. **Data Audit & Preprocessing (Week 1):** Inspecting the dataset, visualizing label distribution and feature histograms, handling implausible zero values via leakage-free median imputation, and creating stratified train/validation/test splits (60%/20%/20%)[cite: 4, 5, 6].
2. **Single Neuron / M1 (Week 2):** Exploring artificial neuron fundamentals (weights, biases, logits, and sigmoid activations) and implementing a single linear neuron using PyTorch (`nn.Linear(8, 1)`)[cite: 4, 5, 6].
3. **Multilayer Neural Network / M2 (Week 3):** Building a small multilayer perceptron architecture ($8 \rightarrow 8 \rightarrow 1$) with a ReLU hidden layer and training it with BCEWithLogitsLoss[cite: 5, 6].
4. **Evaluation & Documentation (Week 4):** Comparing models against a baseline ($M0$) using accuracy, precision, recall, F1-score, and confusion matrices[cite: 5, 6].

## ⚠️ Important Safety Scope
*This is an educational exercise involving historical data, not a diagnostic, screening, or treatment application. Predictions must not be used for individual medical decisions[cite: 1].*


# Pima Indians Diabetes ML Project - Task Distribution

This document outlines the collaborative work division for our machine learning project across all four development phases. Both team members have balanced technical responsibilities in every phase.

---

## Phase 1: Data Audit, Cleaning & Preprocessing

### Member 1
* **Data Loading & EDA:** Load the dataset, check dimensions (768 rows, 9 columns), and run summary statistics (`df.describe()`).
* **Feature Scaling:** Implement feature normalization using Scikit-Learn's `StandardScaler` to standardize numerical inputs.
* **Visualizations:** Create distribution plots or feature correlation heatmaps using Matplotlib/Seaborn.

### Member 2
* **Handling Zero-Values:** Identify biologically impossible zero-values (Glucose, Blood Pressure, BMI, etc.) and write the code for median imputation.
* **Class Distribution Check:** Analyze target balance (`Outcome` column) and check for class imbalance issues.
* **Data Splitting:** Write the stratification code to split the dataset into Train (60%), Validation (20%), and Test (20%) sets.

---

## Phase 2: Baseline Model ($M0$) & Single Neuron ($M1$)

### Member 1
* **Baseline Model ($M0$):** Train a Scikit-learn classifier (like Logistic Regression) on the training set.
* **Baseline Evaluation:** Evaluate $M0$ on the validation set and compute baseline accuracy and confusion matrix.

### Member 2
* **PyTorch Setup:** Convert DataFrames into PyTorch Tensors and instantiate `TensorDataset` and `DataLoader`.
* **Single Neuron Architecture ($M1$):** Build the single-layer linear model using PyTorch (`nn.Linear` and `nn.Sigmoid`).
* **Training & Loss Loop ($M1$):** Code the training loop with Binary Cross-Entropy loss (`BCELoss`) and an optimizer (Adam/SGD) to train the neuron.

---

## Phase 3: Multi-Layer Perceptron ($M2$) & Optimization

### Member 1
* **MLP Architecture ($M2$):** Build the multi-layer neural network with hidden layers and non-linear activation functions (e.g., ReLU).
* **Forward/Backward Passes:** Set up the forward propagation and backpropagation pipeline for $M2$.

### Member 2
* **Hyperparameter Tuning:** Test different learning rates, batch sizes, and hidden unit dimensions.
* **Regularization:** Implement dropout layers or weight decay to prevent overfitting.
* **Model Comparison Check:** Run validation evaluations to compare $M0$, $M1$, and $M2$ side-by-side.

---

## Phase 4: Final Evaluation, Comparison & Documentation

### Member 1
* **Test Set Evaluation:** Run the final unseen test set through the trained models ($M0$, $M1$, and $M2$).
* **Core Metrics Calculation:** Compute final Precision, Recall, F1-Score, and Accuracy matrices.

### Member 2
* **Comparative Visualizations:** Generate loss curves, ROC/AUC curves, and performance comparison bar charts.
* **Repository & Code Polish:** Write the final model summary and usage instructions for the GitHub repository `README.md`.
