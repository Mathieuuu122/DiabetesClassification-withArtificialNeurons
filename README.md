# DiabetesClassification-withArtificialNeurons


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
