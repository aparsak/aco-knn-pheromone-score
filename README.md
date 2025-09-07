# ACO-kNN Pheromone-Score for Imbalanced Classification

## 📌 Introduction
This repository contains experiments on a novel **Pheromone-Score (PS)** feature, computed via **Ant Colony Optimization (ACO)** walks on **k-Nearest Neighbor (kNN)** graphs.  
The goal is to improve **minority-class detection** in highly imbalanced datasets without generating synthetic data or removing samples.  

The method:
- Builds a kNN graph on training data.  
- Lets virtual ants walk from majority nodes toward minority nodes, depositing pheromone along successful paths.  
- Normalizes pheromone levels to produce the **PS feature** for each sample.  
- Uses PS in two ways:  
  - As an **additional feature** for classifiers.  
  - As a **guiding weight/resampling factor** for training.  

The repository demonstrates this approach on two benchmark datasets.

---

## 📘 Breast Cancer — `bc_ps-aco-knn_experiment.ipynb`

### Overview
- **Dataset:** Breast Cancer Wisconsin (malignant = minority, benign = majority).  
- **Focus:** Demonstrate PS on a relatively small, biomedical dataset.  

### Steps
1. Load and standardize the dataset.  
2. Compute PS on the training graph (no leakage).  
3. Evaluate Random Forests with and without PS.  

### Metrics
- Minority F1, Precision, Recall  
- AUC-PR, ROC-AUC  
- Accuracy, Macro-F1  

### Purpose
Show how PS improves recognition of malignant cases and provides interpretability in a controlled dataset.

---

## 📘 Credit Card Fraud — `creditcard_ps-aco-knn_experiment.ipynb`

### Overview
- **Dataset:** Credit Card Fraud (fraud = minority, normal = majority).  
- **Challenge:** Extremely imbalanced and large-scale.  

### Strategies
1. Baseline RF (no PS)  
2. RF + PS as feature with sample weighting  
3. RF + PS-guided resampling  

### Design
- Scalable “light mode” ACO with capped majority per fold.  
- Resampling probability proportional to `(α + β·PS)` to emphasize hard minority cases.  

### Metrics
- F1, Precision, Recall for minority  
- AUC-PR, ROC-AUC, Accuracy  

### Purpose
Show how PS can enhance fraud detection and guide resampling in real-world large datasets.

> ⚠️ **Note:** The Credit Card dataset is not included in this repository.  
> Please download it manually from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in:  
> ```
> data/creditcard.csv
> ```

---

## 🔧 Requirements

- Python ≥ 3.9  
- Dependencies are listed in [`requirements.txt`](requirements.txt).  

---
