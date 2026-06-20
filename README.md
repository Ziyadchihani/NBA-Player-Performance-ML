# NBA Player Performance Analysis: A Multi-Task Machine Learning Study

An end-to-end machine learning project applying supervised and unsupervised learning techniques to analyze NBA player statistics from the 2024-2025 season. The project evaluates player scoring, uncovers style-of-play archetypes, and predicts game outcomes.

## 📊 Project Overview & Dataset
* **Core Goal:** Extract actionable basketball insights using data-driven approaches across regression, clustering, and classification tasks.
* **Dataset:** Game-by-game player statistics utilizing the [Kaggle NBA Player Stats & Secrets (2024-2025 Season)](https://www.kaggle.com/code/devraai/nba-player-stats-secrets-20242025-season/notebook).
* **Frameworks Used:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.

---

## 📈 1. Supervised Learning: Regression (Predicting Player PTS)

The objective of this task is to accurately predict a player's scoring output (`PTS`) based on active features like minutes played (`MP`), field goal attempts (`FGA`), and usage metrics.

### 📊 Regression Performance Metrics
Below is the evaluation breakdown of our regression models measured by Mean Squared Error (MSE) and R-Squared ($R^2$) Score on the Test Set:

| Regression Model | Mean Squared Error (MSE) | R² Score (Variance Explained) |
| :--- | :---: | :---: |
| 🌲 **Random Forest Regressor** | **12.45** | **0.862 (86.2%)** |
| 📈 Linear Regression | 15.30 | 0.814 (81.4%) |
| 🧠 Multi-Layer Perceptron (MLP) | 16.85 | 0.795 (79.5%) |
| 🛡️ Support Vector Regressor (SVR) | 22.10 | 0.710 (71.0%) |

* **Insight:** The **Random Forest Regressor** outperformed linear baselines, proving that individual scoring dependencies in modern basketball are non-linear and rely heavily on interaction fields.

---

## 🧬 2. Unsupervised Learning: Clustering (Discovering Player Archetypes)

Instead of using traditional positions (Guard, Forward, Center), we applied unsupervised algorithms on normalized features to discover natural player clusters.

### 📊 Clustering Evaluation
We used Principal Component Analysis (PCA) to reduce dimensionality and validated the clusters using the Silhouette Coefficient:

* **Optimal Clusters (K):** 3 distinct player archetypes.
* **Silhouette Score:** `0.42` (Indicating solid structural separation).
* **Agglomerative Clustering:** Supported hierarchical grouping, cross-verified with K-Means.

### 📉 Visual Clustering Results
Below is the PCA visualization displaying the discovered natural player archetypes within the dataset:

![NBA Clusters Visualization](assets/pca_cluster_visualization_named.png)

---

## 🛡️ 3. Supervised Learning: Classification (Predicting Game Win/Loss)

We trained and evaluated 4 classifiers to determine team game outcomes based on overall in-game player impacts.

### 📊 Classification Results Comparison (Test Set)

| Classification Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 🌲 **Random Forest Classifier** | **71.2%** | **70.6%** | **71.2%** | **70.4%** |
| 📈 Logistic Regression | 69.5% | 69.0% | 69.5% | 69.1% |
| 🛡️ Support Vector Machine (SVM) | 68.7% | 68.1% | 68.7% | 68.1% |
| 🌿 Decision Tree Classifier | 62.3% | 62.4% | 62.3% | 62.3% |

### 📸 Confusion Matrices Visualizations
The confusion matrices below illustrate the true versus predicted labels for our game outcome classification models on the Test Set:

| Model | Confusion Matrix Image |
|---|---|
| **Random Forest** | ![Random Forest](assets/Random%20Forest.png) |
| **SVM** | ![SVM](assets/SVM.png) |
| **Decision Tree** | ![Decision Tree](assets/Decsion%20Tree.png) |
| **Logistic Regression** | ![Logistic Regression](assets/Logistic%20Regression.png) |

---

## 🎯 Key Project Insights
1. **Context Over Raw Stats:** Regression algorithms highly depend on situational factors like `Minutes Played` which heavily skew individual baseline point predictions.
2. **Modern Positionless Basketball:** Clustering confirmed that traditional positions are outdated; modern NBA players are best grouped into functional roles like "High-Volume offensive drivers" vs. "High-efficiency paint protectors."
