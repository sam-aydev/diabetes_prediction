# 🧠 Machine Learning Model Comparison — Supervised Classification

This project demonstrates the **training, tuning, and evaluation** of multiple supervised machine learning algorithms for a **binary classification task**.  
The goal was to identify the model that provides the **best predictive performance** based on **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 📘 Overview

Supervised learning involves training models on **labeled data** to predict outcomes for unseen data.  
This project focuses on **classification** — predicting **discrete labels** such as “fraud / not fraud”, “yes / no”, or “disease / no disease”.

### 🔹 Models Trained & Tuned

The following algorithms were trained and hyperparameter-tuned using grid/random search:

| # | Algorithm | Type |
|---|------------|------|
| 1 | Logistic Regression | Linear Classifier |
| 2 | Random Forest | Ensemble (Bagging) |
| 3 | Gradient Boosting | Ensemble (Boosting) |
| 4 | AdaBoost | Ensemble (Boosting) |
| 5 | Extra Trees | Ensemble (Bagging) |
| 6 | Decision Tree | Non-linear |
| 7 | K-Nearest Neighbors (KNN) | Distance-based |
| 8 | Naive Bayes | Probabilistic |
| 9 | Support Vector Machine (SVM) | Margin-based |
| 10 | XGBoost | Gradient Boosting |
| 11 | LightGBM | Gradient Boosting (Optimized for Speed) |

---

## ⚙️ Training Logs (Highlights)

- ✅ Training and tuning completed for all models.
- ⚠️ Some expected warnings (e.g., `use_label_encoder` in XGBoost, `No further splits` in LightGBM).
- 💡 These warnings do **not** affect performance — they are related to internal model optimizations. `because I did not convert numpy array to dataframe, which i later did but have not added the newer model to app.py and github`

---

## 🏆 Model Performance Comparison

| Rank | Model | Accuracy | Precision | Recall | F1 Score |
|------|--------|-----------|------------|---------|-----------|
| 🥇 | **XGBoost** | **0.9799** | **0.9818** | **0.6528** | **0.7842** |
| 2 | Random Forest | 0.9800 | 0.9938 | 0.6460 | 0.7830 |
| 3 | Gradient Boosting | 0.9799 | 0.9938 | 0.6433 | 0.7810 |
| 4 | Decision Tree | 0.9798 | 1.0000 | 0.6380 | 0.7790 |
| 5 | AdaBoost | 0.9798 | 1.0000 | 0.6380 | 0.7790 |
| 6 | LightGBM | 0.9796 | 0.9876 | 0.6420 | 0.7781 |
| 7 | Extra Trees | 0.9793 | 0.9814 | 0.6406 | 0.7752 |
| 8 | SVM | 0.9759 | 0.9608 | 0.5935 | 0.7338 |
| 9 | KNN | 0.9744 | 0.9296 | 0.5868 | 0.7195 |
| 10 | Logistic Regression | 0.9723 | 0.8834 | 0.5814 | 0.7013 |
| 11 | Naive Bayes | 0.9272 | 0.3917 | 0.5478 | 0.4568 |

---

## ✅ Best Model

**🏆 Model:** `XGBoostClassifier` - because of it F1
 
**💾 Saved Files:**
- `best_model.pkl` — Serialized best model
- `scaler.pkl` — StandardScaler used for feature scaling

You can easily load and use the model for predictions:

```python
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

`Retraining it might result in Random Forest Classifier being the best`

Here’s a complete, polished **`README.md`** file for your machine learning project — formatted for GitHub and ready to commit 🚀


## 📈 Future Improvements

* Add cross-validation for more robust performance metrics.
* Explore **deep learning models** (e.g., Neural Networks).
* Perform **feature selection** and **SMOTE** for class imbalance.

---

## ✨ Author

**Adetunji Samuel**
📧 Email: [your.email@example.com](mailto:your.email@example.com)
📚 Field: Student Physiotherapist, Software Engineer & ML Engineer


---

### 🧾 License

This project is open-source under the **MIT License**.
Feel free to use, modify, and distribute with proper attribution.

---


