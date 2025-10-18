## 🩺 Diabetes Prediction Web App

This project is a **machine learning web application** that predicts the likelihood of a person having diabetes based on health indicators such as age, BMI, blood glucose level, and HbA1c level.
It is built using **Python, scikit-learn, and Streamlit**.

---

### 🚀 Features

* Upload or manually input patient health data
* Predict diabetes risk using a trained logistic regression model
* Display model confidence and prediction results
* View model performance (accuracy, precision, recall, F1-score)
* Simple, interactive web interface

---

### 🧠 Model Details

* **Algorithm:** Logistic Regression
* **Best Parameters (from GridSearchCV):**

  ```python
  {'C': 1, 'class_weight': None, 'solver': 'lbfgs'}
  ```
* **Accuracy:** 97.4%
* **F1-Score (class 1):** 0.73
* **Top Predictive Features:**

  | Feature             | Coefficient | Meaning                                      |
  | ------------------- | ----------- | -------------------------------------------- |
  | HbA1c_level         | 2.30        | Strongest positive correlation with diabetes |
  | blood_glucose_level | 1.17        | Higher glucose increases diabetes risk       |
  | age                 | 0.81        | Older patients more likely                   |
  | bmi                 | 0.61        | Higher BMI increases risk                    |
  | hypertension        | 0.21        | Mild positive correlation                    |
  | gender              | 0.17        | Slight difference by gender                  |
  | smoking_history     | 0.16        | Minor influence                              |
  | heart_disease       | 0.13        | Slight positive correlation                  |

---

### 🧩 Tech Stack

* **Backend / ML:** Python, scikit-learn, pandas, numpy
* **Frontend:** Streamlit
* **Model Serialization:** pickle
* **Visualization:** matplotlib, seaborn

---

### ⚙️ Setup Instructions

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/diabetes-prediction-app.git
cd diabetes-prediction-app
```

#### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Train the Model (optional)

If you want to retrain the model:

```bash
python train_model.py
```

This will:

* Load and preprocess the dataset
* Train the logistic regression model
* Evaluate and save it as `diabetes_model.pkl`

#### 5️⃣ Run the Web App

```bash
streamlit run app.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**

---

### 📁 Project Structure

```
diabetes-prediction-app/
│
├── app.py                # Streamlit web app
├── train_model.py        # Model training script
├── diabetes_model.pkl    # Saved logistic regression model
├── requirements.txt      # Dependencies
├── dataset.csv           # Dataset used
├── README.md             # Project documentation
└── utils/                # Optional: preprocessing utilities
```

---

### 📊 Example Prediction

| Input                                            | Output                         |
| ------------------------------------------------ | ------------------------------ |
| age = 45, bmi = 30.1, HbA1c = 7.5, glucose = 180 | **Predicted: Diabetes (1)**    |
| age = 22, bmi = 20.2, HbA1c = 4.9, glucose = 85  | **Predicted: No Diabetes (0)** |

---

### 💡 Why Logistic Regression?

We chose **Logistic Regression** because:

* It provides **probabilistic outputs** and clear feature interpretability.
* It’s ideal for **binary classification problems** like diabetes (Yes/No).
* It performs well on **structured medical data** with limited noise.

If you need more accuracy at the cost of interpretability, you can try:

| Model              | When to Use                                               |
| ------------------ | --------------------------------------------------------- |
| **Random Forest**  | Non-linear relationships, high accuracy, less explainable |
| **XGBoost**        | Complex patterns, large datasets                          |
| **SVM**            | Small datasets, clear margin separation                   |
| **Neural Network** | Big data with non-linear relationships                    |

---

### 🧾 Requirements

Create a `requirements.txt` file:

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

### 🧠 Future Improvements

* Add deep learning models for comparison
* Deploy using **Streamlit Cloud / Render / Hugging Face Spaces**
* Add REST API endpoint using **FastAPI**
* Include feature importance visualization in the app

---

### 👨‍⚕️ Author

**Adetunji Samuel**
📧 [[samueladetunji000@gmail.com](mailto:samueladetunji000@gmail.com)]


