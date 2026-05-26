# 🧠 Gen Z Social Media Usage — Mental Health Score Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-1M%20rows-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> Predicting mental health scores of Gen Z users based on their social media usage patterns using **Linear Regression** and **Random Forest Regressor**.

---

## 📌 Project Overview

Social media is deeply embedded in Gen Z's daily life. This project explores how usage behaviour — screen time, platform choice, session frequency, and night usage — correlates with and can predict **mental health scores**.

Using a dataset of **1 million records**, this project covers the full data science pipeline:
- Exploratory Data Analysis (EDA)
- Feature Engineering & Encoding
- Machine Learning Model Training & Evaluation
- Feature Importance Analysis

---

## 📂 Project Structure

```
genz_mental_health_ml/
│
├── mental_health_predictor.py   # Main script — EDA + ML pipeline
├── eda_plots.png                # Generated EDA visualizations
├── feature_importance.png       # Random Forest feature importance chart
└── README.md                    # Project documentation
```

---

## 📊 Dataset

| Property       | Detail                          |
|----------------|---------------------------------|
| Source         | Custom / Kaggle                 |
| Rows           | 1,000,000                       |
| Target Column  | `mental_health_score`           |
| Key Features   | `daily_usage_hours`, `age`, `primary_platform`, `addiction_level`, `night_usage`, `screen_time_before_sleep` |

---

## 🔧 Tech Stack

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| Python          | Core language                    |
| Pandas & NumPy  | Data loading, cleaning, EDA      |
| Seaborn & Matplotlib | Visualization              |
| Scikit-learn    | Encoding, ML models, evaluation  |

---

## ⚙️ ML Pipeline

### 1. Data Loading & Inspection
- Loaded 1M row CSV
- Checked shape, null values, and data types

### 2. Exploratory Data Analysis (EDA)
- Distribution of `daily_usage_hours`
- Platform usage breakdown (countplot)
- Scatter plot: screen time vs mental health score
- Correlation matrix with target variable

### 3. Feature Engineering
| Encoding Type     | Column(s)                                    |
|-------------------|----------------------------------------------|
| Ordinal Encoding  | `addiction_level` → Low=0, Medium=1, High=2  |
| One-Hot Encoding  | `gender`, `country`, `primary_platform`, `purpose` |
| Numeric (as-is)   | `age`, `daily_usage_hours`, `num_platforms_used`, `avg_session_minutes`, `night_usage`, `screen_time_before_sleep` |

### 4. Model Training

#### 🔵 Linear Regression — Full 1M rows
```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

#### 🟠 Random Forest Regressor — 10K sample (for speed)
```python
rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
rf.fit(X_train_rf, y_train_rf)
```

> Random Forest was trained on a 10,000-row sample to manage memory and runtime on the full 1M dataset.

---

## 📈 Results

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| MAE    | —                 | —             |
| MSE    | —                 | —             |
| R²     | —                 | —             |

> ℹ️ Run the script to populate actual metric values from your environment.

---

## 🔍 Key Findings

- **Screen time** and **sleep before bed** were among the strongest predictors of mental health scores
- **Addiction level** (ordinal encoded) showed significant correlation with the target
- Random Forest captured non-linear relationships that Linear Regression missed

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/nikhil24sharma/genz-mental-health-ml.git
cd genz-mental-health-ml
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Update the dataset path in the script
```python
DATA_PATH = r"path/to/genz_social_media_usage_1M.csv"
```

### 4. Run the script
```bash
python mental_health_predictor.py
```

---

## 📬 Connect

**Nikhil Sharma**
- 📧 nikhil241103@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/nikhil-sharma)
- 🐙 [GitHub](https://github.com/nikhil24sharma)

---

> ⭐ If you found this project useful, consider giving it a star!
