# Student Performance Analytics and Prediction

---

## Overview

This project applies data analytics and machine learning techniques to understand
the factors that influence student academic performance and to predict a student''s
exam score — and, by extension, their final grade — based on learning behaviour
and personal attributes.

The analysis covers the complete data science pipeline: data loading and
understanding, exploratory data analysis (EDA), statistical hypothesis testing,
feature engineering, regression modelling, model evaluation, and a prediction
demonstration. All code is self-contained in a single Jupyter Notebook.

---

## Problem Statement

Educational institutions face a persistent challenge: identifying at-risk students
early enough to intervene effectively. Traditional approaches rely on
end-of-semester grades, by which point it is often too late to offer meaningful
support.

**Core question:**
> *Can we use a student''s study habits, attendance, motivation, technology usage,
> and demographic attributes to predict their final academic grade before the end
> of the semester?*

The project addresses this by building regression models that predict `ExamScore`
from 14 behavioural and demographic features, and then mapping the predicted score
to the corresponding final grade band.

---

## Objectives

1. **Explore** the dataset to understand its structure, distributions, and
   inter-variable relationships.
2. **Identify** the most significant predictors of student academic performance
   through EDA and correlation analysis.
3. **Analyse** statistical relationships between behavioural/demographic features
   and final grade using ANOVA and chi-square tests.
4. **Engineer** composite features to capture joint patterns not visible in
   individual raw columns.
5. **Train and evaluate** multiple regression models to predict `ExamScore`.
6. **Select and explain** the best-performing model based on cross-validated metrics.
7. **Demonstrate** the model''s predictive output on held-out test-set students and
   hypothetical student profiles.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Student Performance and Learning Behavior Dataset |
| **Source** | Kaggle — Adil Shamim |
| **Link** | https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style |
| **File** | `student_performance.csv` |
| **Records** | 14,003 rows (12,469 after deduplication) |
| **Features** | 16 columns — 15 input features + 1 target variable |
| **Missing values** | None |
| **Encoding** | All columns are integer-encoded (`int64`) |

### Target Variable

`FinalGrade` — ordinal encoding of academic grade:

| Code | Grade | ExamScore Range |
|------|-------|-----------------|
| 0 | A | 85 – 100 |
| 1 | B | 70 – 84 |
| 2 | C | 55 – 69 |
| 3 | F | 40 – 54 |

> The regression models predict `ExamScore` (the continuous upstream driver of
> `FinalGrade`). The predicted score is then mapped to the corresponding grade band.

### Feature Summary

| Column | Type | Description |
|---|---|---|
| `StudyHours` | Numeric | Weekly hours spent studying (5 – 44) |
| `Attendance` | Numeric | Class attendance percentage (60 – 100) |
| `Resources` | Ordinal | Access to learning resources (0=Low, 1=Medium, 2=High) |
| `Extracurricular` | Binary | Participation in extracurricular activities |
| `Motivation` | Ordinal | Motivation level (0=Low, 1=Medium, 2=High) |
| `Internet` | Binary | Internet access at home |
| `Gender` | Binary | Student gender (0/1) |
| `Age` | Numeric | Student age (18 – 29) |
| `LearningStyle` | Ordinal | Preferred learning style (0=Visual … 3=Kinaesthetic) |
| `OnlineCourses` | Numeric | Number of online courses taken (0 – 20) |
| `Discussions` | Binary | Participation in class discussions |
| `AssignmentCompletion` | Numeric | Assignment completion rate % (50 – 100) |
| `ExamScore` | Numeric | Exam score (40 – 100) — **regression target** |
| `EduTech` | Binary | Use of educational technology |
| `StressLevel` | Ordinal | Stress level (0=Low, 1=Medium, 2=High) |
| `FinalGrade` | Ordinal | Final academic grade (0=A, 1=B, 2=C, 3=F) — **classification label** |

---

## Technologies Used

| Category | Tool / Library | Version (runtime) |
|---|---|---|
| Language | [![Python 3](https://img.shields.io/badge/Python_3-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) | >= 3.9 |
| Data manipulation | [![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) | 3.0.6 |
| Numerical computing | [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) | 2.5.3 |
| Visualisation | [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/) | >= 3.7 |
| Visualisation | [![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/) | 0.13.2 |
| Statistical analysis | [![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/) | >= 1.10 |
| Machine learning | [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) | 1.9.1 |
| Notebook environment | [![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/) | — |

---

## Project Workflow

1. **Import Libraries** — Load all dependencies; verify installed versions.
2. **Load Dataset** — Read `student_performance.csv` into a pandas DataFrame.
3. **Data Understanding** — Inspect shape, data types, descriptive statistics,
   missing values, duplicate rows, and unique value catalogues for each column.
4. **Data Cleaning** — Remove 1,534 exact duplicate rows; validate value ranges;
   confirm no missing values or type corrections are required.
5. **Exploratory Data Analysis** — Four-part EDA: target distribution, univariate
   distributions, bivariate relationships, and multivariate correlation analysis.
6. **Statistical Analysis** — One-Way ANOVA, chi-square tests of independence, and
   point-biserial correlation to test which features have statistically significant
   associations with academic performance.
7. **Feature Engineering** — Construct four composite features:
   `EngagementScore`, `DigitalSupportIndex`, `AttendAssign`, and `StudyPerAge`.
8. **Machine Learning** — Prepare feature matrix (18 features), split 80/20
   train/test, scale for linear model, train four regression models, evaluate via
   5-fold cross-validation and held-out test set.
9. **Model Evaluation** — Compare models on MAE, RMSE, and R²; inspect residuals
   and feature importances.
10. **Prediction Demonstration** — Apply the best model to test-set students and
    three hypothetical student profiles.

---

## Exploratory Data Analysis

The EDA was conducted on the cleaned dataset (12,469 rows) and structured in four parts.

**Part A — Target Variable Analysis**

`FinalGrade` is nearly balanced across all four classes:
Grade A 27.3% · Grade B 23.6% · Grade C 25.8% · Grade F 23.3%.

**Part B — Univariate Analysis**

- `StudyHours`: mean ≈ 20 hrs/week, approximately normally distributed (5–44).
- `Attendance`: mean ≈ 80%, range 60–100%.
- `AssignmentCompletion`: range 50–100%.
- `ExamScore`: mean ≈ 70.3, std ≈ 17.7, range 40–100.
- Categorical features (`Resources`, `Motivation`, `StressLevel`, `LearningStyle`,
  `Gender`) have roughly uniform or slight distributions across their levels.

**Part C — Bivariate Analysis**

- Mean `ExamScore` by grade — A: 92.57 · B: 77.07 · C: 61.88 · F: 46.75 —
  confirming perfectly non-overlapping grade bands.
- `StudyHours`, `Attendance`, and `AssignmentCompletion` show nearly identical
  mean values across all four grade bands (differences < 1 unit), indicating
  negligible individual predictive power.
- `OnlineCourses`, `Discussions`, `EduTech`, `Motivation`, and `StressLevel`
  similarly show no meaningful grade-group differences.
- Low-stress students have a slightly higher proportion of Grade A (30.2%) compared
  to medium- and high-stress groups (≈ 26–27%).

**Part D — Multivariate Analysis**

- Pearson correlation of `ExamScore` with `FinalGrade`: **r = −0.968** — the
  dominant relationship in the entire dataset.
- All 14 behavioural/demographic features have near-zero individual correlation
  with `ExamScore` (|r| < 0.04 for every feature).
- `Discussions` and `StressLevel` show the highest individual correlations with
  `FinalGrade` (r = 0.040 and 0.040 respectively — negligible in magnitude).

---

## Machine Learning

### Problem Framing

| Item | Detail |
|---|---|
| **Problem type** | Regression |
| **Target variable** | `ExamScore` (continuous, 40–100) |
| **Why ExamScore** | It is the sole continuous academic-performance variable and the genuine upstream driver of `FinalGrade`; predicting it yields a more informative output than predicting a class label directly |
| **Leakage prevention** | `FinalGrade` is excluded from all feature sets (deterministically derived from `ExamScore`) |

### Feature Set (18 features)

14 original features: `StudyHours`, `Attendance`, `Resources`, `Extracurricular`,
`Motivation`, `Internet`, `Gender`, `Age`, `LearningStyle`, `OnlineCourses`,
`Discussions`, `AssignmentCompletion`, `EduTech`, `StressLevel`.

4 engineered features: `EngagementScore`, `DigitalSupportIndex`, `AttendAssign`,
`StudyPerAge`.

### Preprocessing

- 80 / 20 train–test split (9,975 train · 2,494 test), `random_state=42`.
- `StandardScaler` applied to the training fold only — used for Linear Regression.
- Tree-based models use unscaled features.

### Models Trained

| Model | Configuration |
|---|---|
| Linear Regression | Default (scaled features) |
| Decision Tree | `max_depth=6`, `random_state=42` |
| Random Forest | `n_estimators=150`, `max_depth=8`, `random_state=42` |
| Gradient Boosting | scikit-learn `GradientBoostingRegressor` (default depth) |

### Evaluation Metrics

5-fold cross-validation (primary criterion) + held-out test set.
Metrics: **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error), **R²**.

---

## Results

All metrics are from the executed notebook; no values have been estimated or fabricated.

### Model Comparison

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| **Random Forest (d=8)** | **13.551** | **15.766** | **0.2047** | **13.270** | **15.489** | **0.2383** |
| Gradient Boosting | 14.211 | 16.511 | 0.1277 | 14.029 | 16.336 | 0.1527 |
| Decision Tree (d=6) | 14.710 | 17.291 | 0.0433 | 14.577 | 17.167 | 0.0644 |
| Linear Regression | 15.318 | 17.689 | −0.0011 | 15.338 | 17.692 | 0.0062 |
| Baseline (mean) | 15.381 | 17.747 | 0.0000 | 15.381 | 17.747 | −0.0000 |

### Selected Model — Random Forest (max_depth=8)

Selected on the basis of highest CV R² and lowest CV RMSE across all folds.

| Metric | Value |
|---|---|
| Test MAE | 13.270 score points |
| Test RMSE | 15.489 score points |
| Test R² | 0.2383 |
| Residual mean | −0.08 (near-zero bias) |
| Residual std | 15.49 score points |
| Predictions within ±10 pts | 39.0% |
| Predictions within ±15 pts | 59.0% |
| Grade-band classification accuracy | 29.47% |

### Top Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| `EngagementScore` (engineered) | 0.1772 |
| `AttendAssign` (engineered) | 0.1662 |
| `StudyPerAge` (engineered) | 0.1428 |
| `OnlineCourses` | 0.1064 |
| `AssignmentCompletion` | 0.0935 |
| `Attendance` | 0.0898 |
| `Age` | 0.0636 |
| `StudyHours` | 0.0629 |
| `LearningStyle` | 0.0330 |
| `StressLevel` | 0.0232 |

> **Interpretation note:** All models achieve low-to-moderate R² values. This
> reflects a fundamental property of the dataset: `ExamScore` appears to be
> synthetically generated from all 14 features via a non-linear function that
> produces near-uniform marginal distributions, making individual linear or
> tree-based signals very weak. The Random Forest does outperform the mean
> baseline, but the dataset itself limits achievable predictive accuracy.

---

## Key Findings

All findings below are directly supported by computed statistics from the cleaned
dataset (n = 12,469).

1. **ExamScore is the near-sole determinant of FinalGrade.** Pearson r = −0.968.
   The four grade bands are perfectly non-overlapping: A = 85–100, B = 70–84,
   C = 55–69, F = 40–54. `FinalGrade` is a deterministic binning of `ExamScore`.

2. **No individual feature is a strong linear predictor of ExamScore.** All 14
   raw features have |r| < 0.04 with `ExamScore`. One-Way ANOVA finds that only
   `Attendance` (F = 4.74, p = 0.003) and `AssignmentCompletion`
   (F = 5.03, p = 0.002) differ significantly across grade groups, but effect
   sizes are negligible (mean differences < 1.5 percentage points).

3. **Study hours and attendance are broadly uniform across grade bands.**
   Mean `StudyHours`: A = 20.12, B = 20.02, C = 20.02, F = 19.96.
   Mean `Attendance`: A = 80.11, B = 80.44, C = 79.71, F = 80.76.
   These features do not separate grade groups in this dataset.

4. **Most categorical features show no statistically significant association with
   FinalGrade.** Chi-square tests find that `Resources`, `Extracurricular`,
   `Motivation`, `Internet`, `Gender`, and `EduTech` are not significantly
   associated with `FinalGrade` (all p > 0.05). Only `LearningStyle`,
   `Discussions`, and `StressLevel` reach significance (p < 0.001), though their
   effect sizes remain small.

5. **Engineered composite features capture the most variance.** The top three
   Random Forest feature importances are all engineered features
   (`EngagementScore`, `AttendAssign`, `StudyPerAge`), suggesting combined
   signals are more informative than any single raw feature.

6. **The dataset is synthetic.** Uniform marginal distributions and near-zero
   pairwise correlations are characteristic of a synthetically generated dataset.
   Findings reflect statistical patterns of this synthetic data and may not
   generalise to real student populations.

---

## Limitations

### Dataset Limitations

- **Synthetic data.** The dataset is synthetically generated; findings may not
  reflect real-world student populations or generalise to actual institutional
  contexts.
- **Near-zero feature–target correlations.** All 14 raw features have |r| < 0.04
  with `ExamScore`, severely limiting predictive power for any model.
- **10.95% duplicate rows** were present in the original dataset (1,534 rows).
- **Restricted value ranges.** `Attendance` is bounded at 60–100% and
  `AssignmentCompletion` at 50–100%, excluding ranges common in real settings.

### Model Limitations

- The best model (Random Forest) achieves Test R² = 0.2383, explaining about 24%
  of variance in `ExamScore`. The remaining ~76% is unexplained by available features.
- Grade-band classification accuracy of 29.47% is marginally above the 25% random
  baseline for four classes.
- Models were not validated on real, external student data.

### Scope Limitations

- No time-series or semester-progression data; the model cannot track student
  improvement over time.
- Causal inference is not possible from observational (and synthetic) data.

---

## Future Scope

- **Validate on real student data** to confirm whether patterns observed in the
  synthetic dataset transfer to genuine institutional records.
- **Incorporate non-linear ensemble models** (e.g., XGBoost, LightGBM) with
  hyperparameter tuning to improve predictive performance.
- **Collect richer features** — prior academic history, socioeconomic background,
  course difficulty, teacher–student engagement — that may provide genuine
  predictive signal.
- **Apply NLP to student feedback** or open-ended survey responses to capture
  qualitative engagement signals.
- **Build an early-warning dashboard** that surfaces at-risk student predictions
  mid-semester for targeted academic support.
- **Extend to multi-institution datasets** to assess cross-context generalisability.

---

## Project Structure

```
.
├── PiyushRaj_StudentPerformanceAnalytics.ipynb
├── student_performance.csv
├── requirements.txt
├── PiyushRaj_StudentPerformanceAnalytics_ProjectReport.docx
└── README.md
```

---

## Setup and Installation

### Prerequisites

- **Python 3.9 or later** — Download from https://www.python.org/downloads/  
  During installation on Windows, check **"Add Python to PATH"**.

### Step-by-step Instructions

**1. Create a virtual environment**

Open a terminal (Command Prompt or PowerShell on Windows, or a terminal on
macOS/Linux) in the project folder and run:

```bash
python -m venv .venv
```

**2. Activate the virtual environment**

Windows — Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

Windows — PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Start Jupyter Notebook**

```bash
jupyter notebook
```

This opens Jupyter in your default web browser.

**5. Open the notebook**

In the Jupyter file browser, click `PiyushRaj_StudentPerformanceAnalytics.ipynb`.

**6. Ensure the dataset is present**

`student_performance.csv` must be in the **same directory** as the notebook file.
The notebook loads it with:

```python
df = pd.read_csv('student_performance.csv')
```

**7. Run all cells**

Select **Kernel → Restart & Run All** to execute all cells from top to bottom in
order.

---

## Academic Project

This project was completed as part of the:

**IBM SkillsBuild — Data Analytics with AI Academic Internship 2026**  
Organised by **BharatCares** in association with **AICTE**

---

## Author

**Piyush Raj**  
B.Tech — Computer Science & Engineering (Data Science)  
Haldia Institute of Technology

- GitHub: [github.com/learner-Piyush](https://github.com/learner-Piyush)
- LinkedIn: [linkedin.com/in/piyush-raj-371200244](https://www.linkedin.com/in/piyush-raj-371200244)