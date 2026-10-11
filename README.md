# Student Performance Analytics & Prediction

<p align="center">
  <strong>Exploratory data analysis • Statistical testing • Machine learning • Interactive dashboard</strong>
</p>

<p align="center">
  <a href="https://github.com/learner-Piyush/student-performance-analytics/actions/workflows/setup.yml">
    <img src="https://github.com/learner-Piyush/student-performance-analytics/actions/workflows/setup.yml/badge.svg" alt="GitHub Actions workflow status">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Portfolio%20Project-blue" alt="Portfolio project">
</p>

A data science project exploring the relationship between student learning behaviour and academic outcomes. It combines a documented Jupyter Notebook analysis with regression experiments and a Streamlit dashboard for interactive exploration.

> **Important:** The dataset is synthetic. Results demonstrate an analytics workflow; they should not be used to make decisions about real students.

## Contents

- [Project at a glance](#project-at-a-glance)
- [Interactive dashboard](#interactive-dashboard)
- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Key findings](#key-findings)
- [Machine-learning results](#machine-learning-results)
- [Tech stack](#tech-stack)
- [Run locally](#run-locally)
- [Automated checks](#automated-checks)
- [Project structure](#project-structure)
- [Limitations and future work](#limitations-and-future-work)
- [Academic context](#academic-context)
- [Author](#author)

## Project at a glance

| Area | Details |
|---|---|
| Goal | Explore student performance patterns and estimate exam scores |
| Analysis | Data cleaning, EDA, correlations, hypothesis tests, feature engineering |
| Model type | Regression |
| Target | `ExamScore` |
| Best tested model | Random Forest Regressor |
| Evaluation | 5-fold cross-validation and held-out test set |
| Interface | Interactive Streamlit dashboard |
| Automation | GitHub Actions installs dependencies and executes Jupyter notebooks |

## Interactive dashboard

The Streamlit app lets users explore the dataset without having to run the analysis notebook.

**Dashboard features**
- KPI cards for record count, average exam score, attendance and study hours
- Sidebar filters for study hours, attendance, grade code and gender (when available)
- Interactive exam-score distribution and final-grade charts
- Learning-factor visualisations and a correlation heatmap
- Filtered data preview and CSV download
- Optional upload of another compatible CSV dataset

**Open the app source:** [`streamlit_app.py`](./streamlit_app.py)

### Run the dashboard locally

1. Clone or download this repository.
2. Install dependencies using the instructions below.
3. Make sure `student_performance.csv` is in the repository root, next to `streamlit_app.py`.
4. Start Streamlit:

```bash
streamlit run streamlit_app.py
```

The app checks for the required columns `StudyHours`, `Attendance`, `ExamScore` and `FinalGrade`. To deploy through Streamlit Community Cloud, select this repository, the `main` branch and `streamlit_app.py` as the app entry point. Add a public live-app link here once deployment is confirmed.

## Problem statement

Can study habits, attendance, learning-related factors and demographic attributes help estimate a student's exam score?

The notebook investigates this question through exploratory and statistical analysis, then compares regression models. Since `FinalGrade` is derived from `ExamScore`, it is excluded from model input features to avoid target leakage. Predicted scores can then be interpreted against the dataset's grade bands.

## Dataset

- **Dataset:** Student Performance and Learning Behavior Dataset
- **Source:** [Kaggle — Adil Shamim](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style)
- **CSV file:** `student_performance.csv`
- **Original rows:** 14,003
- **Rows after exact-duplicate removal:** 12,469
- **Columns:** 16, as documented in the notebook

The dataset includes variables such as `StudyHours`, `Attendance`, `Resources`, `Motivation`, `LearningStyle`, `OnlineCourses`, `AssignmentCompletion`, `StressLevel`, `ExamScore` and `FinalGrade`.

### Grade bands

| Code | Grade | Exam score |
|---:|---|---:|
| 0 | A | 85–100 |
| 1 | B | 70–84 |
| 2 | C | 55–69 |
| 3 | F | 40–54 |

## Workflow

1. **Load and inspect** the CSV, data types, ranges, missing values and duplicates.
2. **Clean the data** and validate observed values.
3. **Explore** target distributions, univariate patterns, relationships and correlations.
4. **Test statistical associations** using ANOVA, chi-square tests and point-biserial correlation.
5. **Engineer features** such as `EngagementScore`, `DigitalSupportIndex`, `AttendAssign` and `StudyPerAge`.
6. **Train regression models** using a train/test split and cross-validation.
7. **Evaluate** MAE, RMSE and R²; inspect residuals and feature importance.
8. **Explore interactively** with the Streamlit dashboard.
9. **Run automated checks** through GitHub Actions.

## Key findings

- `FinalGrade` is a deterministic grade-band mapping of `ExamScore`; their Pearson correlation is approximately **−0.968** because grade codes run in the opposite order to score values.
- The 14 original behavioural and demographic features have near-zero individual linear correlations with `ExamScore` in this dataset.
- The three highest reported Random Forest feature importances were engineered features: `EngagementScore`, `AttendAssign` and `StudyPerAge`.
- Study hours and attendance show little separation across grade groups in this particular dataset.
- Statistical significance does not necessarily mean an effect is practically meaningful; several detected associations are very small.

## Machine-learning results

The notebook compared Linear Regression, Decision Tree, Random Forest and Gradient Boosting models. The reported best-performing model was a Random Forest Regressor with `max_depth=8`.

| Model | CV R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| **Random Forest (depth 8)** | **0.2047** | **13.270** | **15.489** | **0.2383** |
| Gradient Boosting | 0.1277 | 14.029 | 16.336 | 0.1527 |
| Decision Tree (depth 6) | 0.0433 | 14.577 | 17.167 | 0.0644 |
| Linear Regression | −0.0011 | 15.338 | 17.692 | 0.0062 |
| Mean baseline | 0.0000 | 15.381 | 17.747 | approximately 0.0000 |

**How to interpret this:** the Random Forest outperformed the baseline, but its test R² of 0.2383 means it explains only about 24% of the score variance. Prediction quality is limited, and the model should not be presented as a reliable real-world student-risk predictor.

## Tech stack

- **Language:** Python
- **Data work:** pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Statistics:** SciPy, statsmodels
- **Machine learning:** scikit-learn
- **Notebook:** Jupyter
- **Dashboard:** Streamlit
- **CI:** GitHub Actions

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/learner-Piyush/student-performance-analytics.git
cd student-performance-analytics
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the notebook

Ensure `student_performance.csv` is in the repository root, then run:

```bash
jupyter notebook
```

Open `PiyushRaj_StudentPerformanceAnalytics.ipynb` and use **Kernel → Restart & Run All**.

### 5. Or run the dashboard

```bash
streamlit run streamlit_app.py
```

## Automated checks

The [GitHub Actions workflow](./.github/workflows/setup.yml) runs on pushes to `main`/`master`, pull requests and manual dispatch. It sets up Python 3.11, installs dependencies, verifies key imports, and executes the Jupyter notebook(s) with errors treated as failures.

The badge at the top reflects the workflow's current status. Click it to inspect recent runs and logs.

## Project structure

```text
student-performance-analytics/
├── .github/
│   └── workflows/
│       └── setup.yml
├── PiyushRaj_StudentPerformanceAnalytics.ipynb
├── PiyushRaj_StudentPerformanceAnalytics_ProjectReport.docx
├── student_performance.csv
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Limitations and future work

### Limitations
- The dataset is synthetic, so findings may not generalise to real students.
- Most original features have weak individual relationships with the score target.
- The best model's predictive performance is modest and has not been validated on an external real-world dataset.
- Observational patterns do not establish causation.

### Future improvements
- Validate the workflow on a responsibly collected, real-world dataset.
- Add model tuning and stronger baseline comparisons.
- Improve explainability and model diagnostics.
- Add dashboard screenshots and a confirmed public deployment URL.
- Add automated smoke tests for the Streamlit app and its expected dataset schema.

## Academic context

Completed as part of the **IBM SkillsBuild — Data Analytics with AI Academic Internship 2026**, organised by **BharatCares** in association with **AICTE**.

## Author

**Piyush Raj**  
B.Tech, Computer Science & Engineering (Data Science)  
Haldia Institute of Technology

- GitHub: [learner-Piyush](https://github.com/learner-Piyush)
- LinkedIn: [Piyush Raj](https://www.linkedin.com/in/piyush-raj-371200244)

---

If you find this project useful, consider giving the repository a ⭐.
