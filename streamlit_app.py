from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "student_performance.csv"

st.title("📊 Student Performance Analytics")
st.caption(
    "Explore study habits, attendance, learning factors, and exam outcomes "
    "in the student performance dataset."
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 1rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(file_bytes: bytes | None = None) -> pd.DataFrame:
    """Load the repository dataset or a user-uploaded CSV."""
    if file_bytes is not None:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes))
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")
    return pd.read_csv(DATA_PATH)


with st.sidebar:
    st.header("Dashboard controls")
    uploaded_file = st.file_uploader("Use another CSV (optional)", type=["csv"])

try:
    raw_df = load_data(uploaded_file.getvalue() if uploaded_file else None)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("Make sure student_performance.csv is in the same folder as streamlit_app.py.")
    st.stop()

if raw_df.empty:
    st.warning("The selected dataset has no rows.")
    st.stop()

df = raw_df.copy()
df.columns = df.columns.str.strip()

required_columns = {"StudyHours", "Attendance", "ExamScore", "FinalGrade"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.error("The dataset is missing required columns: " + ", ".join(sorted(missing_columns)))
    st.stop()

numeric_columns = [
    "StudyHours", "Attendance", "OnlineCourses", "Discussions",
    "AssignmentCompletion", "ExamScore", "Age",
]
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Sidebar filters adapt to the available columns.
with st.sidebar:
    st.subheader("Filter records")

    filtered = df.copy()

    if "StudyHours" in filtered.columns and filtered["StudyHours"].notna().any():
        low = float(filtered["StudyHours"].min())
        high = float(filtered["StudyHours"].max())
        if low < high:
            study_range = st.slider(
                "Study hours", min_value=low, max_value=high,
                value=(low, high), step=1.0
            )
            filtered = filtered[filtered["StudyHours"].between(*study_range)]

    if "Attendance" in filtered.columns and filtered["Attendance"].notna().any():
        low = float(filtered["Attendance"].min())
        high = float(filtered["Attendance"].max())
        if low < high:
            attendance_range = st.slider(
                "Attendance", min_value=low, max_value=high,
                value=(low, high), step=1.0
            )
            filtered = filtered[filtered["Attendance"].between(*attendance_range)]

    if "FinalGrade" in filtered.columns:
        grade_options = sorted(filtered["FinalGrade"].dropna().unique().tolist())
        selected_grades = st.multiselect(
            "Final grade code", options=grade_options, default=grade_options
        )
        filtered = filtered[filtered["FinalGrade"].isin(selected_grades)]

    if "Gender" in filtered.columns:
        gender_options = sorted(filtered["Gender"].dropna().unique().tolist())
        selected_gender = st.multiselect(
            "Gender code", options=gender_options, default=gender_options
        )
        filtered = filtered[filtered["Gender"].isin(selected_gender)]

if filtered.empty:
    st.warning("No records match these filters. Adjust the filters in the sidebar.")
    st.stop()

st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** records.")

# KPI cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Students / records", f"{len(filtered):,}")
k2.metric("Average exam score", f"{filtered['ExamScore'].mean():.1f}" if filtered["ExamScore"].notna().any() else "N/A")
k3.metric("Average attendance", f"{filtered['Attendance'].mean():.1f}" if filtered["Attendance"].notna().any() else "N/A")
k4.metric("Average study hours", f"{filtered['StudyHours'].mean():.1f}" if filtered["StudyHours"].notna().any() else "N/A")

st.divider()

overview_tab, factors_tab, data_tab = st.tabs(
    ["📈 Performance overview", "🔎 Learning factors", "🧾 Explore data"]
)

with overview_tab:
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered.dropna(subset=["ExamScore"]),
            x="ExamScore",
            nbins=20,
            title="Exam score distribution",
            labels={"ExamScore": "Exam score", "count": "Number of records"},
            color_discrete_sequence=["#4F46E5"],
        )
        fig.update_layout(bargap=0.08)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        grade_counts = (
            filtered["FinalGrade"].value_counts(dropna=False)
            .rename_axis("FinalGrade")
            .reset_index(name="Records")
        )
        grade_counts["FinalGrade"] = grade_counts["FinalGrade"].astype(str)
        fig = px.bar(
            grade_counts,
            x="FinalGrade",
            y="Records",
            title="Records by final-grade code",
            labels={"FinalGrade": "Final-grade code", "Records": "Records"},
            color="FinalGrade",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered.dropna(subset=["StudyHours", "ExamScore"]),
        x="StudyHours",
        y="ExamScore",
        color="FinalGrade" if "FinalGrade" in filtered.columns else None,
        opacity=0.7,
        title="Study hours vs. exam score",
        labels={"StudyHours": "Study hours", "ExamScore": "Exam score", "FinalGrade": "Final-grade code"},
        hover_data=[c for c in ["Attendance", "AssignmentCompletion", "StressLevel"] if c in filtered.columns],
    )
    st.plotly_chart(fig, use_container_width=True)

with factors_tab:
    st.subheader("Compare learning factors with exam scores")
    candidate_features = [
        c for c in [
            "StudyHours", "Attendance", "OnlineCourses", "Discussions",
            "AssignmentCompletion", "Age", "ExamScore"
        ]
        if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c])
    ]

    if len(candidate_features) >= 2:
        chosen_feature = st.selectbox(
            "Choose a factor to compare with exam score",
            [c for c in candidate_features if c != "ExamScore"] or candidate_features,
        )
        if chosen_feature != "ExamScore":
            fig = px.scatter(
                filtered.dropna(subset=[chosen_feature, "ExamScore"]),
                x=chosen_feature,
                y="ExamScore",
                color="FinalGrade" if "FinalGrade" in filtered.columns else None,
                trendline="ols" if len(filtered) >= 3 else None,
                opacity=0.7,
                title=f"{chosen_feature} vs. exam score",
            )
            st.plotly_chart(fig, use_container_width=True)

        corr = filtered[candidate_features].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation between numeric variables",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns are available for factor analysis.")

    st.info(
        "Correlation shows association, not causation. FinalGrade and ExamScore are "
        "outcomes in this dataset; do not treat them as independent causes of one another."
    )

with data_tab:
    st.subheader("Filtered dataset")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_student_performance.csv",
        mime="text/csv",
    )

    with st.expander("Dataset summary"):
        st.write(f"**Rows:** {len(filtered):,}")
        st.write(f"**Columns:** {filtered.shape[1]}")
        st.dataframe(filtered.describe(include="all").transpose(), use_container_width=True)

st.divider()
st.caption(
    "Student Performance Analytics • Built with Python, Pandas, Plotly, and Streamlit. "
    "Visualizations are descriptive and should not be interpreted as causal claims."
)
