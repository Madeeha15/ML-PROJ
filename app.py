import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Prediction & Analytics")
st.write("Machine Learning based Student Placement Prediction System")

# -----------------------------
# GENERATE STUDENT DATASET
# -----------------------------
np.random.seed(42)

n = 500

data = pd.DataFrame({
    "CGPA": np.round(np.random.uniform(5.0, 10.0, n), 2),
    "10th_Percentage": np.round(np.random.uniform(50, 100, n), 1),
    "12th_Percentage": np.round(np.random.uniform(50, 100, n), 1),
    "Backlogs": np.random.randint(0, 5, n),
    "Internship": np.random.randint(0, 2, n),
    "Projects": np.random.randint(0, 5, n),
    "Aptitude_Score": np.round(np.random.uniform(40, 100, n), 1),
    "Communication_Score": np.round(np.random.uniform(40, 100, n), 1)
})

# Create placement score
score = (
    data["CGPA"] * 10
    + data["10th_Percentage"] * 0.20
    + data["12th_Percentage"] * 0.20
    - data["Backlogs"] * 8
    + data["Internship"] * 12
    + data["Projects"] * 4
    + data["Aptitude_Score"] * 0.20
    + data["Communication_Score"] * 0.15
    + np.random.normal(0, 5, n)
)

threshold = score.median()

data["Placement"] = np.where(score >= threshold, 1, 0)

data["Placement_Status"] = data["Placement"].map({
    1: "Placed",
    0: "Not Placed"
})

# -----------------------------
# FEATURES AND TARGET
# -----------------------------
features = [
    "CGPA",
    "10th_Percentage",
    "12th_Percentage",
    "Backlogs",
    "Internship",
    "Projects",
    "Aptitude_Score",
    "Communication_Score"
]

X = data[features]
y = data["Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# MACHINE LEARNING MODELS
# -----------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
}

results = {}
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    predictions[name] = pred

    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred)
    }

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Analytics",
        "Model Comparison",
        "Placement Prediction"
    ]
)

# -----------------------------
# DASHBOARD
# -----------------------------

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    total_students = len(data)
    placed_students = data["Placement"].sum()
    placement_rate = placed_students / total_students * 100
    avg_cgpa = data["CGPA"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        total_students
    )

    col2.metric(
        "Placed Students",
        placed_students
    )

    col3.metric(
        "Placement Rate",
        f"{placement_rate:.1f}%"
    )

    col4.metric(
        "Average CGPA",
        f"{avg_cgpa:.2f}"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Placement Distribution")

        placement_count = data["Placement_Status"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            placement_count.values,
            labels=placement_count.index,
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Placement by CGPA")

        fig, ax = plt.subplots()

        ax.scatter(
            data["CGPA"],
            data["Placement"],
            alpha=0.5
        )

        ax.set_xlabel("CGPA")
        ax.set_ylabel("Placement")

        st.pyplot(fig)

    st.subheader("Dataset Preview")

    display_data = data.drop(
        columns=["Placement"]
    ).head(10)

    st.dataframe(
        display_data,
        use_container_width=True
    )


# -----------------------------
# ANALYTICS
# -----------------------------

elif page == "Analytics":

    st.header("📈 Placement Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Placement by Internship")

        internship_data = data.groupby(
            "Internship"
        )["Placement"].mean() * 100

        internship_data.index = [
            "No Internship",
            "Internship"
        ]

        st.bar_chart(internship_data)

    with col2:

        st.subheader("Placement by Projects")

        project_data = data.groupby(
            "Projects"
        )["Placement"].mean() * 100

        st.bar_chart(project_data)

    st.subheader("Placement by Backlogs")

    backlog_data = data.groupby(
        "Backlogs"
    )["Placement"].mean() * 100

    st.line_chart(backlog_data)

    st.subheader("Average Scores")

    avg_scores = data[
        [
            "CGPA",
            "10th_Percentage",
            "12th_Percentage",
            "Aptitude_Score",
            "Communication_Score"
        ]
    ].mean()

    st.bar_chart(avg_scores)

    # Random Forest Feature Importance

    rf_model = models["Random Forest"]

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.subheader("Important Factors for Placement")

    st.bar_chart(
        importance.set_index("Feature")
    )


# -----------------------------
# MODEL COMPARISON
# -----------------------------

elif page == "Model Comparison":

    st.header("🤖 Machine Learning Model Comparison")

    results_df = pd.DataFrame(results).T

    results_display = results_df.copy()

    results_display = results_display.round(3)

    st.dataframe(
        results_display,
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")

    accuracy_chart = results_df["Accuracy"]

    st.bar_chart(accuracy_chart)

    st.subheader("Confusion Matrices")

    col1, col2, col3 = st.columns(3)

    for column, (name, pred) in zip(
        [col1, col2, col3],
        predictions.items()
    ):

        with column:

            st.write(f"### {name}")

            cm = confusion_matrix(
                y_test,
                pred
            )

            fig, ax = plt.subplots()

            ax.imshow(cm)

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.set_xticklabels(
                ["Not Placed", "Placed"]
            )

            ax.set_yticklabels(
                ["Not Placed", "Placed"]
            )

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center"
                    )

            st.pyplot(fig)


# -----------------------------
# PLACEMENT PREDICTION
# -----------------------------

elif page == "Placement Prediction":

    st.header("🎯 Predict Student Placement")

    st.write(
        "Enter student details to predict placement."
    )

    col1, col2 = st.columns(2)

    with col1:

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1
        )

        tenth = st.number_input(
            "10th Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        twelfth = st.number_input(
            "12th Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        backlogs = st.number_input(
            "Backlogs",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:

        internship = st.selectbox(
            "Internship",
            ["No", "Yes"]
        )

        projects = st.number_input(
            "Number of Projects",
            min_value=0,
            max_value=10,
            value=2
        )

        aptitude = st.number_input(
            "Aptitude Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        communication = st.number_input(
            "Communication Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    if st.button(
        "Predict Placement",
        type="primary"
    ):

        internship_value = 1 if internship == "Yes" else 0

        input_data = pd.DataFrame([[
            cgpa,
            tenth,
            twelfth,
            backlogs,
            internship_value,
            projects,
            aptitude,
            communication
        ]], columns=features)

        st.subheader("Prediction Results")

        for name, model in models.items():

            prediction = model.predict(
                input_data
            )[0]

            if prediction == 1:

                st.success(
                    f"{name}: PLACED"
                )

            else:

                st.error(
                    f"{name}: NOT PLACED"
                )

        st.info(
            "Prediction is based on the trained machine learning models."
        )