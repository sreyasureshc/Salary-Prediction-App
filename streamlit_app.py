import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_dataset, handle_missing_values, encode_categorical_variables, define_features_target
from model_training import train_model

# Load the dataset
data = load_dataset("salary_prediction_data.csv")

# Preprocess the data
data, mappings = encode_categorical_variables(data)
X, y = define_features_target(data)
model = train_model(X, y)

# Streamlit app
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon=":moneybag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
...

# Streamlit app interface
st.title("Salary Prediction App")
st.write(
    """
    Welcome to the Salary Prediction App! This app predicts the salary based on various factors such as gender, 
    education level, job title, job location, and experience.
    """
)

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Visualization"])

if page == "Prediction":
    # Prediction page content
    st.subheader("Salary Prediction")
    st.sidebar.markdown("### Enter Details")

    # Input fields for user to enter gender, education level, job title, job location, and experience
    gender = st.sidebar.selectbox("Gender", list(mappings['Gender'].values()))
    education_level = st.sidebar.selectbox("Education Level", list(mappings['Education'].values()))
    job_title = st.sidebar.selectbox("Job Title", list(mappings['Job_Title'].values()))
    location = st.sidebar.selectbox("Job Location", list(mappings['Location'].values()))
    experience = st.sidebar.slider("Experience", min_value=0, max_value=40, value=10)

    # Predict button
    if st.sidebar.button("Predict Salary", key="predict_button", help="Click to predict salary"):
        # Handle user input and make prediction
        gender_encoded = list(mappings['Gender'].keys())[list(mappings['Gender'].values()).index(gender)]
        education_level_encoded = list(mappings['Education'].keys())[list(mappings['Education'].values()).index(education_level)]
        job_title_encoded = list(mappings['Job_Title'].keys())[list(mappings['Job_Title'].values()).index(job_title)]
        location_encoded = list(mappings['Location'].keys())[list(mappings['Location'].values()).index(location)]

        # Make prediction
        predicted_salary = model.predict([[gender_encoded, education_level_encoded, job_title_encoded, location_encoded, experience]])

        # Display predicted salary
        st.markdown(f'<div class="predicted-salary-box">Predicted Salary: ${predicted_salary[0]:,.2f}</div>', unsafe_allow_html=True)

elif page == "Visualization":
    # Visualization page content
    st.subheader("Visualization")
    st.sidebar.markdown("### Choose a Visualization")

    # Visualization options
    visualization_option = st.sidebar.radio("", ["Histogram", "Box Plot", "Violin Plot"])

    if visualization_option == "Histogram":
        st.markdown("### Histogram of Salary")
        # Histogram of Salary
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Salary'], bins=20, kde=True)
        plt.xlabel("Salary ($)")
        plt.ylabel("Frequency")
        st.pyplot(plt)

    elif visualization_option == "Box Plot":
        st.markdown("### Box Plot of Salary by Gender")
        # Box plot of Salary by Gender
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Gender', y='Salary', data=data)
        plt.xlabel("Gender")
        plt.ylabel("Salary ($)")
        st.pyplot(plt)

    elif visualization_option == "Violin Plot":
        st.markdown("### Violin Plot of Salary by Education Level")
        # Violin plot of Salary by Education Level
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Education', y='Salary', data=data)
        plt.xlabel("Education Level")
        plt.ylabel("Salary ($)")
        st.pyplot(plt)
