
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_performance_model.pkl")

st.write("Expected Features:")
st.write(model.feature_names_in_)
st.title("🎓 Student Performance Predictor")

attendance = st.number_input("Attendance (%)", 0, 100, 80)
maths = st.number_input("Maths Marks", 0, 100, 70)
science = st.number_input("Science Marks", 0, 100, 70)
english = st.number_input("English Marks", 0, 100, 70)
internal = st.number_input("Internal Marks", 0, 20, 15)

if st.button("Predict"):

    data = pd.DataFrame({
        "Attendance":[attendance],
        "Maths":[maths],
        "Science":[science],
        "English":[english],
        "Internal_Marks":[internal]
    })

    prediction = model.predict(data)

    st.success(f"Predicted Score: {prediction[0]:.2f}")
