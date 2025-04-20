# streamlit_app.py

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Recommendation function
def recommend_roles(job_roles, skills, input_role, top_n=3):
    vectorizer = TfidfVectorizer()
    skill_matrix = vectorizer.fit_transform(skills)

    if input_role not in job_roles:
        return []

    input_index = job_roles.index(input_role)
    similarities = cosine_similarity(skill_matrix[input_index], skill_matrix).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    recommended_roles = [job_roles[i] for i in similar_indices]

    return recommended_roles

# Job data
job_roles = ["Data Scientist", "ML Engineer", "Data Analyst", "Data Engineer", "AI Researcher", "Business Analyst", "NLP Engineer"]
skills = [
    "Python, Statistics, Machine Learning, Data Visualization",
    "Python, Machine Learning, Deployment, Algorithms",
    "SQL, Python, Data Visualization, Excel",
    "Python, SQL, ETL, Cloud Computing",
    "Python, Deep Learning, Machine Learning, Algorithms",
    "Excel, SQL, Data Visualization, Business Intelligence",
    "Python, NLP, Machine Learning, Deep Learning"
]

# --- Streamlit Interface ---
st.title("💼 Job Role Recommender")
st.write("Select your job role from the list below and get similar job recommendations based on required skills.")

# Dropdown for job roles
input_role = st.selectbox("Choose a job role:", job_roles)

# Button to recommend
if st.button("Recommend Similar Roles"):
    recommended_roles = recommend_roles(job_roles, skills, input_role)
    if recommended_roles:
        st.success(f"Recommended roles for **{input_role}**:")
        for role in recommended_roles:
            st.write(f"👉 {role}")
    else:
        st.warning("No similar roles found.")
