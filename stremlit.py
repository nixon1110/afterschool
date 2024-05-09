import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('Afterschool')

# Load data from pickle files
data_path = r'D:\SourceCode\afterschool'
with open(os.path.join(data_path, 'data.pkl'), 'rb') as f:
    df = pickle.load(f)

with open(os.path.join(data_path, 'StudyLevel.pkl'), 'rb') as f:
    study_levels = pickle.load(f)

with open(os.path.join(data_path, 'intake_month.pkl'), 'rb') as f:
    intake_month = pickle.load(f)

with open(os.path.join(data_path, 'StudyArea.pkl'), 'rb') as f:
    study_area = pickle.load(f)

# Display study level options
st.title('What Level Do You Want to Study?')
selected_study_level = st.radio('Select option:', study_levels)

# Display year options
st.title('What year do you want to study?')
selected_option = st.radio(
    'Select option:',
    ['2024', '2025', '2026', 'Not Sure']
)

# Display intake month options
st.title('Pick intake month')
selected_intake_months = st.multiselect('PICK minimum one:', intake_month)

# Display study area options
st.title('what do you want to study')
selected_study_area = st.multiselect('PICK one or more:', study_area)

# Function to filter DataFrame based on selected options
def filtered_df(df, study_level, intake_months, study_areas):
    filtered_df = df[(df['StudyLevel'] == study_level) &
                     (df['intake_month'].isin(intake_months)) &
                     (df['studyArea'].isin(study_areas))]
    return filtered_df

# Function to preprocess tags
def preprocess_tags(df):
    df['tags'] = df['courseName'] + ' ' + df['StudyLevel'] + ' ' + df['intake_month'] + ' ' + df['studyArea'] + ' ' + df['collegeName']
    df['tags'] = df['tags'].str.lower()
    df['courseName'] = df['courseName'].str.replace(',', ' ')
    df['tags'] = df['tags'].str.replace(',', ' ')
    return df

# Function to recommend similar courses
def recommend(final, course):
    cv = CountVectorizer(max_features=5000)
    vectors = cv.fit_transform(final['courseName']).toarray()
    similarity = cosine_similarity(vectors)
    
    # Reset index to ensure alignment with similarity matrix
    final.reset_index(drop=True, inplace=True)
    
    course_index = final[final['courseName'] == course].index
    if len(course_index) == 0:
        st.write("Course not found.")
        return
    
    course_index = course_index[0]
    
    distances = similarity[course_index]
    
    sorted_indices = np.argsort(distances)[::-1]
    top_indices = sorted_indices[1:10]
    
    recommended_courses = []
    for idx in top_indices:
        recommended_course = final.iloc[idx]['courseName']
        recommended_university = final.iloc[idx]['collegeName']
        recommended_courses.append((recommended_course, recommended_university))
    
    return recommended_courses


# Apply filters
filtered_data = filtered_df(df, selected_study_level, selected_intake_months, selected_study_area)
final_data = preprocess_tags(filtered_data)

# Display unique course names for selection
unique_course_names = final_data['courseName'].unique()
selected_course_name = st.selectbox('Select course:', unique_course_names)

# Display recommendations
if st.button('Get Recommendations'):
    recommendations = recommend(final_data, selected_course_name)
    if recommendations:
        st.title('Recommended Courses:')
        for course, university in recommendations:
            st.write(f"Course: {course} | University: {university}")
    else:
        st.write("No recommendations found.")
