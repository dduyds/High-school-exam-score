import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
st.set_page_config(
    page_title="Student"
)
#Introduction
img = Image.open('images/Student.jpg')
st.image(img,use_column_width=True)
st.title("Number of Student in Subjects")
st.markdown( """
    On this page you can see compare of each subject that student participate in some year by choice.
    """
)

# @st.cache
def load_data(year):
    path_file='diemthi'+str(year)+'.csv'
    temp = pd.read_csv(path_file)
    df = pd.DataFrame(temp)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df.round(2)

#sidebar
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year',list(reversed(range(2019,2021))))
#repare data
df = load_data(selected_year)
sort_columns = np.array(df.columns)[:-1]
select_subjects = st.sidebar.multiselect('Subject',sort_columns,sort_columns)

st.markdown("# Our Data Set")

totalStudent, totalColumns = df.shape
numberStudent=df.count().reset_index()
numberStudent.columns=['Subject','Number of student']
st.write(numberStudent)
def count_number_student(subject):
    students_num=df[subject].count()
    keys=['Total_student',f"Student in {subject}"]
    values=[totalStudent,students_num]
    number_Student = dict(zip(keys, values))
    number_Student = pd.DataFrame.from_dict(number_Student,orient='index')
    number_Student=number_Student.reset_index()
    number_Student.columns=['Types','Numbers of student']
    return number_Student
    
def visualize_number_student(subject):
    temp_dict= count_number_student(subject)
    fig=px.bar(temp_dict,x='Types',y='Numbers of student',color="Types", text="Types")
    return fig

def count_percent(subject):
    percent=df[subject].count()*100/totalStudent
    return round(float(percent),2)

for subject in select_subjects:
    if subject != 'Ma_mon_ngoai_ngu':
        st.write(f"Compare students are participated on {subject} with total student")
        st.write(f'Student in {subject} {count_percent({subject})} % in total students')
        st.write(visualize_number_student(subject))