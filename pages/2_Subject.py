import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
st.set_page_config(
    page_title="Subject"
)
#Introduction
img = Image.open('images/Subject.png')
st.image(img,use_column_width=True)
st.title("Subject")
st.markdown( """
    On this page, we can see spectrum and comments of each subject and in year by choice.

    Looking at the score spectrum, we can understand the student's learning situation of each year as well as the easy / difficult level of the high school exam that year. Partially assess the quality of education as well as the quality of students.

    The score spectrum also shows that this year's candidates are strong in which exams. In addition, through the score spectrum, we will also be able to assess whether that year's exam is standardized or not.

    Universities base on the score spectrum to develop appropriate benchmarks.
    """
)

@st.cache
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

select_point = st.sidebar.slider(
    'Select a range of point',
    0.0, 10.0,(0.0,10.0))
pointFrom, pointTo = select_point

def visualize_spectrum(subject):
    #histogram
    plt.figure(figsize=(25,12))
    plt.title(f"Spectrum of {subject}")

    t = plt.hist(df[subject],bins=np.round(np.arange(pointFrom, pointTo+0.1, 0.2),1),rwidth=0.5)
    hist, edges = t[0],t[1]

    plt.xticks(edges)
    plt.yticks(np.arange(0, max(hist)+1, 1000))

    plt.xlabel('scores')
    plt.ylabel('number of students')

    #pie chart for %
    t = np.histogram(df[subject],bins=np.round(np.arange(pointFrom, pointTo+0.1, 0.5),1))
    res = dict(map(lambda i,j : (i,j), t[1],t[0]))
    res = pd.DataFrame.from_dict(res,orient='index')
    res = res.reset_index()
    res.columns=['Points','Numbers of student']
    fig = px.pie(res,values='Numbers of student',names='Points')

    return plt, fig

for subject in select_subjects:
    if subject != 'Ma_mon_ngoai_ngu':
        st.write(f"Spectrum of {subject}")
        st.pyplot(visualize_spectrum(subject)[0])
        st.write(f"Point structure in range of {subject}")
        st.write(visualize_spectrum(subject)[1])