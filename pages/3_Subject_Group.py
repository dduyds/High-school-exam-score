import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Subject Group"
)
#Introduction
img = Image.open('images/Subject.png')
st.image(img,use_column_width=True)
st.title("Subject Group")
st.markdown( """
    On this page you can see spectrum and comments of each subject group and in year by choice.
    """
)
#
Groups = {'A00':['Toan','Li','Hoa']
        ,'B00':['Toan','Hoa','Sinh']
        ,'C00':['Van','Su','Dia']
        ,'D01':['Van','Toan','Ngoai_ngu','N1']}#

@st.cache
def load_data(year):
    path_file='diemthi'+str(year)+'.csv'
    temp = pd.read_csv(path_file)
    df = pd.DataFrame(temp)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df.round(decimals=2)

#sidebar
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year',list(reversed(range(2019,2021))))
selected_groups = st.sidebar.multiselect('Subject',Groups.keys(),Groups.keys())

#repare data
df = load_data(selected_year)

st.markdown("# Our Data Set")

# @st.cache
def visualize_spectrum(subject, df=df):
    plt.figure(figsize=(25,12))
    plt.title(f"Spectrum of {subject}")

    t = plt.hist(df[subject],bins=np.round(np.arange(0, 30.1, 3/5),1),rwidth=0.5)
    hist, edges = t[0],t[1]

    plt.xticks(edges)
    plt.yticks(np.arange(0, max(hist)+1, 1000))

    plt.xlabel('scores')
    plt.ylabel('number of students')

    return plt

# @st.cache
def get_df_group(groupName):
    subjects = Groups[groupName][:3]
    language = Groups[groupName][-1] #

    score_list = df[df['Ma_mon_ngoai_ngu'] == language] if language in df['Ma_mon_ngoai_ngu'] else df #
    score_list = score_list[subjects].dropna()

    score_list = pd.concat([score_list,score_list.sum(axis=1)], axis=1).reset_index().drop('index', axis=1)
    score_list.rename(columns = {0:f'Sum {groupName}'}, inplace = True)
    
    return score_list

for group in selected_groups:
    df_group = get_df_group(group)
    st.dataframe(df_group)
    st.write(f"Spectrum of {group}")
    st.pyplot(visualize_spectrum(f'Sum {group}', df_group))

    