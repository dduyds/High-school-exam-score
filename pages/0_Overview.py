import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Overview"
)
#Introduction
img = Image.open('images/Overview.jpg')
st.image(img,use_column_width=True)
st.title("Overview")
st.markdown( """
    On this page, We will explore to understand about the data set in year by choice.
    """
)

#sidebar
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year',list(reversed(range(2019,2021))))
#repare data
path_file='diemthi'+str(selected_year)+'.csv'
temp = pd.read_csv(path_file)
df = pd.DataFrame(temp).round(decimals=2)
df.drop(df.columns[[0]], axis=1, inplace=True)
#print df
st.markdown("# Our Data Set")
st.markdown(f"## Score of all subjects in year {selected_year}.")

st.dataframe(df)
st.markdown(
        """
        ### What do the columns in data mean?
        The dataset includes the student's identification number and the corresponding score of that student:

        sbd - Candidate number

        Van - Literature score

        Toan - Mathematics score

        Ma_mon_ngoai_ngu - Foreign language code

        Ngoai_ngu - Foreign language score

        - N1: English

        - N2: Russian

        - N3: French

        - N4: Chinese

        - N5: German

        - N6: Japanese

        Li - Physics score

        Hoa - Chemistry score

        Sinh - Biology score

        Su - History score

        Dia - Geography score

        GDCD - Civics score
    """
    )

#
st.markdown("# Data Explorations")
df_ex = pd.DataFrame([df.shape[0],df.shape[1],df.duplicated().sum()],
    index=['Rows','Columns','Duplicated Rows'], columns=['Quantity'])
st.dataframe(df_ex)


st.markdown("## Data Types")
dtypes=pd.DataFrame([df.dtypes])
st.dataframe(dtypes.astype(str))


st.markdown("## Numeric columns")
nume_col_list = list(df.select_dtypes(include='float64'))
nume_df = df[nume_col_list]
df1 = pd.DataFrame([
    nume_df.isna().mean() * 100]
    , index=["missing_ratio"])
df2 = df[nume_col_list].describe()
nume_col_profiles_df = np.round(pd.concat([df1, df2], axis=0),2)
st.dataframe(nume_col_profiles_df.astype(str))
#comment
st.markdown("""
    `Are they abnormal?`

    - The ratio of candidates taking the subjects of the social block is twice as high as the rate of the candidates taking the subjects of the natural block.
    - Got 0 in all subjects.
    - Maths and Literatures are the 2 subjects with the most candidates participating.
    - Literatures is the only subject that does not have a maximum score.
    """)


#
st.markdown("## Further distribution for Numeric columns")
nume_df = df[nume_col_list]
further_col_profiles_df = pd.DataFrame([
    nume_df[nume_df == 10].count(),
    nume_df[nume_df <= 1].count(),
    nume_df[nume_df < 5].count(),]
    , index=["10 scores","paralysis scores","below_avg"])
st.dataframe(further_col_profiles_df.astype(str))


#
st.markdown("## Categorical columns")
cate_df = df[['sbd', 'Ma_mon_ngoai_ngu']]
cate_col_profiles_df = pd.DataFrame([
    cate_df.isna().mean() * 100,
    cate_df.apply(lambda x: pd.unique(x.dropna()).size),
    cate_df.apply(lambda x: pd.unique(x.dropna()))]
    , index=["missing_ratio","num_diff_vals","diff_vals"])
st.dataframe(cate_col_profiles_df.astype(str))
#comment
st.markdown("""
    `Are they abnormal?`

    - The data missing ratio of "Ma Mon Ngoai Ngu" is equal to the data missing ratio of "Ngoai Ngu".
    - There are 6 types of foreign languages that the contestants participated in.
    """)


#
st.markdown("## Number of contestants participated in each 'Ma Mon Ngoai Ngu'")
num_each_NgoaiNgu = df[['sbd', 'Ma_mon_ngoai_ngu']].groupby('Ma_mon_ngoai_ngu').count()
num_each_NgoaiNgu.rename(columns = {'sbd':'Quantity'}, inplace = True)
st.dataframe(num_each_NgoaiNgu.astype(str))