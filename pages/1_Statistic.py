from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(
    page_title="Statistic"
)
#Introduction
img = Image.open('images/Statistic.png')
st.image(img,use_column_width=True)
st.title("Statistic")
st.markdown( """
   On this page there are some graphs that statistics the data and from which some conclusions are drawn from the data
    """
)
#
@st.cache
def load_data(year):
    path_file='diemthi'+str(year)+'.csv'
    temp = pd.read_csv(path_file)
    df = pd.DataFrame(temp)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df.round(decimals=2)



def Mean_each_Subject(Data):
    KQ=[]
    test=Data[["Toan"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["Dia"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["GDCD"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["Hoa"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["Li"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["Ngoai_ngu"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["Sinh"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["Su"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    test=Data[["Van"]]
    test=test.dropna()
    KQ.append(round(test.mean().values[0],3))

    KQ1={"Subject": ["Toan","Dia","GDCD","Hoa","Li","Ngoai_ngu","Sinh","Su","Van"],
        "Medium score":KQ 
    }
    R=pd.DataFrame(KQ1)
    return R


def Pecent_St_Sub(Data):
    k=Data
    k=k.drop("Ma_mon_ngoai_ngu",axis=1)
    k=k.groupby(by=["sbd"]).count()
    k=k.reset_index()
    k=k.groupby(by=["sbd"]).sum()
    k=k.sum(axis=1)
    k=pd.DataFrame(k)
    a=k.columns
    kq=np.array(k.values)
    mang ,count_a=np.unique(kq,return_counts=True)
    KQ1=[]
    for i in count_a:
        x=(i/len(kq)) * 100
        KQ1.append(round(x,5))
                
    KQ={"Number_sub":mang,
                "Percent":KQ1}
            
    KQ=pd.DataFrame(KQ)
    KQ.Number_sub=KQ.Number_sub.apply(lambda x: str(x) + " "+ 'Subject')
    return KQ

KHTN=["Sinh","Li","Hoa"]
KHXH=["Su","Dia","GDCD"]

def Count_Number_student(Data):
            Count_KHTN=Data[KHTN].dropna(axis=0)
            Count_KHXH=Data[KHXH].dropna(axis=0)
            textw={"Subject":["KHTN","KHXH"],
                'Number of students taking the exam':[len(Count_KHTN.index),len(Count_KHXH.index)]
                }
            s=pd.DataFrame(textw)
            Number_HS=Data
            Number_HS= Number_HS.drop("Ma_mon_ngoai_ngu",axis=1)
            Number_HS= Number_HS.drop("sbd",axis=1)
            Number_HS= Number_HS.notna().sum()
            Number_HS=pd.DataFrame(Number_HS)
            a=Number_HS.columns
            Number_HS=Number_HS.rename(columns={a[0]:'Number of students taking the exam'})
            Number_HS=Number_HS.reset_index().rename(columns={"index": "Subject"})
            Number_HS1=pd.concat([Number_HS,s],axis=0)
            Number_HS1=Number_HS1.reset_index().drop(columns="index")
            return Number_HS1


def Sutdent_PT_ND(Data):
    Diem_Thi=Data
    Diem_Thi_PT=Diem_Thi[Diem_Thi.sbd>=15000000]
    Diem_Thi_PT=Diem_Thi_PT[Diem_Thi.sbd<16000000]

    Diem_Thi_ND=Data[Data.sbd>25000000]
    Diem_Thi_ND=Diem_Thi_ND[Data.sbd<26000000]
    fig = go.Figure(data=[
         go.Bar(name='Phu Tho', x=["Number of contestants"], y=[Diem_Thi_PT.shape[0]]),
        go.Bar(name='Nam Dinh', x=["Number of contestants"], y=[Diem_Thi_ND.shape[0]])
    ])
    fig.update_layout(barmode='stack',height=500,width=500)
    st.plotly_chart(fig, theme=None)
    


def Student_over_five(Data):
    Diem_Thi=Data
    Diem_Thi_PT=Diem_Thi[Diem_Thi.sbd>=15000000]
    Diem_Thi_PT=Diem_Thi_PT[Diem_Thi.sbd<16000000]

    Diem_Thi_ND=Data[Data.sbd>25000000]
    Diem_Thi_ND=Diem_Thi_ND[Data.sbd<26000000]

    nume_col_list = list(Data.select_dtypes(include='float64'))
    nume_df_PT= Diem_Thi_PT[nume_col_list]
    nume_df_ND=Diem_Thi_ND[nume_col_list]

    scores_PT = pd.DataFrame([nume_df_PT[ nume_df_PT > 5].count(),], index=["above average"])
    scores_PT=scores_PT.T
    scores_ND = pd.DataFrame([nume_df_ND[nume_df_ND > 5].count(),], index=["above average"])
    scores_ND=scores_ND.T
    
   
    scores_PT=scores_PT.apply(lambda x: (x/ Diem_Thi_PT.shape[0])*100)
    scores_ND=scores_ND.apply(lambda x: (x/Diem_Thi_ND.shape[0])*100)


    fig = go.Figure(data=[
        go.Bar(name='Phu Tho', x=scores_PT.index.array, y=scores_PT["above average"].array),
        go.Bar(name='Nam Dinh', x=scores_ND.index.array, y=scores_ND["above average"].array)
                 ])
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, theme=None,use_container_width=True)

#sidebar
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year',list(reversed(range(2019,2021))))
#repare data

df=load_data(selected_year)
Number_HS1=Count_Number_student(df)
c=alt.Chart(Number_HS1).mark_bar().encode( y='Number of students taking the exam',
x= alt.X('Subject:N',axis=alt.Axis(labelAngle=0))).properties(
                    height=500
                )
a=c.interactive()
            # text=c.mark_text(dx=3,align='left').encode(text='Number of students taking the exam')
st.subheader("Number of candidates taking the exam of the subjects")
st.altair_chart(c+c.mark_text(dy=-10,align="center",color="White").encode(text='Number of students taking the exam')+a,use_container_width=True)
st.markdown(
        """
        ***Conclusion from the data***:
         -  Because math, literature, and English are compulsory subjects, the number of candidates taking the exam is very large
         -  In general, the number of candidates taking the KHXH exam is more than the KHTN exam. Maybe because most of the candidates choose KH just to consider graduation.


        """
    )
Mean_each_Sub=Mean_each_Subject(df)
st.subheader("Average score of each subject")
c=alt.Chart(Mean_each_Sub).mark_bar().encode( y='Medium score',
x= alt.X('Subject',axis=alt.Axis(labelAngle=0))).properties(
             height=500
        )
a=c.interactive()
st.altair_chart(c+c.mark_text(dy=-10,align="center",color="White").encode(text='Medium score')+a,use_container_width=True)
st.markdown("""***Conclusion from the data***: Foreign language subject has the lowest average score of all subjects
                     => Foreign language subject is still an issue that needs attention by the education industry""")
Pr=Pecent_St_Sub(df)
fig = px.pie(Pr, values='Percent', names='Number_sub', title='Number of students taking x subjects')
st.subheader("Number of candidates taking x subjects")
st.plotly_chart(fig, theme=None, use_container_width=True)
st.markdown("""***Conclusion from the data***: Most of the candidates choose to take 6 subjects, so this percentage is the highest. As for the candidates who take the exam under 6 subjects, most of them are re-examination of the university  """)
st.subheader("Comparison of university entrance exam statistics of the two provinces of Nam Dinh and Phu Tho")
st.markdown("""#### - Number of contestants """)
Sutdent_PT_ND(df)
st.markdown("""#### - Number of candidates with scores above the average(over 5 points) of the subjects """)
Student_over_five(df)
st.markdown("""***Conclusion from the data***: Looking at the chart, we can see that, in terms of calculation subjects, 
                most students in Nam Dinh tend to do better in school than in Phu Tho. And in social subjects, 
                students in Phu Tho are better than in Nam Dinh.""")   



