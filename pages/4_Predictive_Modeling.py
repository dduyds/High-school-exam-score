from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure

st.set_page_config(
    page_title = "Predictive Modeling"
)

# Introduction
img = Image.open('images/Model.jpg')
st.image(img,use_column_width=True)

st.title("Predictive Modeling")
st.markdown( """
    On this page you see that, we try to train a multi linear regression model for predict score of subject by another 
    subjects in group of that subject.
    """
)
#


# Function
@st.cache
def load_data(year):
    path_file='diemthi'+str(year)+'.csv'
    temp = pd.read_csv(path_file)
    df = pd.DataFrame(temp)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df.round(decimals=2)


@st.cache
def getDataByGroup(dataFrame,group,listGroup):
    if group == 'D01':
        label = predictGroup[group][-1]
        dataLabel = dataFrame.loc[(dataFrame['Ma_mon_ngoai_ngu'] == label)]
        newGroup = predictGroup[group][0:-1]
        data = dataLabel[newGroup]
        data = data.dropna()
    else:
        data = dataFrame[listGroup[group]]
        data = data.dropna()
        
    return data


@st.cache
def prepareDataForModel(dataFrame,subjectWantToPredict):
    dataField = dataFrame.drop([subjectWantToPredict],axis = 1)
    resultField = dataFrame[[subjectWantToPredict]]
    dataField = dataField.iloc[:,:].values
    resultField = resultField.iloc[:,:].values
    resultField = np.array(resultField).reshape(-1,1)
    dataField = np.array(dataField)

    return dataField, resultField


@st.cache
def buildLinearMultiRegressionModel(dataField,resultField):
    X = np.array(np.copy(dataField))
    y = np.array(np.copy(resultField))
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    scoreOnTrain = lr.score(X_train,y_train)
    y_preds = lr.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test,y_preds))

    return y_test, y_preds, scoreOnTrain, RMSE
    

@st.cache
def evaluationModel(scoreOnTrain,Rmse,subjectWantToPredict):
    evaluationModel = pd.DataFrame([subjectWantToPredict,scoreOnTrain, Rmse], columns = ['Result'], 
    index=['Subject want to predict', 'Evaluate the model on training set', 
    'Evaluate the model on testing predicting (RMSE)'])
    return evaluationModel


# @st.cache
def visualizationDataLine(y,yPreds,subjectWantToPredict):
    plt.figure(figsize=(7,5))
    plt.plot(y, y, color = 'red',label = 'Real data' )
    plt.scatter(y, yPreds, color = 'blue', label = 'Predict model')
    plt.title(f'Predict {subjectWantToPredict} score.')
    plt.xlabel('Score')
    plt.ylabel('Score')
    plt.legend()

    return plt
    


# @st.cache
def visualizationDataHis(y,yPreds,subjectWantToPredict):
    plt.figure(figsize=(10,10))
    plt.figure(figsize=(7,5))
    plt.hist(y, bins = 20,color = 'red') 
    plt.hist(yPreds, bins = 20, color = 'blue') 
    plt.title(f'Predict {subjectWantToPredict} score.')
    plt.xlabel('Score')
    plt.ylabel('Total')
    return plt

#
predictGroup = {'A00':['Toan','Li','Hoa']
        ,'B00':['Toan','Hoa','Sinh']
        ,'C00':['Van','Su','Dia']
        ,'D01':['Van','Toan','Ngoai_ngu','N1']}#
#


#sidebar
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year for predicting',list(reversed(range(2019,2021))))
selected_group = st.sidebar.selectbox('Group for predicting',predictGroup.keys())
if selected_group == 'D01':
    predictSubject = predictGroup[selected_group][:-1]
else:
    predictSubject = predictGroup[selected_group]
selected_subject = st.sidebar.selectbox('Subject for predicting',predictSubject)
#

df = load_data(selected_year)

groupWantToPredict = selected_group
dataForModel = getDataByGroup(df,groupWantToPredict,predictGroup)
st.markdown(f"#### Score of group {selected_group}.")
st.dataframe(dataForModel)

subjectWantToPredict = selected_subject
dataField, resultField = prepareDataForModel(dataForModel,subjectWantToPredict)
y_test, y_preds, scoreOnTrain, RMSE = buildLinearMultiRegressionModel(dataField,resultField)
eva = evaluationModel(scoreOnTrain, RMSE,subjectWantToPredict)
eva = eva.astype(str)
st.markdown(f"##### Evaluate multi linear regression model to predict {subjectWantToPredict} score by using another subjects in group {groupWantToPredict}.")
st.dataframe(eva)

st.markdown(f"##### Visualize the result of predicting {subjectWantToPredict} score.")
st.pyplot(visualizationDataLine(y_test,y_preds,subjectWantToPredict))
st.pyplot(visualizationDataHis(y_test,y_preds,subjectWantToPredict)) 