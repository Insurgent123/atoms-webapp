import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

def variable_classification(data):
    continious=[]
    discreate=[]
    categorical=[]
    for i in data.columns:
        k=str(data.dtypes[str(i)])
        if (data[str(i)].unique().size<=10) and (k==(('int64') or ('int32'))):
            categorical.append(i)
        if (data[str(i)].unique().size>10) and (k==(('int64') or ('int32'))):
            discreate.append(i)
        if k==(('float64') or ('float32')):
            continious.append(i)
    return continious,discreate,categorical
 
    
def Outlier_detection(data,data2,variable_name,continious,discreate):
    col1, col2 = st.columns([2,2])
    with col1:
        st.info('Summary statistics before outlier treatment')
        st.dataframe(data.describe()[variable_name])
    with col2:
        st.info('Summary statistics after outlier treatment')
        st.dataframe(data2.describe()[variable_name])
        
    col1, col2 = st.columns([2,2])
    with col1:
        if variable_name in continious:
            st.info('Boxplot before outlier treatment')
            fig1 = px.box(data, y=str(variable_name))
            st.plotly_chart(fig1, use_container_width=True)
            m,n = sns.kdeplot(x=str(variable_name),data=data).get_lines()[0].get_data()
            D=pd.DataFrame()
            D['X']=m
            D['Y']=n
            st.write(D.describe())
            fig2 = px.area(D, x="X", y="Y")
            fig2.update_yaxes(showticklabels=True)
            fig2.update_layout(title="KDE",xaxis_title=str(variable_name),yaxis_title="density",
                              font=dict(family="Courier New, monospace", size=18,color="#7f7f7f"))
            st.plotly_chart(fig2, use_container_width=True)
        elif variable_name in discreate:
            fig = px.histogram(data, x=str(variable_name))
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if variable_name in continious:
            st.info('Boxplot after outlier treatment')
            fig1 = px.box(data2, y=str(variable_name))
            st.plotly_chart(fig1, use_container_width=True)
            m,n = sns.kdeplot(x=str(variable_name),data=data2).get_lines()[0].get_data()
            D1=pd.DataFrame()
            D1['X']=m
            D1['Y']=n
            st.write(D1.describe())
            fig2 = px.area(D1,x="X", y="Y")
            fig2.update_yaxes(showticklabels=True)
            fig2.update_layout(title="KDE",xaxis_title=str(variable_name),yaxis_title="density",
                              font=dict(family="Courier New, monospace", size=18,color="#7f7f7f"))
            st.plotly_chart(fig2, use_container_width=True)
        elif variable_name in discreate:
            fig = px.histogram(data2, x=str(variable_name))
            st.plotly_chart(fig, use_container_width=True)
    return ''


def Outlier_assesment_module(data,Freez):
    variable_class=variable_classification(data)
    continious=variable_class[0]
    discreate=variable_class[1]
    categorical=variable_class[2]
    variable_name=''
    Selected_segment=''
    if 'Segment' not in list(data.columns):
        variable_name=st.selectbox('Select a variable for outlier treatment',continious+discreate)
    else:
        col1, col2 = st.columns([2,2])
        with col1:
            Selected_segment=st.selectbox('Select segment for outlier treatment',list(np.sort(data.Segment.unique())))
        with col2:
            variable_name=st.selectbox('Select a variable for outlier treatment',continious+discreate)
    col1, col2 = st.columns([2,2])
    with col1:
        lower_percentile=st.slider('select a below unacceptable percentile value', 0, 50)
    with col2:
        upper_percentile=st.slider('select a above unacceptable percentile value', 0, 50)
        upper_percentile=100-upper_percentile
        
    lower=data[variable_name].quantile(lower_percentile/100)
    upper=data[variable_name].quantile(upper_percentile/100)
    data1=data[data[variable_name]>=lower]
    data2=data1[data1[variable_name]<=upper]
    clicked = st.button('Compute Statistics')
    if clicked:
        if 'Segment' not in list(data.columns):
                Outlier_detection(data,data2,variable_name,continious,discreate)
        else:
            Outlier_detection(data[data['Segment']==Selected_segment],data2[data2['Segment']==Selected_segment],
                              variable_name,continious,discreate)
    clicked = st.button('Freez')
    if clicked:
        if 'Segment' not in list(data.columns):
            Freez.loc[len(Freez.index)] = [variable_name, lower, upper] 
            pkl.dump(Freez, open('data/Freez.pkl', 'wb'))
        else:
            Freez.loc[len(Freez.index)] = [variable_name,Selected_segment, lower, upper] 
            pkl.dump(Freez, open('data/Freez.pkl', 'wb'))
