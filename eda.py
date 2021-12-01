import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
from Outlier import variable_classification
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Graph plotter
def EDA(Selected_variable,Selected_graph_type,data):
    if Selected_graph_type=='' and Selected_variable=='':
        st.warning('Please select the plot type and variable')
    elif Selected_variable=='':
        st.warning('Please select the variable')
    elif Selected_graph_type=='':
        st.warning('Please select the plot type')
    elif Selected_graph_type=='KDE':
        st.info('KDE')
        x,y = sns.kdeplot(data[str(Selected_variable)]).get_lines()[0].get_data()
        D=pd.DataFrame()
        D['X']=x
        D['Y']=y
        fig = px.area(D, x="X", y="Y")
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(title="KDE",xaxis_title=str(Selected_variable),yaxis_title="density",
                          font=dict(family="Courier New, monospace", size=18,color="#7f7f7f"))
        st.plotly_chart(fig, use_container_width=True)
    elif Selected_graph_type=='Histogram':
        st.info('Histogram')
        fig = px.histogram(data, x=str(Selected_variable))
        st.plotly_chart(fig, use_container_width=True)
    elif Selected_graph_type=='Boxplot': 
        st.info('Boxplot')
        fig = px.box(data, y=str(Selected_variable))
        st.plotly_chart(fig, use_container_width=True)
    elif Selected_graph_type=='Countplot':
        st.info('Countplot')
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = (5,4)
        sns.countplot(data=data,x=str(Selected_variable))
        st.pyplot(fig)
    return ''
 
# summary
def summary(data,continious,discreate,categorical):
    Total_count=data.shape[0]
    TOTAL_DISTINCT_SEGMENTS=''
    TOTAL_DISTINCT_RECORDS=''
    TOTAL_RISK_LEVELS=''
    if 'Customer ID' not in list(data.columns):
        TOTAL_DISTINCT_RECORDS="Nan"
    else:
        TOTAL_DISTINCT_RECORDS=len(list(data['Customer ID'].unique()))
    if 'Segment' not in list(data.columns):
        TOTAL_DISTINCT_SEGMENTS="Nan"
    else:
        TOTAL_DISTINCT_SEGMENTS=len(list(data.Segment.unique()))
    if 'Overall_risk' not in list(data.columns):
        TOTAL_RISK_LEVELS="Nan"
    else:
        TOTAL_RISK_LEVELS=len(list(data['Overall_risk'].unique()))
    summary=pd.DataFrame()    
    summary={'TOTAL DISTINCT RECORDS':[TOTAL_DISTINCT_RECORDS],'TOTAL DISTINCT SEGMENTS':[TOTAL_DISTINCT_SEGMENTS],
         'TOTAL RISK LEVELS':[TOTAL_RISK_LEVELS],'CATEGORICAL VARIABLES':[categorical],'DISCRETE VARIABLES':[discreate],
         'CONTINOUS VARIABLES':[continious]}
    return summary

#EDA
def exploratory_data_analysis(data):
    """ Generates detailed exploratory data analysis"""
    # variable classification
    variable_class=variable_classification(data)
    continious=variable_class[0]
    discreate=variable_class[1]
    categorical=variable_class[2]
    # Information of whole dataset
    st.success('Summary of Whole Dataset')
    st.info('Summary')
    info=pd.DataFrame(summary(data,continious,discreate,categorical))
    st.write(info)
    st.info('First five rows')
    st.dataframe(data.head(6))
    st.info('Last five rows')
    st.dataframe(data.tail(6))
    # Full and Segment wise EDA  
    ## If segment is not form
    if 'Segment' not in list(data.columns):
        st.info('Plots')
        col1, col2 = st.columns([2,2])
        with col1:
            Selected_variable=st.selectbox('Select the variable for analysis',['']+continious+discreate+categorical)
        with col2:
            if Selected_variable=='':
                Selected_graph_type=''
                st.warning('Please select the variable')
            elif Selected_variable in continious:
                Selected_graph_type=st.selectbox('select the plot type for continious variable',['','KDE','Boxplot'])
            elif Selected_variable in discreate:
                Selected_graph_type=st.selectbox('select the plot type for discreate variable',['','Histogram'])
            elif Selected_variable in categorical:
                Selected_graph_type=st.selectbox('select the plot type for categorical variable',
                                                 ['','Histogram','Countplot'])
        clicked =st.button('Plot')
        if clicked:
            EDA(Selected_variable,Selected_graph_type,data)
    else:
        ## If segment is form
        Selected_type=st.selectbox('Select Type of EDA',['','Segmentwise_EDA','Full_EDA'])
        # Full EDA  
        if Selected_type=='Full_EDA':
            st.info('Plots')
            col1, col2 = st.columns([2,2])
            with col1:
                Selected_variable=st.selectbox('Select the variable for analysis',['']+continious+discreate+categorical)
            with col2:
                if Selected_variable=='':
                    Selected_graph_type=''
                    st.warning('Please select the variable')
                elif Selected_variable in continious:
                    Selected_graph_type=st.selectbox('select the plot type for continious variable',['','KDE','Boxplot'])
                elif Selected_variable in discreate:
                    Selected_graph_type=st.selectbox('select the plot type for discreate variable',['','Histogram'])
                elif Selected_variable in categorical:
                    Selected_graph_type=st.selectbox('select the plot type for categorical variable',['','Histogram',
                                                                                                      'Countplot'])
            clicked =st.button('Plot')
            if clicked:
                EDA(Selected_variable,Selected_graph_type,data)
        # Segment wise EDA  
        if Selected_type=='Segmentwise_EDA':
            st.info('Plots')
            col1, col2,col3 = st.columns([3,3,3])
            with col1:
                Selected_segment=st.multiselect('Select segment',list(np.sort(data.Segment.unique())))
            with col2:
                Selected_variable=st.selectbox('Select the variable for analysis',['']+continious+discreate+categorical)
            with col3:
                if Selected_variable=='':
                    Selected_graph_type=''
                    st.warning('Please select the variable')
                elif Selected_variable in continious:
                    Selected_graph_type=st.selectbox('select the plot type for continious variable',['','KDE','Boxplot'])
                elif Selected_variable in discreate:
                    Selected_graph_type=st.selectbox('select the plot type for discreate variable',['','Histogram'])
                elif Selected_variable in categorical:
                    Selected_graph_type=st.selectbox('select the plot type for categorical variable',['','Histogram',
                                                                                                      'Countplot'])
            clicked =st.button('Plot')
            if clicked:
                for i in Selected_segment:
                    st.success(i)
                    col1, col2= st.columns([2,2])
                    with col1:
                        EDA(Selected_variable,Selected_graph_type,data[data['Segment']==i])
                    with col2:
                        st.info('Summary Statistics')
                        st.dataframe(data[data['Segment']==i].describe()[Selected_variable])
                
