from Outlier import variable_classification
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.figure_factory import create_distplot
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns

def Tunning(data):
    # variable classification
    variable_class=variable_classification(data)
    continious=variable_class[0]
    discreate=variable_class[1]
    categorical=variable_class[2]
    col1, col2= st.columns([2,2])
    with col1:
        Selected_segment=st.selectbox('Select segment',list(np.sort(data.Segment.unique())))
    with col2:
        Selected_variable=st.selectbox('Select the variable for Tunning',['']+continious+discreate)
    
    plot=st.radio('Select_plot',['','Inter Quartile Range','Sigma select','Jump Percentile'])
    if plot=='Inter Quartile Range':
        st.info('Inter Quartile Range')
        Q1=data[data['Segment']==Selected_segment][str(Selected_variable)].quantile(0.25)
        Q3=data[data['Segment']==Selected_segment][str(Selected_variable)].quantile(0.75)
        IQR=Q3-Q1
        Median_Outlier=Q3+(1.5*(IQR))
        Extream_Outlier=Q3+(3*(IQR))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[Median_Outlier,Median_Outlier], y=[-0.1,0.1], mode="lines", name="Median Outlier"))
        fig.add_trace(go.Scatter(x=[Extream_Outlier,Extream_Outlier], y=[-0.1,0.1], mode="lines", name="Extreme Outlier"))
        fig.add_trace(go.Box(x=data[data['Segment']==Selected_segment][str(Selected_variable)],name=0,showlegend=False))
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(title="BOXPLOT",xaxis_title=str(Selected_variable),
                          font=dict(family="Courier New, monospace", size=18,color="#7f7f7f"))
        st.plotly_chart(fig,use_container_width=True)
        
    elif plot=='Sigma select': 
        st.info('Sigma select')
        x,y = sns.kdeplot(data[data['Segment']==Selected_segment][str(Selected_variable)]).get_lines()[0].get_data()
        D=pd.DataFrame()
        D['X']=x
        D['Y']=y
        fig = px.area(D, x="X", y="Y")
        fig.add_trace(go.Scatter(x=[data[data['Segment']==Selected_segment][str(Selected_variable)].mean(),
                                    data[data['Segment']==Selected_segment][str(Selected_variable)].mean()],
                                  y=[0,D['Y'].max()],
                                 mode="lines", name="mean"))
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(title="KDE",xaxis_title=str(Selected_variable),yaxis_title="density",
                          font=dict(family="Courier New, monospace", size=18,color="#7f7f7f"))
        st.plotly_chart(fig, use_container_width=True)
    elif plot=='Jump Percentile':   
        st.info('Jump Percentile')
        number=st.number_input('Number of bins', 1, 100)
        fig = px.histogram(data[data['Segment']==Selected_segment], x=str(Selected_variable),nbins=number)
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
