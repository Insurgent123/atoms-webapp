import streamlit as st
import pandas as pd
import pickle
def data_importer():
    data=pd.DataFrame()
    col1, col2, col3 = st.columns([3,3,3])
    with col2:
        data_file = st.file_uploader("Choose a file to import Development data", 
                                             type='csv')

        if data_file is not None:
            data = pd.read_csv(data_file)
    return data
                
def Outlier_data_import():
    with st.container():
        data=data_importer()
    return data
