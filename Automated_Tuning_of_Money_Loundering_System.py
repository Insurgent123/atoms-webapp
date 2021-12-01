import streamlit as st
import pandas as pd
from data_import import Outlier_data_import
from eda import exploratory_data_analysis
from segmentation import data_preprocessing,get_columns,Elbow_method,var_select,clustering
from segmentation import eval_metrics
from Outlier import Outlier_assesment_module,variable_classification
import pickle as pkl
import pandas as pd
from Initial_tuning import Tunning
def Automated_Tuning_page():
    """ Analytics Page """    
    # Process Flow
    select = st.sidebar.selectbox('Select',['','Process Flow','Data Import','Exploratory Data Analysis',"Segmentation",
                                                  'Outlier Treatment','Initial Tuning','Sampling','Sample Generation'])
    data=pd.DataFrame()
    if select=='Process Flow':
        select1=st.expander('Process Flow')
        with select1:
            col1, col2, col3 = st.columns([1,8,1])
            with col2:
                st.image('images/processFlow.png', width=750)

    # Data Import
    elif select=='Data Import':
        select1=st.expander('Data Import')
        with select1:
            st.info('Data Import')
            data=Outlier_data_import()
            if not data.empty:
                st.success('Sucessfully import the data')
            pkl.dump(data, open('data/data.pkl', 'wb'))
            if 'Segment' not in list(data.columns):
                Freez=pd.DataFrame()
                Freez['Variable_name']=''
                Freez['Lower_limit']=''
                Freez['Upper_limit']=''
                pkl.dump(Freez, open('data/Freez.pkl', 'wb'))
            else:
                Freez=pd.DataFrame()
                Freez['Variable_name']=''
                Freez['Segment']=''
                Freez['Lower_limit']=''
                Freez['Upper_limit']=''
                pkl.dump(Freez, open('data/Freez.pkl', 'wb'))
                
    # Exploratory Data Analysis
    elif select=='Exploratory Data Analysis':
        select2=st.expander('Exploratory Data Analysis')
        with select2:
            with open("data/data.pkl", "rb") as f:
                object = pkl.load(f)

            data = pd.DataFrame(object)
            data.to_csv(r'file.csv')
            st.info('Exploratory Data Analysis')
            if data.empty:
                st.warning('please upload data')
            else:
                exploratory_data_analysis(data)

    # Segmentation Model
    elif select == "Segmentation":
        select_s = st.expander("Segmentation model")
        with select_s:
            with open("data/data.pkl", "rb") as f:
                object = pkl.load(f)

            seg = pd.DataFrame(object)
            seg.to_csv(r'file.csv')
            st.info('Segmentation')
            if seg.empty:
                st.warning('please upload data')
            else:
                # Preprocessing the data
                st.markdown("### Preprocessed Data for Segmentation")
                seg = data_preprocessing(seg)
                st.dataframe(seg)

                # Selecting the variables to cluster
                st.markdown("### Feature Selection for Segmentation")
                var_seg = var_select(seg)
                seg_data = seg[var_seg]

                # Elbow method
                if len(var_seg) >= 2:
                    col1, col2,col3 = st.columns([2,8,2])
                    with col2:
                        k_val = st.selectbox("Method to determine number of clusters",("","Elbow method"))
                    if k_val == "Elbow method":
                        n_clusters = Elbow_method(seg_data)
                        st.write(n_clusters)

                        if n_clusters is not None:
                            st.markdown("### Select the clustering algorithm")
                            model = clustering(seg_data,n_clusters,len(var_seg))
                            if model:
                                eval_metrics(seg_data,model)














    
    # Outlier Treatment
    elif select=='Outlier Treatment':
        select3=st.expander('Outlier Treatment')
        with select3:
            st.info('Outlier Treatment')
            
            # Importing data
            with open("data/data.pkl", "rb") as f:
                object = pkl.load(f)

            data = pd.DataFrame(object)
            data.to_csv(r'file.csv')
            
            # Importing Freez dataframe
            with open("data/Freez.pkl", "rb") as f:
                object = pkl.load(f)

            Freez = pd.DataFrame(object)
            Freez.to_csv(r'file.csv')
            if data.empty:
                st.warning('please upload data')
            else:
                Outlier_assesment_module(data,Freez)
    
    # Tunning
    elif select=='Initial Tuning':
        select4=st.expander('Initial Tuning')
        with select4:
            st.info('Initial Tuning')
            with open("data/data.pkl", "rb") as f:
                object = pkl.load(f)

            data = pd.DataFrame(object)
            data.to_csv(r'file.csv')
            # Importing Freez dataframe
            with open("data/Freez.pkl", "rb") as f:
                object = pkl.load(f)

            Freez = pd.DataFrame(object)
            Freez.to_csv(r'file.csv')
            Tunning(data)
    
    # Sampling
    elif select=='Sampling':
        select5=st.expander('Sampling')
        with select5:
            st.info('Sampling')  
        
    # Sample Generation    
    elif select=='Sample Generation':
        select6=st.expander('Sample Generation')
        with select6:
            st.info('Sample Generation')
            clicked = st.button('Sample Generation')
