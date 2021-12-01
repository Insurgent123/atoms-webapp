import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st
import plotly.graph_objs as go
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as pgo
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

def get_columns(data):
    return data.columns

def data_preprocessing(data):
    categorical_features=[feature for feature in data.columns if data[feature].dtypes=='O']
    for feature in categorical_features:
        df_onehot = pd.get_dummies(data[feature],prefix = feature)
        data = pd.concat([data,df_onehot],axis = 1)
        data.drop(feature,axis = 1,inplace = True)

    return data

def var_select(data):
    col1, col2, col3 = st.columns([2,10,2])
    with col2:
        values = st.multiselect("Select the features to cluster",(data.columns))
        seg_data = data[values]
        if len(values) != 0:
            st.write(seg_data)
    return values

def Elbow_method(x):
    col1, col2, col3 = st.columns([6,6,6])
    with col1:
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2,12),metric = "distortion")
        visualizer.fit(x)        # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure
        st.pyplot()
        st.metric(label = "Recommended Number of clusters",value = str(visualizer.elbow_value_))

    with col2:
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2,12),metric = "silhouette")

        visualizer.fit(x)        # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure
        st.pyplot()
        st.metric(label = "Recommended Number of clusters",value = str(visualizer.elbow_value_))

    with col3:
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2,12),metric = "calinski_harabasz")
        visualizer.fit(x)        # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure
        st.pyplot()
        st.metric(label = "Recommended Number of clusters",value = str(visualizer.elbow_value_))


    #return visualizer.elbow_value_
    n_clusters = st.slider("Select number of clusters",1,9)

    if st.button("Freeze"):
        st.markdown(f" #### Number of clusters - {n_clusters}")
    else:
        pass

    return n_clusters


def twoD_k_means_clustering(x,n_clusters):
    model = KMeans(n_clusters = n_clusters, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    ymeans = model.fit_predict(x)

    label = ["Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7"]
    c = ["pink","orange","lightgreen","blue","red","brown"]

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.title(f'Cluster of {x.columns[0]}', fontsize = 30)

    for i in range(n_clusters):
        plt.scatter(x[ymeans == i].iloc[0], x[ymeans == i].iloc[1], s = 100,c =c[i], label = label[i] )
    plt.style.use('fivethirtyeight')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.legend()
    plt.grid()
    plt.show()
    st.pyplot()
    return model

def threeD_k_means_clustering(x,n_clusters):
    model = KMeans(n_clusters = n_clusters, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    ymeans = model.fit_predict(x)

    plt.rcParams['figure.figsize'] = (10, 10)
    Scene = dict(xaxis = dict(title  = x.columns[0]),yaxis = dict(title  = x.columns[1]),zaxis = dict(title  = x.columns[2]))

    labels = model.labels_
    trace = go.Scatter3d(x=x.iloc[:, 0], y=x.iloc[:, 1],z = x.iloc[:,2], mode='markers',
                         marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
    layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    fig.show()
    st.pyplot()
    return model

def spectral_clustering(x,n_clusters):
    col1, col2, col3 = st.columns([6,6,6])
    with col2:
        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                   assign_labels='kmeans')
        labels = model.fit_predict(x)
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1],c = labels,
                    s=50, cmap='viridis');
        plt.style.use('fivethirtyeight')
        plt.xlabel(x.columns[0])
        plt.ylabel(x.columns[1])
        plt.legend()
        plt.grid()
        st.pyplot()
        return model


def aggo_clustering(x,n_clusters):
    col1, col2, col3 = st.columns([3,8,3])
    with col2:
        model = AgglomerativeClustering(n_clusters = n_clusters)
        # Visualizing the clustering
        plt.figure(figsize =(6, 6))
        plt.scatter(x.iloc[:,0], x.iloc[:,1],
                   c = model.fit_predict(x), cmap ='rainbow')
        plt.style.use('fivethirtyeight')
        plt.xlabel(x.columns[0])
        plt.ylabel(x.columns[1])
        plt.legend()
        plt.grid()
        plt.show()
        st.pyplot()
        return model

def clustering(x,n_clusters,var):
    clust = st.selectbox("Choose the clustering algorithm",(" ","K Means Clustering","Spectral Clustering",
                                                    "Agglomerative Clustering"))
    if clust == "K Means Clustering":
        if var == 2:
            return twoD_k_means_clustering(x,n_clusters)
        elif var == 3:
            return threeD_k_means_clustering(x,n_clusters)
    if clust == "Spectral Clustering":
        return spectral_clustering(x,n_clusters)

    if clust == "Agglomerative Clustering":
        return aggo_clustering(x,n_clusters)

def silhouette_score(x,model):
    score = metrics.silhouette_score(x,model.labels_, metric='euclidean')
    st.write('Silhouetter Score: %.3f' % score)
    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    for i in [2, 3, 4, 5]:
        '''
        Create KMeans instance for different number of clusters
        '''

        kmeans_1 = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(kmeans_1, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(x)
        st.pyplot()

def silhouette_comp(x,model):
    k = [2, 3, 4, 5, 6]
    # Appending the silhouette scores of the different models to the list
    silhouette_scores = []
    silhouette_scores.append(
            silhouette_score(x, model.fit_predict(x)))
    silhouette_scores.append(
            silhouette_score(x, model.fit_predict(x)))
    silhouette_scores.append(
            silhouette_score(x, model.fit_predict(x)))
    silhouette_scores.append(
            silhouette_score(x, model.fit_predict(x)))
    silhouette_scores.append(
            silhouette_score(x, model.fit_predict(x)))

    # Plotting a bar graph to compare the results
    plt.bar(k, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('S(i)', fontsize = 20)
    plt.show()
    st.pyplot()

# davies_boulding_score
def davies_metrics(x,model):
    return st.metric(label = "Davies_Boulding_Score",value  = str(metrics.davies_bouldin_score(x,model.labels_)))

# calinski_harabasz_score
def calinski_metrics(x,model):
    return st.metric(label = "Calinski_Harabasz_Score",value  = str(metrics.calinski_harabasz_score(x, model.labels_)))

def eval_metrics(x,model):
    eval = st.selectbox("Select the performance metrics",("Select","Silhouette Score","Silhouette comparision",
                                                          "Rand Index","Adjusted Rand Index",
                                                   "Davies-Bouldin Index","Calinski-Harabasz Index"))
    if eval == "Silhouette Score":
        return silhouette_score(x,model)
    elif eval == "Silhouette comparision":
        return silhouette_comp(x,model)
    elif eval == "Rand Index":
        pass
    elif eval == "Adjusted Rand Index":
        pass
    elif eval ==  "Davies-Bouldin Index":
        return davies_metrics(x,model)
    elif eval ==  "Calinski-Harabasz Index":
        return calinski_metrics(x,model)







































