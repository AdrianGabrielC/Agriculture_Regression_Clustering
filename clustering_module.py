import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import umap
from umap import UMAP
import plotly.express as px

def visualize_3d(dataframe):
    df = dataframe.copy()
    features = df.iloc[:,1:]
    umap_3d = UMAP(n_components=3, init='random', random_state=5)
    proj_3d = umap_3d.fit_transform(features)
    fig_3d = px.scatter_3d(proj_3d, x=0, y=1, z=2, color=df.Cultivar,
                           labels={'color': 'Cultivar'})
    fig_3d.update_layout(width=1000, height=800)
    fig_3d.show()

def vizualize_3d_clusters(dataframe):
    df = dataframe.copy()
    features = df.iloc[:,1:]
    umap_3d = UMAP(n_components=3, init='random', random_state=5)
    proj_3d = umap_3d.fit_transform(features)
    fig_3d = px.scatter_3d(proj_3d, x=0, y=1, z=2, color=df.cluster, labels={'color': 'cluster'})
    fig_3d.update_layout(width=1000, height=800)
    fig_3d.show()

def kmeans(dataframe):
    km = KMeans(n_clusters=3, init='k-means++', max_iter=1000, random_state=5)
    features = dataframe.iloc[:,1:]
    umap_3d = UMAP(n_components=3, init='random', random_state=42)
    proj_3d = umap_3d.fit_transform(features)
    predict = km.fit_predict(proj_3d)
    df = dataframe.copy()
    df['cluster'] = predict
    return df