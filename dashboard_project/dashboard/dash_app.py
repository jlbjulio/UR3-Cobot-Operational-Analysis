import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and preprocess the dataset
dataset_path = "/home/gtfojulio/dataset.xlsx"
df = pd.read_excel(dataset_path)

# Data Preprocessing (as per your provided steps)
df['grip_lost'] = df['grip_lost'].astype(float)
df['Robot_ProtectiveStop'] = df['Robot_ProtectiveStop'].astype(float)
df.dropna(inplace=True)
df.drop(columns=['Timestamp', 'Num'], inplace=True)
columns_to_normalize = df.columns.difference(['grip_lost', 'Robot_ProtectiveStop'])

Q1 = df[columns_to_normalize].quantile(0.25)
Q3 = df[columns_to_normalize].quantile(0.75)
IQR = Q3 - Q1

for col in columns_to_normalize:
    lower_bound = Q1[col] - 1.5 * IQR[col]
    upper_bound = Q3[col] + 1.5 * IQR[col]
    df[col] = df[col].apply(lambda x: max(min(x, upper_bound), lower_bound))

df[columns_to_normalize] += 1 - df[columns_to_normalize].min()

scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

pt = PowerTransformer(method='yeo-johnson')
df[columns_to_normalize] = pt.fit_transform(df[columns_to_normalize])

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(df)

# Clustering (K-Means, Hierarchical, DBSCAN)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_pca)
df['KMeans_Cluster'] = kmeans.labels_

# Hierarchical Clustering
best_silhouette = -1
best_params = None
best_model = None

param_grid = {'n_clusters': list(range(2, 11)), 'linkage': ['ward', 'complete', 'average', 'single']}

for linkage_method in param_grid['linkage']:
    for n_clusters in param_grid['n_clusters']:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(data_pca)
        silhouette_avg = silhouette_score(data_pca, labels)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_params = {'n_clusters': n_clusters, 'linkage': linkage_method}
            best_model = model

df['Hierarchical_Cluster'] = best_model.labels_

# DBSCAN
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(data_pca)
distances, indices = neighbors_fit.kneighbors(data_pca)
distances = np.sort(distances[:, 4], axis=0)

epsilon = 0.05
dbscan = DBSCAN(eps=epsilon, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_pca)
df['DBSCAN_Cluster'] = dbscan_labels

# Create lists of columns
temp_cols = [col for col in ['Temperature_T0', 'Temperature_J1', 'Temperature_J2', 'Temperature_J3', 'Temperature_J4', 'Temperature_J5'] if col in df.columns]
current_cols = [col for col in ['Current_J0', 'Current_J1', 'Current_J2', 'Current_J3', 'Current_J4', 'Current_J5'] if col in df.columns]

# Define the layout
app.layout = html.Div([
    html.H1('Dashboard de Comparación'),
    dcc.Dropdown(
        id='variable-dropdown',
        options=[
            {'label': 'Temperaturas', 'value': 'temp'},
            {'label': 'Corrientes', 'value': 'current'},
        ],
        value='temp'
    ),
    dcc.Graph(id='time-series-graph'),
    dcc.RangeSlider(
        id='index-slider',
        min=0,
        max=len(df) - 1,
        value=[0, len(df) - 1],
        marks={0: 'Inicio', len(df) - 1: 'Fin'},
        step=1
    )
])

# Define the callbacks
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('variable-dropdown', 'value'),
     Input('index-slider', 'value')]
)
def update_graph(selected_var, index_range):
    start_index, end_index = index_range
    filtered_df = df.iloc[start_index:end_index+1]

    if selected_var == 'temp':
        cols = temp_cols
        title = 'Comparación de Temperaturas'
    elif selected_var == 'current':
        cols = current_cols
        title = 'Comparación de Corrientes'

    melted_df = pd.melt(filtered_df.reset_index(), id_vars=['index'], value_vars=cols, var_name='Variable', value_name='Valor')

    fig = px.line(melted_df, x='index', y='Valor', color='Variable', title=title)
    fig.update_layout(xaxis_title='Índice', yaxis_title='Valor')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
