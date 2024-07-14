import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import traceback

# Cargar y preprocesar los datos
try:
    ruta_dataset = "dataset.xlsx"
    df = pd.read_excel(ruta_dataset)
    print(f"Datos cargados. Shape: {df.shape}")
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    df = pd.DataFrame()  # Create an empty DataFrame if loading fails

# Pasos de preprocesamiento de datos
try:
    if not df.empty:
        df['grip_lost'] = df['grip_lost'].astype(float)
        df['Robot_ProtectiveStop'] = df['Robot_ProtectiveStop'].astype(float)
        df.dropna(inplace=True)
        df.drop(columns=['Timestamp', 'Num'], inplace=True)

        columnas_a_normalizar = df.columns.difference(['grip_lost', 'Robot_ProtectiveStop'])

        # Eliminación de valores atípicos
        Q1 = df[columnas_a_normalizar].quantile(0.25)
        Q3 = df[columnas_a_normalizar].quantile(0.75)
        IQR = Q3 - Q1

        for col in columnas_a_normalizar:
            limite_inferior = Q1[col] - 1.5 * IQR[col]
            limite_superior = Q3[col] + 1.5 * IQR[col]
            df[col] = df[col].apply(lambda x: max(min(x, limite_superior), limite_inferior))

        df[columnas_a_normalizar] += 1 - df[columnas_a_normalizar].min()

        # Normalización y transformación
        escalador = MinMaxScaler()
        df[columnas_a_normalizar] = escalador.fit_transform(df[columnas_a_normalizar])

        pt = PowerTransformer(method='yeo-johnson')
        df[columnas_a_normalizar] = pt.fit_transform(df[columnas_a_normalizar])

        # PCA
        pca = PCA(n_components=2)
        datos_pca = pca.fit_transform(df[columnas_a_normalizar])

        # Clustering K-Means
        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(datos_pca)
            sse.append(kmeans.inertia_)

        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(datos_pca)
        df['Cluster_KMeans'] = kmeans.labels_

        # Clustering jerárquico
        modelo_jerarquico = AgglomerativeClustering(n_clusters=n_clusters)
        df['Cluster_Jerarquico'] = modelo_jerarquico.fit_predict(datos_pca)

        # Clustering DBSCAN
        vecinos = NearestNeighbors(n_neighbors=5)
        vecinos_fit = vecinos.fit(datos_pca)
        distancias, indices = vecinos_fit.kneighbors(datos_pca)
        distancias = np.sort(distancias[:, 4], axis=0)

        epsilon = 0.05
        dbscan = DBSCAN(eps=epsilon, min_samples=5)
        df['Cluster_DBSCAN'] = dbscan.fit_predict(datos_pca)

        print("Preprocesamiento completado.")
    else:
        print("No se pudo realizar el preprocesamiento debido a que el DataFrame está vacío.")
except Exception as e:
    print(f"Error en el preprocesamiento: {e}")
    traceback.print_exc()

# Crear aplicación Dash
app = dash.Dash(__name__)

# Definir el diseño
app.layout = html.Div([
    html.H1('Panel de Control de Análisis de Datos de Robot'),
    
    dcc.Tabs([
        dcc.Tab(label='Visión General de Datos', children=[
            html.Div([
                html.H3('Diagrama de caja para visualizar valores atípicos'),
                html.P('Este gráfico muestra la distribución de cada variable y ayuda a identificar valores atípicos.'),
                dcc.Graph(id='diagrama-caja-valores-atipicos')
            ]),
            html.Div([
                html.H3('Matriz de correlación'),
                html.P('Este gráfico muestra las correlaciones entre las diferentes variables del conjunto de datos.'),
                dcc.Graph(id='matriz-correlacion')
            ]),
        ]),
        dcc.Tab(label='Análisis de Clustering', children=[
            html.Div([
                html.H3('Método del codo para K-Means'),
                html.P('Este gráfico ayuda a determinar el número óptimo de clusters para el algoritmo K-Means.'),
                dcc.Graph(id='grafico-metodo-codo')
            ]),
            html.Div([
                html.H3('Visualización de clusters K-Means'),
                html.P('Este gráfico muestra los clusters formados por el algoritmo K-Means.'),
                dcc.Graph(id='clusters-kmeans')
            ]),
            html.Div([
                html.H3('Visualización de clusters jerárquicos'),
                html.P('Este gráfico muestra los clusters formados por el algoritmo de clustering jerárquico.'),
                dcc.Graph(id='clusters-jerarquicos')
            ]),
            html.Div([
                html.H3('Gráfico K-distancia para DBSCAN'),
                html.P('Este gráfico ayuda a determinar el valor óptimo de epsilon para el algoritmo DBSCAN.'),
                dcc.Graph(id='grafico-kdistancia')
            ]),
            html.Div([
                html.H3('Visualización de clusters DBSCAN'),
                html.P('Este gráfico muestra los clusters formados por el algoritmo DBSCAN.'),
                dcc.Graph(id='clusters-dbscan')
            ]),
        ]),
        dcc.Tab(label='Series Temporales', children=[
            html.Div([
                html.H3('Gráfico de Series Temporales'),
                html.P('Este gráfico muestra la evolución de las variables seleccionadas a lo largo del tiempo.'),
                dcc.Dropdown(
                    id='dropdown-variable',
                    options=[
                        {'label': 'Temperaturas', 'value': 'temp'},
                        {'label': 'Corrientes', 'value': 'current'},
                    ],
                    value='temp'
                ),
                dcc.Graph(id='grafico-series-temporales'),
                dcc.RangeSlider(
                    id='slider-indice',
                    min=0,
                    max=len(df) - 1 if not df.empty else 100,
                    value=[0, len(df) - 1 if not df.empty else 100],
                    marks={0: 'Inicio', len(df) - 1 if not df.empty else 100: 'Fin'},
                    step=1
                )
            ]),
        ]),
    ])
])



@app.callback(
    Output('diagrama-caja-valores-atipicos', 'figure'),
    Input('diagrama-caja-valores-atipicos', 'id')
)
def actualizar_diagrama_caja_valores_atipicos(_):
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for col in columnas_a_normalizar:
        fig.add_trace(go.Box(y=df[col], name=col))
    fig.update_layout(title='Diagrama de caja para visualizar valores atípicos', xaxis_tickangle=-45)
    return fig

@app.callback(
    Output('matriz-correlacion', 'figure'),
    Input('matriz-correlacion', 'id')
)
def actualizar_matriz_correlacion(_):
    if df.empty:
        return go.Figure()
    matriz_correlacion = df.corr()
    fig = go.Figure(data=go.Heatmap(z=matriz_correlacion.values, x=matriz_correlacion.columns, y=matriz_correlacion.columns))
    fig.update_layout(title='Matriz de correlación')
    return fig

@app.callback(
    Output('grafico-metodo-codo', 'figure'),
    Input('grafico-metodo-codo', 'id')
)
def actualizar_grafico_metodo_codo(_):
    if 'sse' not in globals():
        return go.Figure()
    fig = go.Figure(data=go.Scatter(x=list(range(1, 11)), y=sse, mode='lines+markers'))
    fig.update_layout(title='Método del codo para número óptimo de clusters', xaxis_title='Número de clusters', yaxis_title='SSE')
    return fig

@app.callback(
    Output('clusters-kmeans', 'figure'),
    Input('clusters-kmeans', 'id')
)
def actualizar_clusters_kmeans(_):
    if 'datos_pca' not in globals() or 'df' not in globals() or df.empty:
        return go.Figure()
    fig = go.Figure(data=go.Scatter(x=datos_pca[:, 0], y=datos_pca[:, 1], mode='markers', marker=dict(color=df['Cluster_KMeans'])))
    fig.update_layout(title='Visualización de clusters K-Means', xaxis_title='Componente PCA 1', yaxis_title='Componente PCA 2')
    return fig

@app.callback(
    Output('clusters-jerarquicos', 'figure'),
    Input('clusters-jerarquicos', 'id')
)
def actualizar_clusters_jerarquicos(_):
    if 'datos_pca' not in globals() or 'df' not in globals() or df.empty:
        return go.Figure()
    fig = go.Figure(data=go.Scatter(x=datos_pca[:, 0], y=datos_pca[:, 1], mode='markers', marker=dict(color=df['Cluster_Jerarquico'])))
    fig.update_layout(title='Visualización de clusters jerárquicos', xaxis_title='Componente PCA 1', yaxis_title='Componente PCA 2')
    return fig

@app.callback(
    Output('grafico-kdistancia', 'figure'),
    Input('grafico-kdistancia', 'id')
)
def actualizar_grafico_kdistancia(_):
    if 'distancias' not in globals():
        return go.Figure()
    fig = go.Figure(data=go.Scatter(x=list(range(len(distancias))), y=distancias, mode='lines'))
    fig.update_layout(title='Gráfico K-distancia para DBSCAN', xaxis_title='Puntos de datos ordenados por distancia', yaxis_title='Epsilon')
    return fig

@app.callback(
    Output('clusters-dbscan', 'figure'),
    Input('clusters-dbscan', 'id')
)
def actualizar_clusters_dbscan(_):
    if 'datos_pca' not in globals() or 'df' not in globals() or df.empty:
        return go.Figure()
    fig = go.Figure(data=go.Scatter(x=datos_pca[:, 0], y=datos_pca[:, 1], mode='markers', marker=dict(color=df['Cluster_DBSCAN'])))
    fig.update_layout(title='Visualización de clusters DBSCAN', xaxis_title='Componente PCA 1', yaxis_title='Componente PCA 2')
    return fig

@app.callback(
    Output('grafico-series-temporales', 'figure'),
    [Input('dropdown-variable', 'value'),
     Input('slider-indice', 'value')]
)
def actualizar_grafico(variable_seleccionada, rango_indice):
    if df.empty:
        return go.Figure()
    
    indice_inicio, indice_fin = rango_indice
    df_filtrado = df.iloc[indice_inicio:indice_fin+1]

    if variable_seleccionada == 'temp':
        cols = [col for col in ['Temperature_T0', 'Temperature_J1', 'Temperature_J2', 'Temperature_J3', 'Temperature_J4', 'Temperature_J5'] if col in df.columns]
        titulo = 'Comparación de Temperaturas'
    elif variable_seleccionada == 'current':
        cols = [col for col in ['Current_J0', 'Current_J1', 'Current_J2', 'Current_J3', 'Current_J4', 'Current_J5'] if col in df.columns]
        titulo = 'Comparación de Corrientes'

    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=df_filtrado.index, y=df_filtrado[col], mode='lines', name=col))
    
    fig.update_layout(title=titulo, xaxis_title='Índice', yaxis_title='Valor')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)