###############################################################################################################
# App de clusterizacion de base generica usando kmeans
###############################################################################################################

#**************************************************************************************************************
# [A] Importar LIbrerias a Utilizar
#**************************************************************************************************************

import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings('ignore')


#**************************************************************************************************************
# [B] Crear funciones utiles para posterior uso
#**************************************************************************************************************

#**************************************************************************************************************
# B.1 Funcion de Diagnostico para Kmeans 

@st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
def diagnostico_kmeans(
  df, # df original con data 
  cols, # listado de vars numericas para segmentar
  max_k = 11 # maximo de opciones a revisar 
  
):
  
  # crear df estantarizado
  df_est = StandardScaler().fit_transform(df[cols])

  # crear dfs en blanco donde se iran guardando datos
  df_metricas = pd.DataFrame([])
  df_cluster = pd.DataFrame([])

  # ir iterando por cada opcion de numero de cluster
  for i in range(2,max_k):
    
    # ajustar kmeans
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_est)

    # valor de inercia
    metrica_inercia = kmeans.inertia_

    # valor de silhouette_score
    metrica_silhouette = silhouette_score(
      X=df_est,
      labels=kmeans.labels_,
      metric='euclidean'
      )

    # valor de davies_bouldin_score
    metrica_davies_bouldin = davies_bouldin_score(
      X=df_est,
      labels=kmeans.labels_
      )

    # valor de BIC score (https://datascience.oneoffcoder.com/kmc-bic-aic.html)
    gmm = GaussianMixture(n_components=i, init_params='kmeans')
    gmm.fit(df_est)

    metrica_bic = gmm.bic(df_est)
    metrica_aic = gmm.aic(df_est)

    # obtener peso promedio de cada cluster segun muestra y su dispersion
    df_c = pd.DataFrame({'c': kmeans.labels_}).value_counts(normalize=True)

    c_min = round(100*min(df_c),1)
    c_prom = round(100*np.mean(df_c),1)
    c_std = round(100*np.std(df_c),1)
    c_max = round(100*max(df_c),1)

    df_metricas = pd.concat([
      df_metricas,
      pd.DataFrame({
        'k': i,
        'inercia': metrica_inercia,
        'silhouette': metrica_silhouette,
        'davies_bouldin': metrica_davies_bouldin,
        'bic': metrica_bic,
        'aic': metrica_aic,
        'c_min': c_min,
        'c_prom': c_prom,
        'c_std': c_std,
        'c_max': c_max,
        'c_error_sup': c_max-c_prom,
        'c_error_inf': c_prom-c_min
      },index=[0])
    ])

    df_cluster = pd.concat([
      df_cluster,
      pd.DataFrame({
        'cluster_'+str(i): ['C'+str(x+1) for x in kmeans.labels_]
      })
    ],axis=1)

  # Generar graficos: Inercia
  fig1 = go.Figure()
  fig1.add_trace(go.Scatter(
    x=df_metricas['k'],
    y=df_metricas['inercia'],
    mode='lines',
    showlegend=False
    ))
  fig1.update_layout(title_text='Indicador: Inercia')
  fig1.update_xaxes(title_text='Cantidad de Clusters')
  fig1.update_yaxes(title_text='Inercia')
  
  # Generar graficos: silhouette
  fig2 = go.Figure()
  fig2.add_trace(go.Scatter(
    x=df_metricas['k'],
    y=df_metricas['silhouette'],
    mode='lines',
    showlegend=False
    ))
  fig2.update_layout(title_text='Indicador: silhouette')
  fig2.update_xaxes(title_text='Cantidad de Clusters')
  fig2.update_yaxes(title_text='silhouette')

  # Generar graficos: davies_bouldin
  fig3 = go.Figure()
  fig3.add_trace(go.Scatter(
    x=df_metricas['k'],
    y=df_metricas['davies_bouldin'],
    mode='lines',
    showlegend=False
    ))
  fig3.update_layout(title_text='Indicador: davies bouldin')
  fig3.update_xaxes(title_text='Cantidad de Clusters')
  fig3.update_yaxes(title_text='davies bouldin')

  # Generar graficos: BIC
  fig4 = go.Figure()
  fig4.add_trace(go.Scatter(
    x=df_metricas['k'],
    y=df_metricas['bic'],
    mode='lines',
    showlegend=False
    ))
  fig4.update_layout(title_text='Indicador: bayesian information criterion (BIC)')
  fig4.update_xaxes(title_text='Cantidad de Clusters')
  fig4.update_yaxes(title_text='bic')

  # Generar graficos: AIC
  fig5 = go.Figure()
  fig5.add_trace(go.Scatter(
    x=df_metricas['k'],
    y=df_metricas['aic'],
    mode='lines',
    showlegend=False
    ))
  fig5.update_layout(title_text='Indicador: akaike information criterion (AIC)')
  fig5.update_xaxes(title_text='Cantidad de Clusters')
  fig5.update_yaxes(title_text='aic')

  # Generar graficos: distribucion de pesos de cluster
  fig6 = go.Figure()
  fig6.add_trace(go.Scatter(
    x=df_metricas['k'],
    y=df_metricas['c_prom'],
    error_y=dict(
      type='data',
      symmetric=False,
      array=df_metricas['c_error_sup'],
      arrayminus=df_metricas['c_error_inf'],
      visible=True
      ),
    error_x=dict(
      type='data',
      array=df_metricas['c_std']/100,
      color='red'
      ),
    mode='markers',
    showlegend=False
    ))
  fig6.update_layout(title_text='Indicador: variacion en pesos de clusters')
  fig6.update_xaxes(title_text='Cantidad de Clusters')
  fig6.update_yaxes(title_text='frecuencia')
  
  
  # Generar graficos: Unir todos los graficos
  fig = make_subplots(
    rows=2, 
    cols=3,
    subplot_titles=['Plot '+str(i) for i in range(1,8-1)],
    vertical_spacing=0.15,
    horizontal_spacing=0.05,
    shared_yaxes=False
    )

  fig.add_trace(fig1.data[0], row=1, col=1)
  fig.add_trace(fig2.data[0], row=1, col=2)
  fig.add_trace(fig3.data[0], row=1, col=3)
  fig.add_trace(fig4.data[0], row=2, col=1)
  fig.add_trace(fig5.data[0], row=2, col=2)
  fig.add_trace(fig6.data[0], row=2, col=3)

  names = {
    'Plot 1':'Inercia', 
    'Plot 2':'Silhouette',
    'Plot 3':'Davies Bouldin',
    'Plot 4':'Bayesian Information Criterion (BIC)',
    'Plot 5':'Akaike Information Criterion (AIC)',
    'Plot 6':'Peso prom. por cluster + dispersion'
    }
  fig.for_each_annotation(lambda a: a.update(text = names[a.text]))

  fig['layout']['xaxis4']['title']='Cantidad de Clusters'
  fig['layout']['xaxis5']['title']='Cantidad de Clusters'
  fig['layout']['xaxis6']['title']='Cantidad de Clusters'

  fig.update_layout(
    height=600, 
    width=1200,
    xaxis = dict(dtick = 1),
    xaxis2 = dict(dtick = 1),
    xaxis3 = dict(dtick = 1),
    xaxis4 = dict(dtick = 1),
    xaxis5 = dict(dtick = 1),
    xaxis6 = dict(dtick = 1)
    )


  # Generar df con todos los clusters
  df2 = pd.concat([df,df_cluster],axis=1)
  
  # retornar entregables (Grafico + DF)
  return fig,df2


#**************************************************************************************************************
# B.2 Funcion de Entregables para Kmeans 

@st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
def entregables_kmeans(
  df2, # df original con todos los clusters
  k_final, # valor de k deseado 
  cols, # columnas numericas elegidas para clusterizar
  max_distintos = 10 # maximo de valores distintos en categoricas
  ):

  # crear df solo con columnas deseadas (originales + cluster especifico)
  df3 = df2[
    [i for i in df2.columns if 'cluster_' not in i]+['cluster_'+str(k_final)]
    ].rename(
      columns = {'cluster_'+str(k_final):'Cluster'}
      )

  # reordenar df para manera mas sencilla
  df3_k = pd.melt(
    df3[cols+['Cluster']],
    id_vars='Cluster',
    var_name='Variable',
    value_name='Valor'
    )

  # Graficos de analisis numerico: BoxPlot
  fig_a = px.box(
    df3_k,
    x = 'Cluster',
    y = 'Valor',
    facet_col= 'Variable',
    facet_col_wrap=4,
    facet_row_spacing=0.1, 
    facet_col_spacing=0.04
  )
  fig_a.update_layout(
    height=600, 
    width=1200
    )
  fig_a.update_yaxes(matches=None)
  fig_a.for_each_yaxis(lambda axis: axis.update(showticklabels=True))
  fig_a.for_each_xaxis(lambda axis: axis.update(showticklabels=True))
  fig_a.update_xaxes(type='category')
  fig_a.update_xaxes(categoryorder='category ascending')
  fig_a.update_traces(width=0.5)

  # Graficos de analisis numerico: Violin
  fig_b = px.violin(
    df3_k,
    x = 'Cluster',
    y = 'Valor',
    facet_col= 'Variable',
    facet_col_wrap=4,
    facet_row_spacing=0.1, 
    facet_col_spacing=0.04,
    box=True
  )
  fig_b.update_layout(
    height=600, 
    width=1200
    )
  fig_b.update_yaxes(matches=None)
  fig_b.for_each_yaxis(lambda axis: axis.update(showticklabels=True))
  fig_b.for_each_xaxis(lambda axis: axis.update(showticklabels=True))
  fig_b.update_xaxes(type='category')
  fig_b.update_xaxes(categoryorder='category ascending')
  fig_b.update_traces(width=0.5)

  # Graficos de analisis numerico: Prom+Desv
  fig_c = px.scatter(
    df3_k.groupby(['Cluster','Variable']).agg(
      Conteo = pd.NamedAgg(column = 'Valor', aggfunc = len),
      Prom = pd.NamedAgg(column = 'Valor', aggfunc = np.mean),
      Desv = pd.NamedAgg(column = 'Valor', aggfunc = np.std)
      ).reset_index(),
    x = 'Cluster',
    y = 'Prom',
    error_y='Desv',
    facet_col= 'Variable',
    facet_col_wrap=4,
    facet_row_spacing=0.1, 
    facet_col_spacing=0.04
  )
  fig_c.update_layout(
    height=600, 
    width=1200
    )
  fig_c.update_yaxes(matches=None)
  fig_c.for_each_yaxis(lambda axis: axis.update(showticklabels=True))
  fig_c.for_each_xaxis(lambda axis: axis.update(showticklabels=True))
  fig_c.update_xaxes(type='category')
  fig_c.update_xaxes(categoryorder='category ascending')
  
  # Identificar variables categoricas para posterior grafico 
  cols_cats = []
  for i in [i for i in df2.columns if 'cluster_' not in i]:
    if len(np.unique(df2[i].dropna()))<=max_distintos:
      cols_cats.append(i)

  # Crear df solamente con esas variables
  df3_cat1 = pd.melt(
    df3[cols_cats+['Cluster']],
    id_vars='Cluster',
    var_name='Variable',
    value_name='Valor'
    )

  # crear version agrupada con total por cluster (+ calcular frecuencia)
  df3_cat2 = pd.merge(
    df3_cat1.groupby(['Variable','Cluster','Valor']).agg(
      Conteo = pd.NamedAgg(column = 'Variable', aggfunc = len)
      ).reset_index(),
    df3_cat1.groupby(['Variable','Cluster']).agg(
      Conteo = pd.NamedAgg(column = 'Variable', aggfunc = len)
      ).reset_index(),
    how='left',
    on=['Variable','Cluster']
  )
  df3_cat2['Freq'] = df3_cat2.apply(lambda x: x['Conteo_x']/x['Conteo_y'],axis=1)

  # Agregar artificialmente totales
  df3_aux = df3.groupby(['Cluster']).agg(
      Conteo = pd.NamedAgg(column = 'Cluster', aggfunc = len)
      ).reset_index()
  df3_aux = pd.DataFrame({
    'Variable':'Total por cluster',
    'Cluster': df3_aux['Cluster'],
    'Valor': 'Total',
    'Conteo_x': 1,
    'Conteo_y': 1,
    'Freq': df3_aux['Conteo']
  })
  df3_cat3 = pd.concat([df3_aux,df3_cat2],axis=0)

  # Calcular tupla de ubicaciones filas-columnas en subplot posterior
  tupla_fc = []
  total_grilla = len(np.unique(df3_cat3['Variable']))
  for i in range(0,total_grilla):
    f = int(np.floor((i)/4)+1)
    c = i%4 + 1
    tupla_fc.append((f,c))

  # Graficos de analisis categorico: todas las variables categoricas
  cols_cats2 = ['Total por cluster']+cols_cats
  fig = make_subplots(
    rows= max(tupla_fc, key=lambda x: x[0])[0], 
    cols= max(tupla_fc, key=lambda x: x[1])[1],
    subplot_titles=cols_cats2,
    vertical_spacing=0.15,
    horizontal_spacing=0.05,
    shared_yaxes=False
    )

  for i1 in range(0,len(tupla_fc)):
    
    f = tupla_fc[i1][0]
    c = tupla_fc[i1][1]
    
    df3_cat4 = df3_cat3.loc[df3_cat3['Variable']==cols_cats2[i1]]
    categorias = list(df3_cat4['Valor'].unique())
    
    for i2 in range(0,len(categorias)):
      fig.add_trace(go.Bar(
        x=df3_cat4.loc[df3_cat4['Valor']==categorias[i2 ],'Cluster'],
        y=df3_cat4.loc[df3_cat4['Valor']==categorias[i2 ],'Freq'],
        name=categorias[i2],
        legendgroup = str(i1+1)
      ),row=f,col=c)

  fig.update_xaxes(categoryorder='category ascending')
  
  fig.update_layout(
    height=600, 
    width=1200,
    barmode='stack',
    legend_tracegroupgap = 15
    )

  # retornar entregables
  return fig_a,fig_b,fig_c,fig,df3


#**************************************************************************************************************
# [Z] Comenzar a diseñar App
#**************************************************************************************************************

def main():
  
  # Use the full page instead of a narrow central column
  st.set_page_config(layout='wide')
    
  #=============================================================================================================
  # [01] SideBar
  #=============================================================================================================   

  # titulo inicial 
  st.markdown('## Clusterizacion de bases usando kmeans')
  
  # autoria 
  st.sidebar.markdown('**Autor: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')
      
  # subir archivo 
  Archivo = st.sidebar.file_uploader('Subir Data.csv',type=['csv'])

  if Archivo is not None:
    
    

    # leer archivo (con distintos delimitadores)
    df = pd.read_csv(Archivo,delimiter ='[:,|;]',engine='python')

    # desplegar columnas numericas para eleccion    
    cols_nums = st.sidebar.multiselect(
      'Seleccionar variables numericas',
      list(df.select_dtypes([np.number]).columns),
      list(df.select_dtypes([np.number]).columns)[0:3],
      key = 1
      )
        
    # agregar boton de generar nubes
    Boton_ejecutar = st.sidebar.button(label='Segmentar')

  #=============================================================================================================
  # [02] Main
  #=============================================================================================================   

  #-------------------------------------------------------------------------------------------------------------
  # [02.1] Main: Diagnostico
  #-------------------------------------------------------------------------------------------------------------

    # truco para no alterar resultados desde cero apretando boton
    if st.session_state.get('button') != True:
      st.session_state['button'] = Boton_ejecutar

    if st.session_state['button']:
      
      # calcular entregables de diagnostico
      fig_diagnostico,df_clusters=diagnostico_kmeans(
        df = df,
        cols = cols_nums,
        max_k=11
        )
      
      # titulo entregable 
      st.markdown('### 1. Graficos para desicion de k')

      # mostrar grafico de diagnostico 
      st.plotly_chart(fig_diagnostico, use_container_width=True) 


  #-------------------------------------------------------------------------------------------------------------
  # [02.2] Main: Resultados segun k
  #-------------------------------------------------------------------------------------------------------------

      # titulo a mostrar 
      col_2a,col_2b,col_2c = st.columns((1.4,1,4))

      col_2a.markdown('### 2. Eleccion de k')
      
      num_k = col_2b.slider(
        label = 'Elegir cantidad de cluster', 
        min_value=2, 
        max_value=10, 
        value=4
        )


      # calcular entregables segun eleccion de k
      fig_boxplot,fig_violin,fig_promdesv,fig_cat,df_k = entregables_kmeans(
        df2 = df_clusters, # df original con todos los clusters
        k_final = num_k, # valor de k deseado 
        cols = cols_nums, # columnas numericas elegidas para clusterizar
        max_distintos = 10 # maximo de valores distintos en categoricas
        )

  #_____________________________________________________________________________________________________________
  # Grafico de Variables numericas 

      # titulo a mostrar 
      col_21a,col_21b,col_21c = st.columns((2,2,1))
      
      col_21a.markdown('#### 2.1 Descripcion de variables numericas')
      
      # seleccion de opcion de grafico a mostrar 
      tipo_num1 = col_21b.radio(
        'Elegir tipo de grafico', 
        ['Boxplot','Violin','Prom+Desv'], 
        horizontal=True
      )
      
      # asignar grafico a mostrar 
      if tipo_num1=='Boxplot':
        fig_num1 = fig_boxplot
      elif tipo_num1=='Violin':
        fig_num1 = fig_violin
      else:
        fig_num1 = fig_promdesv
      
      # mostrar grafico 
      st.plotly_chart(fig_num1, use_container_width=True)


  #_____________________________________________________________________________________________________________
  # Grafico de Variables categoricas  
  
  
      # titulo a mostrar 
      st.markdown('#### 2.2 Descripcion de variables categoricas')

      # mostrar grafico 
      st.plotly_chart(fig_cat, use_container_width=True)

  #_____________________________________________________________________________________________________________
  # Tabla con valor de k final (pudiendo filtrar por cluster)

      # titulo a mostrar 
      col_23a,col_23b,col_23c = st.columns((1.5,1,2))
      
      col_23a.markdown('#### 2.3 Tabla con detalle cluster')

      # filtro para tabla 
      filtro_c = col_23c.multiselect(
        'Seleccionar cluster a filtrar',
        list(np.unique(df_k['Cluster'])),
        list(np.unique(df_k['Cluster'])),
        key = 2
        )
      
      # dejar disponible para descagar df
      col_23b.download_button(
        'Descargar tabla',
        df_k[df_k['Cluster'].isin(filtro_c)].to_csv().encode('utf-8'),
        'Tabla_segmentada.csv',
        'text/csv',
        key='download-csv'
        )

      # Mostrar df resultante 
      st.dataframe(df_k[df_k['Cluster'].isin(filtro_c)])
      




# arrojar main para lanzar App
if __name__=='__main__':
    main()
    
# Escribir en terminal: streamlit run App_Kmeans2.py
# !streamlit run App_Kmeans2.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/14_Streamlit App de Kmeans2/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit

