# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:08:59 2024

@author: ROCIOCARRASCO
"""
#%% 
# Se utiliza para ilustrar conceptos de machine learning
#import mglearn 
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

#%% Cargar conjunto de datos (cáncer de mama)
iris=load_iris()
print(iris['DESCR'])
data=pd.DataFrame(iris.data)
data.columns=iris.feature_names

# Análisis de Correlación 
cor=data.corr()
# Análisis de varianza
print(data.var())

#%% Estandarizar los datos media 0 y desviación estándar 1
#Normales y sin outliers
# X-X_med/Std_X
scaler=StandardScaler()
scaler.fit(data)
scaled_data=scaler.transform(data)

#MinMaxScaler transformará los valores proporcionalmente dentro del rango [0,1]
#Presencia de outliers, preserva la forma de los datos
#X-Xmin/Xmax-Xmin
# scaler=MinMaxScaler()
# scaler.fit(data)
# scaled_data=scaler.transform(data)

#%% Algoritmo pca
pca=PCA()
#pca=PCA(n_components=10) # Indicar el número de componentes principales 
pca.fit(scaled_data)

# Ponderación de los componentes principales (vectores propios)
pca_score=pd.DataFrame(data    = pca.components_, columns = data.columns,)

#%% Gráfica del aporte a cada componente principal
# Aporte al primer componente principal 
matrix_transform = pca.components_.T
plt.bar(np.arange(4),matrix_transform[:,0])
plt.xticks(range(len(data.columns)), data.columns,rotation = 90)
plt.ylabel('Loading Score')
plt.show()



# # Mapa de calor para visualizar in influencia de las variables
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
# componentes = pca.components_
# plt.imshow(componentes.T, cmap='plasma', aspect='auto')
# plt.yticks(range(len(data.columns)), data.columns)
# plt.xticks(range(len(data.columns)), np.arange(pca.n_components_)+ 1)
# plt.grid(False)
# plt.colorbar();

#%%Obtener las primeras 3 variables con mayor aporte
# # Pesos por componente principal
# loading_scores = pd.DataFrame(pca.components_[0])
# #Nombre de las variables (columnas) asociada a cada peso
# loading_scores.index=cancer.feature_names
# # Ordena de mayor a menor los pesos
# sorted_loading_scores = loading_scores[0].abs().sort_values(ascending=False)
# #Selección de las 10 variables que más aportan a cada componente principal
# top_10_variables= sorted_loading_scores[0:10].index.values
# print(top_10_variables)


top_vars_per_component = {}
for i in range(pca.n_components_):
    component = pd.Series(pca.components_[i], index=data.columns)
    sorted_component = component.abs().sort_values(ascending=False)
    top_vars_per_component[f'PCA{i+1}'] = sorted_component.index[:3].tolist()  # Top 3 variables
print("Variables de mayor aporte para los primeros componentes:", top_vars_per_component)

#%% Nuevas variables llamadas components principales (569x30)
pca_data=pca.transform(scaled_data) 

#Visualización de los componentes principales

#2D
plt.figure(figsize=(10, 7))

# Graficar los puntos, diferenciando las clases
for i, target_name in enumerate(iris.target_names):
    plt.scatter(pca_data[iris.target == i, 0], pca_data[iris.target == i, 1], 
                label=target_name, s=40, edgecolor='k')

# Etiquetas de los ejes y leyenda
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Proyección primeros 2 Componentes Principales")
plt.legend(loc='best')
plt.show()


# 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos con etiquetas de "Benigno" y "Maligno"
for i, target_name in enumerate(iris.target_names):
    ax.scatter(pca_data[iris.target == i, 0], pca_data[iris.target == i, 1], pca_data[iris.target == i, 2], 
               label=target_name, s=40, edgecolor='k')

# Etiquetas de los ejes y título
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
plt.title("Proyección primeros 3 Componentes Principales")

# Añadir la leyenda para indicar "Benigno" y "Maligno"
ax.legend(loc='best')
plt.show()



#%%Porcentaje de varianza explicada por cada componente principal proporciona
#Lambda/suma_Lambda (valor_propio/suma_valores_propios)
per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)

#Scree plot para visualizar el porcentaje de varianza explicada
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(per_var) + 1), per_var, marker='o', linestyle='-', color='b')

for i, var in enumerate(per_var):
    plt.text(i + 1.5, var + 1,f"{var:.1f}", ha='center', va='bottom', fontsize=8, color="blue")
    
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada (%)')
plt.title('Scree Plot')
plt.show()

#%% Porcentaje de varianza acumulada

porcent_acum = np.cumsum(per_var)
porcent_acum = np.minimum(porcent_acum, 100)  # Limita el máximo valor acumulado a 100%

# Impresión de resultados
for i, valor in enumerate(porcent_acum, start=1):
    print(f"Componente {i}: {valor:.2f}%")

#Selección del número de componentes principales
threshold = 85  # umbral deseado
n_components = np.argmax(porcent_acum >= threshold) + 1
print(f"Número de componentes necesarios para capturar el {threshold}% de la varianza: {n_components}")





