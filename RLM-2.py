# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:53:46 2024

@author: ROCIOCARRASCO
"""

#%%Importar librerias
import numpy as np
import pandas as pd
# Generación de gráficos
import seaborn as sns
import matplotlib.pyplot as plt
# Explorar datos, estimar modelos estadísticos y realizar pruebas estadísticas
import statsmodels.api as sm 
import statsmodels.stats.api as sms
# Optimización y métodos numéricos
from scipy import stats
from scipy.stats import norm
# Aprendizaje automático
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

#%% Cargar base de datos
#Boston House prices
data_url = "insurance.csv"
raw_df = pd.read_csv(data_url)
data = raw_df.iloc[:,:]

# age - Edad del asegurado
# sex - genero del asegurado (1) male (2) female
# bmi - indice masa corporal
# children - numero de hijos
# smoker - (1) fuma (2) no fuma
# region - region del aegurado:
    # southeast (1)
    # southwest (2)
    # northeast (3)
    # northwest (4)
# charges - cargos del seguro


#%% Selección de Variables
#Variables Independientes (INDUS,RM ,PTRATIO, LSTAT)
X_multiple = np.matrix(data.iloc[:, [0,2,4,5]])
Y_multiple=raw_df.values[:, 6]

#%% Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_multiple, Y_multiple, test_size=0.2, random_state=42)

# Agregar una constante para el término independiente (B0)
X_train = sm.add_constant(X_train)
# Crear un modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(Y_train, X_train).fit()
# Resumen estadístico
print(modelo.summary())

#predicciones en los datos de prueba
X_test = sm.add_constant(X_test)
Y_pred = modelo.predict(X_test)

#%% Métricas de Evaluación del modelo dtaos de prueba

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n**Evaluación del modelo de regresión lineal **")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")


#%% VERIFICACIÓN DE SUPUESTOS DEL MODELO DE REGRESIÓN

## NORMALIDAD DE LOS RESIDUOS
#Obtener los residuos del modelo
residuals = modelo.resid

### NORMALIDAD DE LOS RESIDUOS ###

### HISTOGRAMA  ###

plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=False, color='lightblue', stat='density', edgecolor='black', alpha=0.7)
# Calcular la media y desviación estándar de los residuos
mu, std = np.mean(residuals), np.std(residuals)
# Crear una curva de densidad normal usando la media y desviación estándar
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
# Dibujar la curva normal
plt.plot(x, p, 'r', linewidth=2)
# Añadir título y etiquetas
plt.title("Histograma de los Residuos con Curva de Densidad Normal", fontsize=14)
plt.xlabel("Residuos", fontsize=12)
plt.ylabel("Densidad", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

### QQ PLOT ###

fig, ax = plt.subplots(figsize=(8, 6))  # Ajusta el tamaño de la figura
sm.qqplot(residuals, line='45', fit=True, marker='*', ax=ax)
ax.get_lines()[0].set_markerfacecolor('black') 
ax.get_lines()[0].set_markeredgecolor('black')  
ax.get_lines()[1].set_linestyle('--')
ax.set_title("QQ Plot de los Residuos", fontsize=14)
ax.set_xlabel("Cuantiles Teóricos", fontsize=12)
ax.set_ylabel("Cuantiles de los Residuos", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)  # Cuadrícula con líneas suaves
plt.show()


### SHAPIRO-WILK ###
#H0: Normalidad (p>0.05)
#H1: Sin normalidad (p<0.05)
test_statistic, p_value = stats.shapiro(residuals)
print("Resultados de la Prueba de Normalidad Shapiro-Wilk:")
print(f"Estadístico de prueba: {test_statistic:.4f}")
print(f"p-valor: {p_value:.4f}")
# Interpretación del p-valor
alpha = 0.05  # Nivel de significancia
if p_value > alpha:
    print("No se rechaza la hipótesis nula: los residuos siguen una distribución normal.")
else:
    print("Se rechaza la hipótesis nula: los residuos no siguen una distribución normal.")


### HOMOCEDASTICIDAD (VARIANZA CONSTANTE) ###

plt.figure(figsize=(10, 6))
plt.scatter(modelo.fittedvalues, residuals, color='black', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residuos vs. Valores Ajustados', fontsize=16)
plt.xlabel('Valores Ajustados', fontsize=14)
plt.ylabel('Residuos', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


### PRUEBA DE BREUSCH-PAGAN ###
#H0: Homocedasticidad (p>0.05)
#H1: No homocedasticidad (p<0.05)
test_statistic, p_value, f_value, f_p_value = sms.het_breuschpagan(residuals, X_train)
print("Resultados de la Prueba de Heterocedasticidad Breusch-Pagan:")
print(f"Estadístico de Lagrange: {test_statistic:.4f}")
print(f"p-valor (Lagrange): {p_value:.4f}")
print(f"Estadístico F: {f_value:.4f}")
print(f"p-valor (F): {f_p_value:.4f}")
# Interpretación del p-valor
alpha = 0.05  # Nivel de significancia
if p_value > alpha:
    print("No se rechaza la hipótesis nula: no hay evidencia de heterocedasticidad.")
else:
    print("Se rechaza la hipótesis nula: hay evidencia de heterocedasticidad.")