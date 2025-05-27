import pandas as pd

# Se agregan librerías para ampliar los datos
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

# Se agregan librerías para realizar la Regresión Lineal
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Se agrega la librería necesaria para guardar el modelo entrenado
import joblib

#---------------------------------Carga de Datos para Ampliar----------------------------------------

# Especifica la ruta al archivo de Excel
ruta_excel = 'C:/Users/solva/OneDrive/Escritorio/CUNDINAMARCA/Machine_Learning/ProyectoFinalDefinitivo/Datasets/Nieve_Caudal.xlsx'  

# Especifica las columnas que deseas extraer
columnas_a_extraer = ['Fecha', 'Caudal[l/s]', 'Nivel_Nieve[m]']

try:
    # Lee el archivo de Excel, extrayendo solo las columnas especificadas
    Datos_Reducidos = pd.read_excel(ruta_excel, usecols=columnas_a_extraer)
    Datos_Reducidos['Fecha'] = pd.to_datetime(Datos_Reducidos['Fecha'])

    # Imprime las primeras filas del DataFrame resultante para verificar
    print("DataFrame Datos_Reducidos creado exitosamente:")
    print(Datos_Reducidos.head())

except FileNotFoundError:
    print(f"Error: El archivo '{ruta_excel}' no fue encontrado.")
    exit()
except KeyError as e:
    print(f"Error: La columna '{e}' no se encuentra en el archivo de Excel.")
    exit()
except Exception as e:
    print(f"Ocurrió un error durante la carga del archivo: {e}")
    exit()

#---------------------------------------Generación de datos sintéticos---------------------------------
def generar_datos_sinteticos(num_anios=10):
    """
    Genera datos sintéticos de caudal y nivel de nieve para febrero y marzo
    durante un número especificado de años, con tendencia a la disminución.

    Args:
        num_anios (int): El número de años para generar datos.

    Returns:
        pandas.DataFrame: Un DataFrame con las columnas 'Fecha', 'Caudal [l/s]',
                          y 'Nivel_Nieve [m]'.
    """
    datos = []
    fecha_actual = date(2018, 2, 1)  # Año inicial arbitrario

    for _ in range(num_anios):
        for mes in [2, 3]:
            if mes == 2:
                num_dias = 28  # Consideramos febrero con 28 días para simplificar
            else:
                num_dias = 31

            for dia in range(1, num_dias + 1):
                fecha = date(fecha_actual.year, mes, dia)

                # Simulación de disminución gradual del caudal y nivel de nieve
                dias_transcurridos = (fecha - date(fecha_actual.year, 2, 1)).days
                factor_disminucion_caudal = 1 - (dias_transcurridos / (31 + 28)) * 0.6  # Disminución del 60% aprox.
                factor_disminucion_nieve = 1 - (dias_transcurridos / (31 + 28)) * 0.5  # Disminución del 50% aprox.

                caudal_base = 2200  # Valor base de caudal
                nieve_base = 2400   # Valor base de nivel de nieve

                # Añadiendo algo de aleatoriedad
                caudal = max(100, caudal_base * factor_disminucion_caudal + np.random.normal(0, 50))
                nieve = max(100, nieve_base * factor_disminucion_nieve + np.random.normal(0, 30))

                datos.append([fecha, round(caudal, 2), round(nieve)])

        fecha_actual = date(fecha_actual.year + 1, 2, 1)  # Pasar al siguiente año

    df = pd.DataFrame(datos, columns=['Fecha', 'Caudal [l/s]', 'Nivel_Nieve [m]'])
    return df

# Generar los datos sintéticos para 10 años
df_sintetico = generar_datos_sinteticos(num_anios=10)

# Mostrar las primeras filas del DataFrame generado
print('Datos sintéticos generados con éxito.\n')
print(df_sintetico.head())

# Si deseas guardar los datos en un archivo CSV:
# df_sintetico.to_csv('datos_sinteticos_nieve_caudal.csv', index=False)

#-------------------------Dispersión de datos para asegurar linealidad-----------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(df_sintetico['Nivel_Nieve [m]'], df_sintetico['Caudal [l/s]'])
plt.xlabel('Nivel de Nieve [m]')
plt.ylabel('Caudal [l/s]')
plt.title('Dispersión de Caudal en función del Nivel de Nieve')
plt.grid(True)
plt.show(block=False)  # Mostrar la gráfica sin bloquear la ejecución
plt.pause(5)           # Pausar por 5 segundos
plt.close()            # Cerrar la gráfica

#---------------------------------REGRESIÓN LINEAL----------------------------------------------------

#------------------------------Entrenamiento del modelo-----------------------------------------------
# Preparar los datos
X = df_sintetico[['Nivel_Nieve [m]']]  # Variable independiente 
y = df_sintetico['Caudal [l/s]']       # Variable dependiente 

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test) 

#-------------------------------- Evaluación del modelo -------------------------------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Coeficiente de Determinación (R²): {r2:.2f}') 

# Graficar las predicciones contra los valores reales
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Valores Reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicciones')
plt.xlabel('Nivel de Nieve (m)')
plt.ylabel('Caudal (l/s)')
plt.title('Predicciones del Caudal vs. Valores Reales')
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(5)
plt.close  

#----------------------------Guardar el modelo entrenado en un archivo-------------------------------
nombre_archivo_modelo = 'modelo_regresion_CaudalVsNieve.joblib'
joblib.dump(modelo, nombre_archivo_modelo)

print(f'Modelo entrenado guardado en: {nombre_archivo_modelo}')
print('Fin de ejecución del modelo.') 