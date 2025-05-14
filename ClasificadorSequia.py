import pandas as pd
import glob
import os

#---------------------------------------- Tratamiento de datos -----------------------------------------------------------
"""
En esta sección se filtran los valores promedio de caudal
registrados a lo largo de distintos meses de los años 2018 y 2019 correspondientes al Río Mendoza.
Se destaca que varios archivos de Excel hallados en la web estaban dañados, aún así, se recolectaron
1346 datos de caudales, que se consideran suficientes para el entrenamiento del modelo.
"""

def cargar_y_filtrar_multiples_csv(ruta_carpeta):
    """
    Carga datos de múltiples archivos CSV y filtra las filas donde
    'nombre_de_zona' es 'SUBDELEGACION DEL RIO MENDOZA', manteniendo
    las columnas 'nombre_de_zona' y 'valor_promedio_relevado'.

    Args:
        ruta_carpeta (str): Ruta de la carpeta que contiene los archivos CSV.

    Returns:
        pandas.DataFrame: DataFrame con los datos filtrados, o vacío si no hay coincidencias.
    """
    all_filtered_data = pd.DataFrame()
    archivos_csv = glob.glob(os.path.join(ruta_carpeta, "*.csv"))

    if not archivos_csv:
        print(f"No se encontraron archivos CSV en la carpeta: {ruta_carpeta}")
        return all_filtered_data

    for archivo in archivos_csv:
        try:
            print(f"\nProcesando archivo: {archivo}")

            # Intenta leer el CSV detectando automáticamente el delimitador
            df = pd.read_csv(archivo, sep=None, engine='python')

            if 'nombre_de_zona' in df.columns and 'valor_promedio_relevado' in df.columns:
                # Filtra las filas por la zona requerida
                filtered_df = df[df['nombre_de_zona'] == 'SUBDELEGACION DEL RIO MENDOZA'][['nombre_de_zona', 'valor_promedio_relevado']].copy()

                if not filtered_df.empty:
                    all_filtered_data = pd.concat([all_filtered_data, filtered_df], ignore_index=True)
                    print(f"  - Se encontraron {len(filtered_df)} registros.")
                else:
                    print("  - No se encontraron registros para 'SUBDELEGACION DEL RIO MENDOZA'.")
            else:
                print("  - El archivo no contiene las columnas requeridas ('nombre_de_zona' y/o 'valor_promedio_relevado').")

        except Exception as e:
            print(f"Error al leer el archivo '{archivo}': {e}")

    if all_filtered_data.empty:
        print("\nNo se encontraron datos para 'SUBDELEGACION DEL RIO MENDOZA' en ningún archivo.")
    else:
        print(f"\n-----SE CARGARON Y FILTRARON UN TOTAL DE {len(all_filtered_data)} REGISTROS DE TODOS LOS ARCHIVOS-----.")

    return all_filtered_data


# Uso:
ruta_de_la_carpeta = "C:/Users/solva/OneDrive/Escritorio/CUNDINAMARCA/Machine_Learning/ProyectoFinalDefinitivo/Datasets"
datos_filtrados = cargar_y_filtrar_multiples_csv(ruta_de_la_carpeta)

if not datos_filtrados.empty:
    print("\nPrimeras filas del DataFrame resultante (antes de invertir):")
    print(datos_filtrados.head())

    # Intenta convertir la columna a numérica, los errores se convierten en NaN
    datos_filtrados['valor_promedio_relevado'] = pd.to_numeric(datos_filtrados['valor_promedio_relevado'], errors='coerce')

    # Elimina las filas con NaN en la columna de caudal (si es apropiado)
    datos_filtrados = datos_filtrados.dropna(subset=['valor_promedio_relevado'])

    # Filtrar los caudales mayores a 50000
    datos_filtrados = datos_filtrados[datos_filtrados['valor_promedio_relevado'] <= 50000]

    print(f"\nNúmero de registros después de eliminar caudales mayores a 50000: {len(datos_filtrados)}")

    # Invertir el orden del DataFrame
    """
    Esta línea se agrega debido a que hubo problemas al intentar extraer las fechas de los archivos de Excel,
    entonces se procedió a guardarlos manualmente del más nuevo al más viejo, dado que las fechas en cada archivo
    estaban ordenadas de la más nueva a la más vieja, y luego se invirtió el orden para obtener el orden correcto
    para graficar los datos 
    """
    datos_filtrados = datos_filtrados.iloc[::-1].reset_index(drop=True)

    print("\nPrimeras filas del DataFrame resultante (después de eliminar e invertir):")
    print(datos_filtrados.head())

    #-------------------------Dispersión para asegurar la linealidad de los datos--------------------------------------------------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(datos_filtrados.index, datos_filtrados['valor_promedio_relevado'], alpha=0.6)
    plt.xlabel("Índice de Datos (Invertido)")
    plt.ylabel("Valor Promedio de Caudal [l/s]")
    plt.title("Distribución de Valores de Caudal (Orden Invertido)")
    plt.grid(True)
    plt.show()

    #-------------------------Regresión Lineal para Predecir el Caudal en [l/s]------------------------------------------------
    ########### NO GRAFICA Y NO SÉ POR QUÉ
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Variable independiente (días representados por el índice)
    X = datos_filtrados.index.values.reshape(-1, 1)
    # Variable dependiente (caudal)
    y = datos_filtrados['valor_promedio_relevado'].values

    # Dividir los datos en conjuntos de entrenamiento (70%) y evaluación (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False) # shuffle=False importante si el orden es temporal

    # Crear y entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de evaluación
    y_pred = modelo.predict(X_test)

    # Evaluar el modelo (métricas para regresión)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n----- Evaluación del Modelo de Regresión Lineal -----")
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinación (R^2): {r2:.2f}")
    print(f"Pendiente (Coeficiente): {modelo.coef_[0]:.2f}")
    print(f"Intercepto: {modelo.intercept_:.2f}")

    # ------------------------- Visualización del Modelo -------------------------
    plt.figure(figsize=(12, 7))
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Datos de Evaluación')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regresión Lineal')
    plt.scatter(X_train, y_train, color='green', alpha=0.4, label='Datos de Entrenamiento')
    plt.xlabel("Días (Índice)")
    plt.ylabel("Valor Promedio de Caudal [l/s]")
    plt.title("Modelo de Regresión Lineal para Predicción de Caudal (Usando el Tiempo como Índice)")
    plt.grid(True)
    plt.legend()
    plt.show()
    ################ HASTA ACÁ LO QUE NO ANDA, REVISAR
else:
    print("No se encontraron datos para 'SUBDELEGACION DEL RIO MENDOZA' en los archivos.")

# Observaciones
# Directorio que reconoce Python C:/Users/solva/OneDrive/Escritorio/CUNDINAMARCA/Machine_Learning/ProyectoFinal/Proyecto_Final_ML/Datasets
# Uso Python 3.11.9

    