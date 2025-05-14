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


# Ejemplo de uso:
ruta_de_la_carpeta = "C:/Users/solva/OneDrive/Escritorio/CUNDINAMARCA/Machine_Learning/ProyectoFinalDefinitivo/Datasets"
datos_filtrados = cargar_y_filtrar_multiples_csv(ruta_de_la_carpeta)

if not datos_filtrados.empty:
    print("\nPrimeras filas del DataFrame resultante:")
    print(datos_filtrados.head())
else:
    print("No se encontraron datos para 'SUBDELEGACION DEL RIO MENDOZA' en los archivos.")

#Observaciones
#Directorio que reconoce Python C:/Users/solva/OneDrive/Escritorio/CUNDINAMARCA/Machine_Learning/ProyectoFinal/Proyecto_Final_ML/Datasets
#Uso Python 3.11.9
