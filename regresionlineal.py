import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

# Cargar el modelo entrenado
modelo = joblib.load("modelo_regresion_CaudalVsNieve.joblib")

def calcularCaudal(nivel_nieve):
    """Recibe nivel de nieve y devuelve la predicción del caudal"""
    nivel_nieve = np.array([[nivel_nieve]])  # 2D array para sklearn
    caudal_predicho = modelo.predict(nivel_nieve)
    return round(caudal_predicho[0], 2)

def grafica_regresion(nivel_nieve_input):
    """Genera una gráfica de regresión mostrando la predicción"""
    nivel_rango = np.linspace(100, 2400, 100).reshape(-1, 1)
    caudal_predicho = modelo.predict(nivel_rango)

    # Punto ingresado
    caudal_input = modelo.predict([[nivel_nieve_input]])

    plt.figure(figsize=(10, 6))
    plt.plot(nivel_rango, caudal_predicho, color='red', label='Modelo de Regresión')
    plt.scatter(nivel_nieve_input, caudal_input, color='blue', s=100, label=f'Entrada: {nivel_nieve_input} m')
    plt.xlabel('Nivel de Nieve [m]')
    plt.ylabel('Caudal [l/s]')
    plt.title('Regresión Lineal: Nivel de Nieve vs Caudal')
    plt.legend()
    plt.grid(True)

    # Guardar imagen
    filename = f"static/plot_{uuid.uuid4().hex}.png"
    plt.savefig(filename)
    plt.close()
    return filename