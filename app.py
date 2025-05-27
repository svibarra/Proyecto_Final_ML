from flask import Flask, request, render_template
from regresionlineal import calcularCaudal, grafica_regresion

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    grafico_url = None

    if request.method == 'POST':
        try:
            nivel_nieve = float(request.form['Nivel_Nieve'])
            resultado = calcularCaudal(nivel_nieve)
            grafico_url = grafica_regresion(nivel_nieve)
        except (ValueError, KeyError):
            resultado = "Por favor ingresa un valor numérico válido."

    return render_template('index.html', resultado=resultado, grafico_url=grafico_url)



if __name__ == '__main__':
    app.run(debug=True)
