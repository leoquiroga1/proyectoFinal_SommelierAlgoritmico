from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar modelo
with open("wine_quality_model.pkl", "rb") as f:
    modelo = pickle.load(f)


@app.route("/")
def inicio():
    return render_template(
        "index.html",
        resultado=None,
        probabilidad=None
    )


@app.route("/predecir", methods=["POST"])
def predecir():

    alcohol = float(request.form["alcohol"])
    volatil = float(request.form["volatile"])

    datos = pd.DataFrame([{
        "fixed acidity": 7.0,
        "volatile acidity": volatil,
        "citric acid": 0.1,
        "residual sugar": 2.0,
        "chlorides": 0.07,
        "free sulfur dioxide": 10,
        "total sulfur dioxide": 40,
        "density": 0.997,
        "pH": 3.4,
        "sulphates": 0.6,
        "alcohol": alcohol
    }])

    prediccion = modelo.predict(datos)[0]
    probabilidad = modelo.predict_proba(datos)[0]

    resultado = "🍷 PREMIUM" if prediccion == 1 else "🍷 ESTÁNDAR"

    return render_template(
        "index.html",
        resultado=resultado,
        probabilidad=round(probabilidad[1] * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)