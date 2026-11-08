from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# =====================================
# CARGA DEL MODELO
# =====================================

with open("wine_quality_model.pkl", "rb") as f:
    modelo = pickle.load(f)

print("Modelo cargado:")
print(modelo)

# =====================================
# PAGINA PRINCIPAL
# =====================================

@app.route("/")
def inicio():
    return render_template(
        "index.html",
        resultado=None,
        recomendacion=None,
        comentarios=None
    )

# =====================================
# PREDICCION
# =====================================

@app.route("/predecir", methods=["POST"])
def predecir():

    alcohol = float(request.form["alcohol"])
    volatile_acidity = float(request.form["volatile_acidity"])
    sulphates = float(request.form["sulphates"])

    datos = pd.DataFrame([{
        "fixed acidity": float(request.form["fixed_acidity"]),
        "volatile acidity": volatile_acidity,
        "citric acid": float(request.form["citric_acid"]),
        "residual sugar": float(request.form["residual_sugar"]),
        "chlorides": float(request.form["chlorides"]),
        "free sulfur dioxide": float(request.form["free_sulfur_dioxide"]),
        "total sulfur dioxide": float(request.form["total_sulfur_dioxide"]),
        "density": float(request.form["density"]),
        "pH": float(request.form["ph"]),
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    prediccion = modelo.predict(datos)[0]

    # =====================================
    # ANALISIS COMPLEMENTARIO
    # =====================================

    comentarios = []

    if alcohol >= 11:
        comentarios.append(
            "✅ Nivel de alcohol favorable para vinos de alta calidad"
        )

    if volatile_acidity < 0.5:
        comentarios.append(
            "✅ Acidez volátil controlada"
        )

    if sulphates >= 0.6:
        comentarios.append(
            "✅ Buena concentración de sulfatos"
        )

    # =====================================
    # RESULTADO DEL MODELO
    # =====================================

    if prediccion == 1:

        resultado = "🍷 VINO PREMIUM"

        recomendacion = (
            "Calidad estimada igual o superior a 7. "
            "Se recomienda proceder con el embotellado de reserva especial."
        )

    else:

        resultado = "🍷 VINO ESTÁNDAR"

        recomendacion = (
            "Calidad estimada inferior a 7. "
            "Se recomienda destinar el producto a la línea comercial estándar."
        )

    return render_template(
    "index.html",
    resultado=resultado,
    recomendacion=recomendacion,
    comentarios=comentarios
)

# =====================================
# INICIO APP
# =====================================

if __name__ == "__main__":
    app.run(debug=True)