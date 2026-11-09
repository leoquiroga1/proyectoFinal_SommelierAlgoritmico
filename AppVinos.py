import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# =====================================
# CARGA DEL MODELO
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.environ.get(
    "WINE_MODEL_PATH", os.path.join(BASE_DIR, "wine_quality_model.pkl")
)

with open(RUTA_MODELO, "rb") as f:
    modelo = pickle.load(f)

print(f"Modelo cargado desde {RUTA_MODELO}: {modelo}")

# =====================================
# DEFINICION DE LAS 11 VARIABLES
# clave del formulario/JSON -> (nombre de columna del modelo, minimo, maximo)
# =====================================

VARIABLES = {
    "fixed_acidity":        ("fixed acidity",        0.0, 20.0),
    "volatile_acidity":     ("volatile acidity",     0.0, 2.0),
    "citric_acid":          ("citric acid",          0.0, 1.5),
    "residual_sugar":       ("residual sugar",       0.0, 70.0),
    "chlorides":            ("chlorides",            0.0, 1.0),
    "free_sulfur_dioxide":  ("free sulfur dioxide",  0.0, 300.0),
    "total_sulfur_dioxide": ("total sulfur dioxide", 0.0, 500.0),
    "density":              ("density",              0.98, 1.05),
    "ph":                   ("pH",                   2.5, 4.5),
    "sulphates":            ("sulphates",            0.0, 2.5),
    "alcohol":              ("alcohol",              5.0, 20.0),
}


def validar_entrada(datos):
    """Valida las 11 variables. Devuelve (valores, errores)."""
    valores = {}
    errores = []

    for clave, (_, minimo, maximo) in VARIABLES.items():
        if clave not in datos or datos[clave] in (None, ""):
            errores.append(f"Falta la variable '{clave}'")
            continue
        try:
            valor = float(datos[clave])
        except (TypeError, ValueError):
            errores.append(f"'{clave}' debe ser numérico")
            continue
        if not (minimo <= valor <= maximo):
            errores.append(f"'{clave}' fuera de rango ({minimo} - {maximo})")
            continue
        valores[clave] = valor

    return valores, errores


def clasificar(valores):
    """Ejecuta el modelo y arma la respuesta."""
    fila = {VARIABLES[c][0]: valores[c] for c in VARIABLES}
    df = pd.DataFrame([fila])

    prediccion = int(modelo.predict(df)[0])
    probabilidad = float(modelo.predict_proba(df)[0][1])  # prob. de Premium

    return {
        "prediccion": prediccion,
        "categoria": "Premium" if prediccion == 1 else "Estándar",
        "probabilidad_premium": round(probabilidad, 4),
    }


def comentarios_complementarios(valores):
    comentarios = []
    if valores["alcohol"] >= 11:
        comentarios.append("✅ Nivel de alcohol favorable para vinos de alta calidad")
    else:
        comentarios.append("⚠️ Nivel de alcohol bajo para un vino premium")
    if valores["volatile_acidity"] < 0.5:
        comentarios.append("✅ Acidez volátil controlada")
    else:
        comentarios.append("⚠️ Acidez volátil elevada")
    if valores["sulphates"] >= 0.6:
        comentarios.append("✅ Buena concentración de sulfatos")
    else:
        comentarios.append("⚠️ Concentración de sulfatos baja")
    return comentarios


# =====================================
# API REST (JSON)
# =====================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modelo": os.path.basename(RUTA_MODELO)})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        return jsonify({"error": "El cuerpo debe ser un JSON con las 11 variables"}), 400

    valores, errores = validar_entrada(datos)
    if errores:
        return jsonify({"error": "Datos inválidos", "detalles": errores}), 422

    resultado = clasificar(valores)
    app.logger.info("Predicción: %s -> %s", valores, resultado)
    return jsonify(resultado), 200


# =====================================
# INTERFAZ WEB (usa la misma lógica que la API)
# =====================================

@app.route("/")
def inicio():
    return render_template("index.html", resultado=None, recomendacion=None, comentarios=None)


@app.route("/predecir", methods=["POST"])
def predecir():
    valores, errores = validar_entrada(request.form)
    if errores:
        return render_template(
            "index.html",
            resultado="❌ Datos inválidos",
            recomendacion=None,
            comentarios=errores,
        ), 422

    r = clasificar(valores)

    if r["prediccion"] == 1:
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

    recomendacion += f" (Probabilidad de ser Premium: {r['probabilidad_premium']:.0%})"

    return render_template(
        "index.html",
        resultado=resultado,
        recomendacion=recomendacion,
        comentarios=comentarios_complementarios(valores),
    )


# =====================================
# INICIO APP
# =====================================

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
