from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load('model.pkl')

# Définir un point de terminaison pour effectuer des prédictions
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données du formulaire
        V1 = float(request.form.get("V1"))
        V2 = float(request.form.get("V2"))
        Amount = float(request.form.get("Amount"))


        # Faire la prédiction avec le modèle
        prediction = model.predict([[V1, V2, Amount]])

        # Retourner la prédiction au format JSON
        return jsonify({"prediction": int(prediction[0])})

    except ValueError as e:
        return jsonify({"error": "Valeur de fonctionnalité manquante ou invalide"}), 400

if __name__ == "__main__":
    app.run(debug=True)
