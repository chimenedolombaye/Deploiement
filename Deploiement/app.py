from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load('model.pkl')

# Variables pour le suivi des transactions
total_transactions = 0
fraudulent_transactions_detected = 0

# Définir un point de terminaison pour effectuer des prédictions
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    global total_transactions, fraudulent_transactions_detected

    try:
        # Récupérer les données du formulaire
        V1 = float(request.form.get("V1"))
        V2 = float(request.form.get("V2"))
        Amount = float(request.form.get("Amount"))

        # Faire la prédiction avec le modèle
        prediction = model.predict([[V1, V2, Amount]])

        # Mettre à jour les variables de suivi
        total_transactions += 1
        if int(prediction[0]) == 1:  # Si la transaction est détectée comme frauduleuse
            fraudulent_transactions_detected += 1

        # Calculer le taux de fraude
        fraud_rate = (fraudulent_transactions_detected / total_transactions) * 100

        # Retourner la prédiction et le taux de fraude au format JSON
        return jsonify({"prediction": int(prediction[0]), "fraud_rate": fraud_rate})

    except ValueError as e:
        return jsonify({"error": "Valeur de fonctionnalité manquante ou invalide"}), 400

if __name__ == "__main__":
    app.run(debug=True)
