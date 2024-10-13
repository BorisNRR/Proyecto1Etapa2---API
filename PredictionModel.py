from joblib import load, dump

class Model:

    def __init__(self):
        # Cargar el pipeline cuando se crea la instancia de la clase
        try:
            self.pipeline = load("unfpa_text_classification_pipeline.pkl")
        except Exception as e:
            raise Exception(f"Error loading pipeline: {str(e)}")

    # Función para realizar predicciones
    def make_predictions(self, data):
        try:
            prediction = self.pipeline.predict(data['Textos_espanol'])
            probability = self.pipeline.predict_proba(data['Textos_espanol'])
            return {"prediction": int(prediction[0]), "probability": probability[0].max()}
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

    # Función para reentrenar el modelo
    def retrain_model(self, data, labels):
        try:
            self.pipeline.fit(data['Textos_espanol'], labels)
            dump(self.pipeline, 'unfpa_text_classification_pipeline.pkl')  # Guardar el pipeline reentrenado
        except Exception as e:
            raise Exception(f"Error in retraining: {str(e)}")
