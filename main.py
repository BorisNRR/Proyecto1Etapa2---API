from fastapi import FastAPI, HTTPException
import pandas as pd
import DataModel
import PredictionModel

app = FastAPI()

# Instancia de PredictionModel
model = PredictionModel.Model()

@app.get("/")
def read_root():
    return {"message": "UNFPA Text Classification API"}

# Endpoint de predicción
@app.post("/predict/")
def predict(dataModel: DataModel.DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    try:
        result = model.make_predictions(df)
        return {"prediction": result['prediction'], "probability": result['probability']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# Endpoint para reentrenar el modelo
@app.post("/retrain/")
def retrain(dataModel: DataModel.RetrainModel):
    df = pd.DataFrame(dataModel.new_data, columns=dataModel.columns())
    labels = dataModel.new_labels
    try:
        message = model.retrain_model(df, labels)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in retraining: {str(e)}")
