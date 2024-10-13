from pydantic import BaseModel

# Modelo de datos para predicciones
class DataModel(BaseModel):
    Textos_espanol: str

    def columns(self):
        return ["Textos_espanol"]

# Modelo de datos para el reentrenamiento
class RetrainModel(BaseModel):
    new_data: list
    new_labels: list

    def columns(self):
        return ["Textos_espanol", "sdg"]
