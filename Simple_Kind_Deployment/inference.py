from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the model
model = joblib.load("model.joblib")

# Define a prediction endpoint
@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([data["features"]])
    return {"prediction": prediction.tolist()}