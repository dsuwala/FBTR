import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import torch
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        )


def get_model():
    logger.info("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained("./model", local_files_only=True)
    return model


def get_tokenizer():
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./model", local_files_only=True)
    return tokenizer


class PredictionRequest(BaseModel):
    smiles: str


#TODO: summarize prediction to a 
@app.post("/predict")
def predict(request: PredictionRequest, model=Depends(get_model), tokenizer=Depends(get_tokenizer)):
    tastes = ["bitter", "sour", "sweet", "umami", "undefined"]

    inputs = tokenizer(request.smiles, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.tolist()

    predictions = predictions[0]
    logger.info(f"Predicted values {predictions}")
    leading_taste_index = predictions.index(max(predictions))
    logger.info(f"Leading taste index {leading_taste_index} for taste {tastes[leading_taste_index]}")

    return {"predictions": tastes[leading_taste_index]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
