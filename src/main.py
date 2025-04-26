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
    text: str


@app.post("/predict")
def predict(request: PredictionRequest, model=Depends(get_model), tokenizer=Depends(get_tokenizer)):
    inputs = tokenizer(request.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.tolist()
    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
