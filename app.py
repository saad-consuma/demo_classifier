import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chatbot import (
    init_model,
    init_prompt,
    generate_for_datapoint,
    generate_for_batch,   # <-- import batch helper
)

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
tokenizer, model = init_model(MODEL_NAME)
prompt = init_prompt()


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[str]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    try:
        preds = generate_for_batch(request.instances, tokenizer, model, prompt)
        return PredictResponse(predictions=preds)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
