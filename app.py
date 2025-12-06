import os
import json
import sys, traceback
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
    instances: Any

class PredictResponse(BaseModel):
    predictions: List[str]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="No instances provided")
    
    norm_instances = []
    for i in request.instances:
        if isinstance(i, str):
            data_dict = json.loads(i)
            if isinstance(data_dict, list):
                data_dict = data_dict[0]
            if isinstance(data_dict, dict):
                norm_instances.append(data_dict)
        elif isinstance(i, dict):
            norm_instances.append(i)

    try:
        preds = generate_for_batch(norm_instances, tokenizer, model, prompt)
        return PredictResponse(predictions=preds)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
