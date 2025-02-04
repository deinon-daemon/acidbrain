# src/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from models import AccentClassifier
import torch
import torchaudio
import numpy as np
from pydantic import BaseModel
from typing import List, Dict
import logging
from pathlib import Path
import tempfile

app = FastAPI(title="Accent Classification API")


class PredictionResponse(BaseModel):
    accent: str
    confidence: float
    all_predictions: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    file_urls: List[str]


@app.on_event("startup")
async def load_model():
    global classifier
    try:
        # Initialize your SpeechBrain classifier here
        classifier = AccentClassifier.load_model()
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


@app.post("/predict/", response_model=PredictionResponse)
async def predict_accent(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Run prediction
        out_prob, score, index, text_lab = classifier.classify_file(temp_path)

        # Format response
        predictions = {label: float(prob) for label, prob in zip(text_lab, out_prob[0])}

        return PredictionResponse(
            accent=text_lab[0], confidence=float(score), all_predictions=predictions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception as _:
            pass


@app.post("/batch-predict/")
async def batch_predict(files: List[UploadFile] = File(...)):
    predictions = []
    temp_files = []

    try:
        # Save all files temporarily
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)

        # Process in smaller batches for CPU
        batch_size = 8  # Reduced batch size for CPU processing
        for i in range(0, len(temp_files), batch_size):
            batch = temp_files[i : i + batch_size]
            batch_predictions = []

            for file_path in batch:
                out_prob, score, index, text_lab = classifier.classify_file(file_path)
                batch_predictions.append(
                    {
                        "filename": Path(file_path).name,
                        "accent": text_lab[0],
                        "confidence": float(score),
                    }
                )

            predictions.extend(batch_predictions)

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            Path(temp_file).unlink(missing_ok=True)
