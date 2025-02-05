# src/main.py
import time
import torch
import logging
import asyncio
import pandas as pd
from typing import List
from models import AccentClassifier
from fastapi.responses import JSONResponse
from concurrent.futures import ProcessPoolExecutor
from speechbrain.inference.interfaces import foreign_class
from fastapi import FastAPI, UploadFile, File, HTTPException
from rest_models import PredictionResponse  # , BatchPredictionRequest


app = FastAPI(title="Accent Classification API")


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_classifier_model():
    global classifier
    device = "cuda" if torch.cuda_is_available() else "cpu"
    accent_model = foreign_class(
        source="warisqr7/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir="pretrained_model",
        run_opts={"device": device},
    )
    accent_model.eval()
    classifier = AccentClassifier(accent_model)


pool = ProcessPoolExecutor(max_workers=1, initializer=create_classifier_model)


@app.post("/predict/", response_model=PredictionResponse)
async def predict_accent(file: UploadFile = File(...)):
    ts = time.time()
    prediction_response = {}
    loop = asyncio.get_event_loop()
    try:

        print(f"Processing file: {file.filename}")
        print(f"Content type: {file.content_type}")

        content = await file.read()
        print(f"Content length: {len(content)} bytes")

        # Try to log the first few bytes to see if content is valid
        print(f"First few bytes: {content[:20]}")

        prediction_response = await loop.run_in_executor(
            pool, lambda: classifier.classify_bytes(content)
        )
        prediction_response["filename"] = file.filename
        print(f"Model : {int((time.time() - ts) * 1000)}ms")
        return PredictionResponse(**prediction_response)
    except Exception as e:
        print(f"Error details: {str(e)}")
        print(f"Error type: {type(e)}")
        # Print full traceback
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict/")
async def batch_predict(files: List[UploadFile] = File(...)):
    predictions = []
    temp_data = []
    ts = time.time()
    try:
        processed_audio = []
        # Save all files temporarily and keep track of filenames
        for i, file in enumerate(files):
            content = await file.read()
            waveform, success = classifier.preprocess_audio(content)
            processed_audio.append(waveform)
            processed_audio.append(
                {"index": i, "filename": file.filename, "preprocessing_status": success}
            )

        # Assuming your batch predictions code here...
        df = pd.DataFrame(temp_data)
        predictions = AccentClassifier.classify_batch(df, processed_audio)
        print(f"Model Batch Job : {int((time.time() - ts) * 1000)}ms")
        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
