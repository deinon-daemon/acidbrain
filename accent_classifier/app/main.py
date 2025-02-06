import time
import torch
import logging
import asyncio
import pandas as pd
import traceback as tb
from typing import List
from models import AccentClassifier
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference.interfaces import foreign_class
from fastapi import FastAPI, UploadFile, File, HTTPException
from rest_models import PredictionResponse

app = FastAPI(title="Accent Classification API")
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create a lock for thread-safe initialization
init_lock = asyncio.Lock()
classifier = None
pool = None


async def get_classifier():
    global classifier, pool

    if classifier is None:
        async with init_lock:
            # Double-check pattern
            if classifier is None:
                logger.info("Initializing classifier...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    accent_model = foreign_class(
                        source="warisqr7/accent-id-commonaccent_xlsr-en-english",
                        pymodule_file="custom_interface.py",
                        classname="CustomEncoderWav2vec2Classifier",
                        savedir="pretrained_model",
                        run_opts={"device": device},
                    )
                    classifier = AccentClassifier(accent_model)
                    # Initialize the thread pool after classifier is ready
                    pool = ThreadPoolExecutor(max_workers=1)
                    logger.info("Classifier initialization complete")
                except Exception as e:
                    logger.error(f"Failed to initialize classifier: {str(e)}")
                    raise HTTPException(
                        status_code=500, detail="Failed to initialize the classifier"
                    )

    return classifier


@app.post("/predict/", response_model=PredictionResponse)
async def predict_accent(file: UploadFile = File(...)):
    ts = time.time()

    try:
        # Ensure classifier is initialized
        classifier = await get_classifier()

        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Content type: {file.content_type}")

        content = await file.read()
        logger.debug(f"Content length: {len(content)} bytes")

        # Use the thread pool to run CPU-intensive operations
        loop = asyncio.get_event_loop()
        prediction_response = await loop.run_in_executor(
            pool, classifier.classify_bytes, content
        )

        prediction_response["filename"] = file.filename
        logger.info(f"Prediction completed in {int((time.time() - ts) * 1000)}ms")

        return PredictionResponse(**prediction_response)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Full traceback: {tb.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict/")
async def batch_predict(files: List[UploadFile] = File(...)):
    ts = time.time()

    try:
        # Ensure classifier is initialized
        classifier = await get_classifier()

        processed_audio = []
        temp_data = []
        for i, file in enumerate(files):
            content = await file.read()
            waveform, success = classifier.preprocess_audio(content)
            processed_audio.append(waveform)
            temp_data.append(
                {
                    "index": i,
                    "filename": file.filename,
                    "preprocessing_status": success,
                }
            )

        df = pd.DataFrame(temp_data)
        # Use thread pool & EncoderClassifier.classify_batch to parallelize work
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            pool, classifier.classify_batch, df, processed_audio
        )

        predictions = predictions.to_dict("records")
        print(f"batch predictions: {predictions}")

        logger.info(f"Batch prediction completed in {int((time.time() - ts) * 1000)}ms")
        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        logger.error(f"Full traceback: {tb.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
