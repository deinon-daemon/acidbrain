import pytest
import aiofiles
from pathlib import Path
from fastapi.testclient import TestClient
from main import app
test_client = TestClient(app)
# Constants
TEST_DATA_DIR = Path(__file__).parent / "test_data"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}




@pytest.fixture
def test_audio_files():
    """Get list of test audio files from test_data directory"""
    files = []
    for file in TEST_DATA_DIR.iterdir():
        if file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            files.append(file)
    return files

async def create_test_file_payload(filepath):
    """Helper function to create file payload from filepath"""
    async with aiofiles.open(filepath, "rb") as f:
        content = await f.read()
    return {"file": (filepath.name, content, "audio/wav")}


@pytest.mark.asyncio
async def test_single_file_prediction(test_audio_files):
    """Test single file prediction endpoint"""
    if not test_audio_files:
        pytest.skip("No test audio files found in test_data directory")

    for audio_file in test_audio_files:
        print(f"\nTesting file: {audio_file}")
        files = await create_test_file_payload(audio_file)
        print(f"File payload created, size: {len(files['file'][1])} bytes")

        response = test_client.post("/predict/", files=files)

        if response.status_code != 200:
            print(f"Error response: {response.text}")

        assert response.status_code == 200


@pytest.mark.asyncio
async def test_batch_prediction(test_audio_files):
    """Test batch prediction endpoint"""
    # Skip if not enough test files
    if len(test_audio_files) < 2:
        pytest.skip("Not enough test audio files for batch testing")

    # Create payload for multiple files
    files = []
    for audio_file in test_audio_files:
        file_payload = await create_test_file_payload(audio_file)
        files.append(("files", file_payload["file"]))

    response = test_client.post("/batch-predict/", files=files)

    assert response.status_code == 200
    data = response.json()
    print(f"Test Server Inference Pipeline Response: {data=}")

    assert "predictions" in data
    predictions = data["predictions"]
    assert len(predictions) == len(test_audio_files)


@pytest.mark.asyncio
async def test_prediction_performance(test_audio_files):
    """Test prediction endpoint performance"""
    import time

    if not test_audio_files:
        pytest.skip("No test audio files found")

    audio_file = test_audio_files[0]
    files = await create_test_file_payload(audio_file)

    start_time = time.time()
    response = test_client.post("/predict/", files=files)
    end_time = time.time()

    assert response.status_code == 200
    processing_time = end_time - start_time

    # Add performance assertions based on your requirements
    assert processing_time < 5.0  # Example: prediction should take less than 5 seconds
