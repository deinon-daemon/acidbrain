# src/models.py
import io
import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from speechbrain.inference.classifiers import EncoderClassifier

class AccentClassifier:
    def __init__(self, model: EncoderClassifier):
        self.model = model

    def preprocess_audio(
        self,
        audio_bytes: bytes | None = None,
        filepath: str | None = None,
        sample_rate: int = 16000,
        max_length: float = 30.0,
    ) -> torch.Tensor:
        """
        Processes audio files with robust error handling and validation

        Args:
            filepath (str): Path to audio file
            max_length (float): max length of audio sample in seconds
            sample_rate (int): Target sample rate

        Returns:
            numpy.ndarray: Processed audio array
            bool: Success status
        """
        try:

            if audio_bytes:
                buffer = io.BytesIO(audio_bytes)
                waveform, orig_sr = torchaudio.load(buffer)

            elif filepath:
                # Verify file exists and is accessible
                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                    return None, False

                # Load audio with error handling
                try:
                    waveform, orig_sr = torchaudio.load(filepath, normalize=True)
                except Exception as e:
                    print(f"Error loading audio {filepath}: {e}")
                    return None, False

                # Validate audio data
                if waveform.nelement() == 0:
                    print(f"Empty audio file: {filepath}")
                    return None, False

            else:
                raise ValueError(
                    "Invalid Input! Either filepath or audio_bytes must be not None!"
                )

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Create resampler with bounds checking
            if orig_sr != sample_rate:
                try:
                    resampler = torchaudio.transforms.Resample(
                        orig_sr, sample_rate, dtype=waveform.dtype
                    )
                    waveform = resampler(waveform)
                except Exception as e:
                    print(f"Resampling error for {filepath}: {e}")
                    return None, False

            # Calculate target samples
            target_samples = int(max_length * sample_rate)
            current_samples = waveform.size(1)

            # Handle empty or invalid audio
            if current_samples == 0:
                print(f"Invalid audio length in {filepath}")
                return None, False

            # Trim if needed
            if current_samples > target_samples:
                # Take center portion
                start = (current_samples - target_samples) // 2
                waveform = waveform[:, start : start + target_samples]

            return waveform.numpy().squeeze(), True

        except Exception as e:
            print(f"Unexpected error processing {filepath}: {e}")
            return None, False

    def _classify(self, waveform):
        """
        Process audio data directly from bytes using SpeechBrain without saving to disk

        Parameters:
            audio_bytes (bytes): Raw audio data in bytes

        Returns:
            dict: Classification results from the model
        """

        # Move to the same device as the model
        waveform = waveform.to(self.model.device)

        # Process the audio through the model
        # Note: We bypass classify_file() and use the internal processing directly
        # so we don't have to save audio data to fs and slow the worker / server w/
        # excessive reads and writes
        features = self.model.mods.compute_features(waveform)
        embeddings = self.model.mods.embedding_model(features)
        outputs = self.model.mods.classifier(embeddings)

        # Get predictions
        probs = torch.softmax(outputs, dim=-1)
        score, pred_idx = torch.max(probs, dim=-1)

        # Convert prediction to label using the model's label encoder
        predicted_label = self.model.hparams.label_encoder.decode_ndim(pred_idx)

        return {
            "prediction": predicted_label,
            "score": score.item(),
            "probabilities": probs.squeeze().tolist(),
            "embeddings": embeddings.squeeze().tolist(),
        }

    def classify_bytes(self, audio_bytes: str) -> Tuple[torch.Tensor, float, int, str]:
        waveform, success = self.preprocess_audio(audio_bytes=audio_bytes)
        return self._classify(waveform)

    def classify_batch(
        self, df: pd.DataFrame, processed_audio: list, batch_size: int = 8
    ) -> pd.DataFrame:

        predictions = []
        scores = []

        if self.model.device == "cuda":
            print(f"Optimizing batch size for gpu {batch_size} => 32")
            batch_size = 32

        for i in tqdm(range(0, len(processed_audio), batch_size)):
            batch = processed_audio[i : i + batch_size]
            waveforms = torch.from_numpy(batch).float()

            # Use classify_batch for optimized inference
            _, batch_scores, _, batch_preds = self.model.classify_batch(waveforms)
            predictions.extend(batch_preds)
            scores.extend(batch_scores)

        df["prediction"] = predictions
        df["score"] = [float(s) for s in scores]
        return df
