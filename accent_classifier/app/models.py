# src/models.py
import io
import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
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

            return waveform, True

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

        # Process the audio through the model
        # Note: We bypass classify_file() and use the internal processing directly
        # so we don't have to save audio data to fs and slow the worker / server w/
        # excessive reads and writes
        outputs = self.model.encode_batch(waveform)
        outputs = self.model.mods.output_mlp(outputs)
        out_prob = self.model.hparams.softmax(outputs)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.model.hparams.label_encoder.decode_torch(index)

        return {
            "prediction": text_lab,
            "score": score.item(),
            "probabilities": out_prob.tolist(),
            "embeddings": outputs.tolist(),
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

                # Process each waveform in the batch individually
                for b in batch:
                    # Convert waveform to tensor
                    waveform = b.clone().detach()  # Fix tensor conversion warning

                    # Ensure the waveform is mono (1 channel)
                    if waveform.dim() == 2:  # Stereo audio (2, length)
                        waveform = waveform.mean(dim=0)  # Convert to mono by averaging channels
                    elif waveform.dim() == 1:  # Mono audio (length,)
                        waveform = waveform.unsqueeze(0)  # Add a channel dimension

                    # Move waveform to the same device as the model
                    waveform = waveform.to(self.model.device)

                    # Classify the waveform
                    output = self.model.encode_batch(waveform.unsqueeze(0))  # Add batch dimension
                    output = self.model.mods.output_mlp(output).squeeze(1)
                    out_prob = self.model.hparams.softmax(output)
                    score, index = torch.max(out_prob, dim=-1)
                    pred = self.model.hparams.label_encoder.decode_torch(index)

                    # Append results
                    predictions.append(pred[0])  # Extract prediction from batch
                    scores.append(score.item())

        # Add predictions and scores to the DataFrame
        df["prediction"] = predictions
        df["score"] = scores
        return df
