# src/models.py
from dataclasses import dataclass
from pathlib import Path
import os
import torch
import torchaudio
import pandas as pd
from typing import List, Tuple, Optional
from speechbrain.inference.interfaces import foreign_class
from speechbrain.inference.classifiers import EncoderClassifier


@dataclass
class AccentClassifier:
    model_path: Path
    model_name: str = "warisqr7/accent-id-commonaccent_xlsr-en-english"
    hardware: str = "cpu"
    model = None

    def __post_init__(self):
        self.model = self.load_model()
        self.model.eval()

    def load_model(self):
        # Load pretrained model with custom class
        if self.model_name == "warisqr7/accent-id-commonaccent_xlsr-en-english":
            classifier = foreign_class(
                source="warisqr7/accent-id-commonaccent_xlsr-en-english",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir="pretrained_model",
                run_opts={"device": self.hardware},
            )
        else:
            classifier = EncoderClassifier(
                source=self.model_name,
                savedir="pretrained_model",
                run_opts={"device": self.hardware},
            )

        self.model = classifier
        return classifier

    def preprocess_audio(
        self, filepath: str, sample_rate: int = 16000, target_length: float = 10.0
    ) -> torch.Tensor:
        """
        Processes audio files with robust error handling and validation

        Args:
            filepath (str): Path to audio file
            target_length (float): Desired length in seconds
            sample_rate (int): Target sample rate

        Returns:
            numpy.ndarray: Processed audio array
            bool: Success status
        """
        try:
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
            target_samples = int(target_length * sample_rate)
            current_samples = waveform.size(1)

            # Handle empty or invalid audio
            if current_samples == 0:
                print(f"Invalid audio length in {filepath}")
                return None, False

            # Trim or pad
            if current_samples > target_samples:
                # Take center portion
                start = (current_samples - target_samples) // 2
                trimmed = waveform[:, start : start + target_samples]
            else:
                # Pad with zeros
                padding = target_samples - current_samples
                trimmed = torch.nn.functional.pad(waveform, (0, padding))

            # Final validation
            if trimmed.size(1) != target_samples:
                print(f"Unexpected output size for {filepath}")
                return None, False

            return trimmed.numpy().squeeze(), True

        except Exception as e:
            print(f"Unexpected error processing {filepath}: {e}")
            return None, False

    def classify_file(self, audio_path: str) -> Tuple[torch.Tensor, float, int, str]:
        waveform = self.preprocess_audio(audio_path)
        with torch.no_grad():
            output = self.model(waveform)
        return self.process_output(output)

    def classify_batch(
        self, waveforms: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float], List[int], List[str]]:
        with torch.no_grad():
            outputs = self.model(waveforms)
        return self.process_batch_output(outputs)

    @staticmethod
    def process_output(audio_path: str) -> dict:
        _, score, _, text_lab = self.model.classify_file(
            audio_path
        )
        return {
            "prediction": text_lab[0],
            "score": score
        }

    @staticmethod
    def process_batch_output(
        df: pd.DataFrame,
        processed_audio: list,
        batch_size: int = 8
    ) -> pd.DataFrame:

        predictions = []
        scores = []

        if self.hardware == "cpu" and batch_size > 8:
            batch_size = 8

        for i in tqdm(range(0, len(processed_audio), batch_size)):
            batch = processed_audio[i:i+batch_size]
            waveforms = torch.from_numpy(batch).float()

            # Use classify_batch for optimized inference
            _, batch_scores, _, batch_preds = self.model.classify_batch(waveforms)
            predictions.extend(batch_preds)
            scores.extend(batch_scores)


        df["prediction"] = predictions
        df["score"] = [float(s) for s in scores]
        return df
