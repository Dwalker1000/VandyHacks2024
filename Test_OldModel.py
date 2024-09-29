import os
import queue
import sys
import sounddevice as sd
import vosk
import json
import numpy as np
import torch
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the Wav2vec2 class
class Wav2vec2(OnnxInferenceModel):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(model_path, *args, **kwargs)
        self.input_name = "input"  # This should match the name expected by your ONNX model
        self.output_name = "output"  # Modify if your model has a different output name
        self.metadata = {"vocab": []}  # Load your vocab here if needed

        # Load your model here (assuming you have a method to load the ONNX model)
        self.model = self.load_model(model_path)

    def predict(self, audio: np.ndarray):
        audio = np.expand_dims(audio, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: audio})[0]

        # Debugging output
        logging.debug(f"Raw Predictions: {preds}")

        # Apply softmax to get probabilities
        preds = preds.squeeze()  # Remove unnecessary dimensions
        probabilities = F.softmax(torch.tensor(preds), dim=0).detach().numpy()

        # Get the index of the max probability
        predicted_index = np.argmax(probabilities)
        return predicted_index

    def load_model(self, model_path):
        # This method should load your ONNX model and return it
        import onnxruntime
        return onnxruntime.InferenceSession(model_path)

# Path to the Vosk model
model_path = "C:\\Users\\25wal\\OneDrive\\Desktop\\Vanderbilt\\2025\\CS Module\\AI_Therapist2_2024\\vosk-model-small-en-us-0.15"

# Load the Vosk model
if not os.path.exists(model_path):
    logging.error("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

vosk_model = vosk.Model(model_path)
q = queue.Queue()

# Load the Wav2Vec2 model
wav2vec_model = Wav2vec2(model_path="Emotion-Model/model.onnx")

# Emotion mapping (modify based on your actual model output)
emotion_mapping = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Neutral",
    # Add other emotions as necessary
}

# Audio callback function
def audio_callback(indata, frames, time, status):
    if status:
        logging.warning(status)  # Log any warnings from the status
    q.put(bytes(indata))  # Put the audio data in the queue

# Configure the audio stream
try:
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        logging.info('#' * 80)
        logging.info('Press Ctrl+C to stop the recording')
        logging.info('#' * 80)

        rec = vosk.KaldiRecognizer(vosk_model, 16000)

        # Prepare for evaluation metrics
        accum_cer, accum_wer = [], []

        # Load validation dataset
        val_dataset = pd.read_csv("Emotion-Model/val.csv").values.tolist()
        logging.info("Validation dataset loaded successfully.")

        pbar = tqdm(val_dataset)

        while True:
            logging.info("Waiting for audio data...")
            data = q.get()
            logging.info("Received audio data.")

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result['text']

                # Convert audio to NumPy array for Wav2Vec2 prediction
                audio_array = np.frombuffer(data, dtype=np.int16)
                audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

                # Use Wav2Vec2 to predict the index of the emotion
                predicted_emotion_index = wav2vec_model.predict(audio_array)

                # Get the corresponding emotion
                predicted_emotion = emotion_mapping.get(predicted_emotion_index, "Unknown Emotion")

                # Compare prediction with ground truth for the current audio
                for vaw_path, label in pbar:
                    cer = get_cer(predicted_emotion, label)  # Assuming label corresponds to emotions
                    wer = get_wer(predicted_emotion, label)

                    accum_cer.append(cer)
                    accum_wer.append(wer)

                    # Print only the predicted emotion
                    logging.info(f"Predicted Emotion: {predicted_emotion}")

                    pbar.set_description(
                        f"Average CER: {np.average(accum_cer):.4f}, Average WER: {np.average(accum_wer):.4f}")
            else:
                result = json.loads(rec.PartialResult())
                logging.info(result['partial'])
except Exception as e:
    logging.error(f"Error during audio input: {e}")
