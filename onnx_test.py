# /// script
# dependencies = [
#   "torch==2.6.0",
#   "onnx==1.17.0",
#   "onnxruntime==1.20.1",
#   "onnxscript==0.1.0",
#   "huggingface_hub",
#   "loguru",
#   "misaki[en]>=0.7.6",
#   "numpy==1.26.4",
#   "scipy",
#   "transformers",
#   "soundfile",
#   "pip"
# ]
# ///

import time
import onnxruntime as ort
import numpy as np
import torch
from kokoro.model import KModel
from kokoro.pipeline import KPipeline
from loguru import logger
import soundfile as sf
import json
from huggingface_hub import hf_hub_download
from torch.nn.functional import mse_loss

# Load the ONNX model
onnx_path = "kokoro.onnx"
session = ort.InferenceSession(onnx_path)

# Initialize the pipeline
pipeline = KPipeline(lang_code='a', model=False)

# Load vocabulary from Hugging Face
repo_id = "hexgrad/Kokoro-82M"
config_filename = "config.json"
config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
vocab = config["vocab"]

# Initialize the KModel
torch_model = KModel().eval()

# Example text and voice
text = "Hello! This is a test of the Kokoro text-to-speech system."
voice = "af_heart"

# Tokenize and phonemize
_, tokens = pipeline.g2p(text)

# Use en_tokenize to process tokens
for graphemes, phonemes, token_list in pipeline.en_tokenize(tokens):
    with torch.no_grad():
        # Convert phonemes to input_ids
        input_ids = torch.LongTensor([[0, *map(lambda p: vocab.get(p), phonemes), 0]])
        print("graphemes", graphemes)
        print("phonemes", phonemes)
        print("token_list", token_list)
        print("input_ids", input_ids)
        input_lengths = torch.LongTensor([input_ids.shape[-1]])
        print("input_lengths", input_lengths)
        
        # Load and process the style vector
        ref_s = pipeline.load_voice(voice)
        ref_s = ref_s[input_ids.shape[1] - 1]  # Select the appropriate style vector

        # Run the PyTorch model
        torch_output, torch_pred_dur, torch_intermediates = torch_model.model(
            input_ids=input_ids, input_lengths=input_lengths, ref_s=ref_s, speed=1.0
        )

        # Run the ONNX model
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "input_lengths": input_lengths.numpy(),
            "ref_s": ref_s.numpy(),
            "speed": np.array([1.0], dtype=np.float64)
        }
        ort_outputs = session.run(None, ort_inputs)

        # Compare outputs using MSE
        for name, torch_intermediate in torch_intermediates.items():
            onnx_output = ort_outputs[list(torch_intermediates.keys()).index(name) + 2]  # +2 to skip audio and pred_dur
            onnx_output_tensor = torch.tensor(onnx_output)

            # Calculate MSE
            mse = mse_loss(torch_intermediate.flatten(), onnx_output_tensor.flatten())
            logger.info(f"MSE for {name}: {mse.item():.5f}")

        # Write audio to file
        torch_audio = torch_output.cpu().numpy()
        onnx_audio = ort_outputs[0]  # Assuming the first output is audio
        
        # Calculate MSE for audio outputs
        audio_mse = mse_loss(torch.tensor(torch_audio).flatten(), torch.tensor(onnx_audio).flatten())
        logger.info(f"MSE for audio output: {audio_mse.item():.5f}")

        sf.write('torch_output.wav', torch_audio, 24000)  # Assuming a sample rate of 24000 Hz
        sf.write('onnx_output.wav', onnx_audio, 24000)

        logger.info("Audio comparison complete. Files written: 'torch_output.wav', 'onnx_output.wav'.")

