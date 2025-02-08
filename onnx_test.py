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
from typing import Dict
from huggingface_hub import hf_hub_download

def load_onnx_model(onnx_path: str):
    """Load the ONNX model."""
    return ort.InferenceSession(onnx_path)

def run_inference(session, input_ids, input_lengths, ref_s, speed=1):
    """Run inference using the ONNX model."""
    inputs = {
        'input_ids': input_ids.numpy(),
        'input_lengths': input_lengths.numpy(),
        'ref_s': ref_s.numpy(),
        'speed': np.array([speed], dtype=np.float64)
    }
    outputs = session.run(None, inputs)
    return outputs

def load_vocab_from_hf(repo_id: str, config_filename: str) -> Dict[str, int]:
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["vocab"]

def main():
    # Load the ONNX model
    onnx_path = "kokoro.onnx"
    session = load_onnx_model(onnx_path)

    # Initialize the pipeline
    pipeline = KPipeline(lang_code='a', model=False)

    # Example text and voice
    text = "Hello! This is a test of the Kokoro text-to-speech system."
    voice = "af_heart"

    # Load vocabulary from Hugging Face
    repo_id = "hexgrad/Kokoro-82M"  # Replace with the actual repository ID
    config_filename = "config.json"  # Replace with the actual config filename
    vocab = load_vocab_from_hf(repo_id, config_filename)

    # Tokenize and phonemize
    _, tokens = pipeline.g2p(text)
    
    # Use en_tokenize to process tokens
    for graphemes, phonemes, token_list in pipeline.en_tokenize(tokens):
        # Convert phonemes to input_ids
        input_ids = torch.LongTensor([[0, *map(lambda p: vocab.get(p), phonemes), 0]])
        input_lengths = torch.LongTensor([input_ids.shape[-1]])
        
        # Load and process the style vector
        ref_s = pipeline.load_voice(voice)
        ref_s = ref_s[len(input_ids) - 1]  # Select the appropriate style vector

        # Run inference
        start = time.time()
        audio, pred_dur = run_inference(session, input_ids, input_lengths, ref_s)
        end = time.time()
        logger.info(f"Inference time: {end - start} seconds")

        # Write audio to file
        audio = torch.tensor(audio).squeeze().numpy()
        sf.write('onnx_output.wav', audio, 24000)  # Assuming a sample rate of 24000 Hz

        logger.info("Inference complete. Audio written to 'onnx_output.wav'.")

if __name__ == "__main__":
    main()
