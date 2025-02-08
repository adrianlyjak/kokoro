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
# ]
# ///

import onnxruntime as ort
import numpy as np
import torch
from kokoro.model import KModel
from kokoro.pipeline import KPipeline
from loguru import logger
import soundfile as sf

def load_onnx_model(onnx_path: str):
    """Load the ONNX model."""
    return ort.InferenceSession(onnx_path)

def run_inference(session, input_ids, input_lengths, ref_s, speed=1):
    """Run inference using the ONNX model."""
    inputs = {
        'input_ids': input_ids.numpy(),
        'input_lengths': input_lengths.numpy(),
        'ref_s': ref_s.numpy(),
        'speed': np.array([speed], dtype=np.float32)
    }
    outputs = session.run(None, inputs)
    return outputs

def main():
    # Load the ONNX model
    onnx_path = "kokoro.onnx"
    session = load_onnx_model(onnx_path)

    # Initialize the pipeline
    pipeline = KPipeline(lang_code='a', model=False)

    # Example text and voice
    text = "Hello world!"
    voice = "af_heart"

    # Tokenize and phonemize
    _, tokens = pipeline.g2p(text)
    input_ids = torch.LongTensor([[0, *map(lambda p: pipeline.model.vocab.get(p), tokens), 0]])
    input_lengths = torch.LongTensor([input_ids.shape[-1]])
    ref_s = pipeline.load_voice(voice)

    # Run inference
    audio, pred_dur = run_inference(session, input_ids, input_lengths, ref_s)

    # Write audio to file
    audio = torch.tensor(audio).squeeze().numpy()
    sf.write('output.wav', audio, 24000)  # Assuming a sample rate of 24000 Hz

    logger.info("Inference complete. Audio written to 'output.wav'.")

if __name__ == "__main__":
    main()
