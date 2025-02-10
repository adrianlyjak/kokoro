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
# ]
# ///
import time
import torch
from kokoro.model import KModelForONNX, KModel
import onnx
from loguru import logger
import onnxruntime as ort
import numpy as np

model = KModelForONNX(KModel(disable_complex=True)).eval()

# Constants from the model
batch_size = 1
context_length = 512  # Based on position embeddings size
style_dim = 256  # Total style dimension (128 + 128) based on ref_s splitting

# Create dummy inputs
dummy_seq_length = 12
input_ids = torch.zeros((batch_size, dummy_seq_length), dtype=torch.long)
input_ids[0, :] = torch.LongTensor([0] + [1] * (dummy_seq_length - 2) + [0])

# Input lengths tensor
input_lengths = torch.LongTensor([dummy_seq_length])

# Style reference tensor
ref_s = torch.randn(batch_size, style_dim)

# validate the inputs agains the model first
start_time = time.time()

output, pred_dur = model(input_ids=input_ids, input_lengths=input_lengths, ref_s=ref_s, speed=1.0)
logger.info(f"Time for dummy inputs: {time.time() - start_time}")

# Define dynamic axes
dynamic_axes = {
    "input_ids": {0: "batch", 1: "sequence"},
    "input_lengths": {0: "batch"},
    "ref_s": {0: "batch"},
    "audio": {0: "batch", 1: "sequence"},
    "pred_dur": {0: "batch", 1: "sequence"},
}

logger.info("Starting ONNX export...")
try:
    torch.onnx.export(
        model,
        (input_ids, input_lengths, ref_s, 1.0),  # Inputs as positional arguments
        "kokoro.onnx",
        input_names=["input_ids", "input_lengths", "ref_s", "speed"],
        output_names=["audio", "pred_dur"],
        dynamic_axes=dynamic_axes,
        opset_version=20,
        export_params=True,
        do_constant_folding=True,
    )

    # Verify the model
    onnx_model = onnx.load("kokoro.onnx")
    onnx.checker.check_model(onnx_model)
    logger.info("Model was successfully exported to ONNX")

    # Additional check: Run a simple inference to validate the exported model
    ort_session = ort.InferenceSession("kokoro.onnx")
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "input_lengths": input_lengths.numpy(),
        "ref_s": ref_s.numpy(),
        "speed": np.array([1.0], dtype=np.float64)
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    # Compare audio output
    torch_audio = output
    onnx_audio = torch.tensor(ort_outputs[0])
    audio_mse = (torch_audio - onnx_audio).pow(2).mean().item()
    logger.info(f"MSE for audio waveform: {audio_mse:.5f}")

except Exception as e:
    logger.error(f"Export failed: {str(e)}")
    raise
