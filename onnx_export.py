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
#   "torch",
#   "transformers",
# ]
# ///
from kokoro import KModel
import time
import torch
import onnx
from torch.export import Dim

model = KModel(None, None).eval()

print(model)

# Constants from the model
batch_size = 1
context_length = 512  # Based on position embeddings size
style_dim = 256  # Total style dimension (128 + 128) based on ref_s splitting
embedding_dim = 128  # From the model architecture

# Create dummy inputs with proper padding to context_length
dummy_seq_length = 12
input_ids = torch.zeros(
    (batch_size, context_length), dtype=torch.long
)  # Initialize with padding
input_ids[0, :dummy_seq_length] = torch.LongTensor(
    [0] + [1] * (dummy_seq_length - 2) + [0]
)  # Add content

# Style reference tensor
ref_s = torch.randn(batch_size, style_dim)  # [batch_size, 256]

# Test the inputs
start_time = time.time()
print("trying dummy inputs")
print(f"input_ids shape: {input_ids.shape}")
print(f"ref_s shape: {ref_s.shape}")
# Use named arguments to match the model's expectations
output = model(phonemes=input_ids, ref_s=ref_s)
print(f"time for dummy inputs: {time.time() - start_time}")

# Define dynamic dimensions
batch = Dim("batch", min=1, max=32)
seq = Dim("sequence", min=1, max=context_length)

# Define dynamic shapes for inputs only
dynamic_shapes = {"phonemes": {0: batch, 1: seq}, "ref_s": {0: batch}}

print("Starting ONNX export...")
try:
    torch.onnx.export(
        model,
        ({"phonemes": input_ids, "ref_s": ref_s},),  # Keep dict format for named params
        "kokoro.onnx",
        input_names=["phonemes", "ref_s"],
        output_names=["output"],
        dynamic_shapes=dynamic_shapes,
        opset_version=20,
        dynamo=True,
        report=True,
        export_params=True,
        do_constant_folding=True,
        verbose=True,
    )

    # Verify the model
    onnx_model = onnx.load("kokoro.onnx")
    onnx.checker.check_model(onnx_model)
    print("Model was successfully exported to ONNX")

except Exception as e:
    print(f"Export failed: {str(e)}")
    raise
