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
#   "matplotlib",
# ]
# ///
import torch
import torch.nn as nn
from kokoro.custom_stft import CustomSTFT
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt

class STFTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, imag_out, real_out):
        phase = torch.atan2(imag_out, real_out)
        # Handle the case where imag_out is 0 and real_out is negative to correct ONNX atan2 to match PyTorch
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return phase

# Create wrapper module
stft_wrapper = STFTWrapper()
stft_wrapper.eval()

# Create a random test waveform
batch_size = 1
wave_len = 16000
real_out = torch.randn(1, 11, 18721, dtype=torch.float32)
imag_out = torch.randn(1, 11, 18721, dtype=torch.float32)



# Export to ONNX
onnx_name = "test_customstft.onnx"
torch.onnx.export(
    stft_wrapper,
    (imag_out, real_out),
    onnx_name,
    input_names=["imag_out", "real_out"],
    output_names=["result"],
    do_constant_folding=True,
    opset_version=20,  # or higher if needed
)


real_data = np.load("debug_stft.npz")

# Run in PyTorch
with torch.no_grad():
    torch_result = stft_wrapper(torch.from_numpy(real_data["imag_out"]), torch.from_numpy(real_data["real_out"]))

# Run in ONNXRuntime
ort_session = onnxruntime.InferenceSession(onnx_name)
ort_inputs = {"imag_out": real_data["imag_out"], "real_out": real_data["real_out"]}
ort_outs = ort_session.run(None, ort_inputs)
onnx_result = torch.from_numpy(ort_outs[0])

# Compare
result_mse = (torch_result - onnx_result).pow(2).mean().item()

print(f"Result MSE between PyTorch vs. ONNX: {result_mse:.6f}")
print("torch_result", torch_result[:, :2, :5])
print("onnx_result", onnx_result[:, :2, :5])

# Define a threshold for significant difference
threshold = 0.1

# Compare and find significant differences
significant_differences = []
all_differences = []
for idx in np.ndindex(torch_result.shape):
    diff = torch_result[idx].item() - onnx_result[idx].item()
    all_differences.append(diff)
    if abs(diff) > threshold:
        significant_differences.append((idx, torch_result[idx].item(), onnx_result[idx].item(), diff))

# Calculate the volume of significant differences
num_significant_differences = len(significant_differences)
total_elements = torch_result.numel()
percentage_significant = (num_significant_differences / total_elements) * 100

# Print results
print(f"Number of significant differences: {num_significant_differences}")
print(f"Percentage of significant differences: {percentage_significant:.2f}%")

# Print some of the significant differences with input indices
print("Some significant differences (index, torch_value, onnx_value, difference, imag_out, real_out):")
for i, (idx, torch_val, onnx_val, diff) in enumerate(significant_differences[:16000]):
    imag_val = real_data["imag_out"][idx]
    real_val = real_data["real_out"][idx]
    print(f"Index: {idx}, Torch: {torch_val:.6f}, ONNX: {onnx_val:.6f}, Diff: {diff:.6f}, Imag: {imag_val:.6f}, Real: {real_val:.6f}")

# # Plot histogram of significant differences
# plt.hist([x[3] for x in significant_differences], bins=50, color='blue', alpha=0.7)
# plt.title('Histogram of Significant Differences between PyTorch and ONNX Results')
# plt.xlabel('Difference')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

