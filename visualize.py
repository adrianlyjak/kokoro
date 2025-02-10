# /// script
# dependencies = [
#   "torch==2.6.0",
#   "numpy==1.26.4",
#   "matplotlib",
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.utils.rnn as rnn

def visualize_packing_vs_masking():
    # Create sample data
    batch_size = 3
    max_len = 6
    hidden_dim = 4
    
    # Sample sequences of different lengths
    sequences = [
        [1, 1, 1, 0, 0, 0],  # length 3
        [1, 1, 1, 1, 0, 0],  # length 4
        [1, 1, 0, 0, 0, 0],  # length 2
    ]
    lengths = [3, 4, 2]
    
    # Convert to tensor
    data = torch.FloatTensor(sequences)
    lengths_tensor = torch.LongTensor(lengths)
    
    # Create mask
    mask = torch.arange(max_len)[None, :] < lengths_tensor[:, None]
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Original padded sequences
    ax1.imshow(sequences, cmap='Blues')
    ax1.set_title('Original Padded Sequences')
    ax1.set_ylabel('Batch')
    for i in range(batch_size):
        ax1.text(-0.5, i, f'Seq {i}')
    
    # 2. Packed sequence visualization
    packed = rnn.pack_padded_sequence(data, lengths_tensor, batch_first=True, enforce_sorted=False)
    packed_data = packed.data.numpy()
    
    # Create packed visualization
    packed_viz = np.zeros((batch_size, max_len))
    current_pos = 0
    for t in range(max(lengths)):
        batch_t = sum(1 for l in lengths if l > t)
        packed_viz[:batch_t, t] = 2  # Different color for packed
    
    ax2.imshow(packed_viz, cmap='Reds')
    ax2.set_title('Packed Sequence (only real data processed)')
    ax2.set_ylabel('Batch')
    
    # 3. Masked sequence visualization
    masked_data = (data * mask).numpy()
    ax3.imshow(masked_data, cmap='Greens')
    ax3.set_title('Masked Sequence (zeros still processed)')
    ax3.set_ylabel('Batch')
    
    plt.tight_layout()
    plt.show()

visualize_packing_vs_masking()