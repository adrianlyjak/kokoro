import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window


def build_dft_matrix(n_fft: int, onesided: bool = True):
    """
    Build forward DFT kernels (real, imag) to mimic PyTorch stft(..., onesided=True).

    PyTorch stft uses e^{-j 2 pi k n / N}.
    => real = cos( 2 pi k n / N )
       imag = - sin(2 pi k n / N )   (note negative sign!)
    """
    freq_bins = n_fft // 2 + 1 if onesided else n_fft
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(1)  # shape (n_fft, 1)
    k = torch.arange(freq_bins, dtype=torch.float32).unsqueeze(
        0
    )  # shape (1, freq_bins)
    angle = 2 * np.pi * n * k / n_fft

    dft_real = torch.cos(angle)
    dft_imag = -torch.sin(angle)  # negative sign to match e^{-j 2 pi ...}
    return dft_real, dft_imag


def build_idft_matrix(n_fft: int, onesided: bool = True):
    """
    Build real-valued iFFT kernels for onesided STFT data. This matches torch.istft.

    If onesided, bins [1..n_fft//2 - 1] get doubled in the real iFFT sum.
    DC and Nyquist (if even n_fft) do NOT get doubled. Then multiply by 1/n_fft overall.

    We'll produce:
      idft_cos, idft_sin of shape (freq_bins, n_fft)
    so that time_frame = sum_{k=0 to freq_bins-1}[ real[k] * idft_cos[k,:] - imag[k] * idft_sin[k,:] ]
    """
    freq_bins = n_fft // 2 + 1 if onesided else n_fft
    k = torch.arange(freq_bins, dtype=torch.float32).unsqueeze(1)  # (freq_bins, 1)
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)  # (1, n_fft)
    angle = 2 * np.pi * k * n / n_fft  # iFFT uses +2*pi kn/N

    # scale: DC and possibly Nyquist do not get doubled, others do
    scale = torch.ones(freq_bins, dtype=torch.float32)
    if onesided:
        # If n_fft is even, bin = n_fft//2 is the Nyquist bin => do not double
        if n_fft % 2 == 0:
            # scale = [1, 2, 2, ..., 2, 1] length n_fft//2+1
            scale[1:-1] = 2.0
        else:
            # If n_fft is odd, there's no "Nyquist" bin
            scale[1:] = 2.0
    scale = scale / n_fft  # always 1 / n_fft

    scale = scale.unsqueeze(-1)  # shape (freq_bins,1)
    idft_cos = scale * torch.cos(angle)  # (freq_bins, n_fft)
    idft_sin = scale * torch.sin(angle)
    return idft_cos, idft_sin


class CustomSTFT(nn.Module):
    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.onesided = True
        self.center = True

        # Use the same style of Hann window that TorchSTFT uses:
        #    get_window("hann", ..., fftbins=True)
        win_np = get_window(window, self.win_length, fftbins=True).astype(np.float32)
        window_tensor = torch.from_numpy(win_np)

        # If win_length < n_fft, pad up to n_fft, if > n_fft, truncate down
        if self.win_length < self.n_fft:
            pad_amount = self.n_fft - self.win_length
            window_tensor = F.pad(window_tensor, (0, pad_amount))
        else:
            window_tensor = window_tensor[: self.n_fft]

        self.register_buffer("window", window_tensor)

        # Precompute forward DFT for STFT
        dft_real, dft_imag = build_dft_matrix(self.n_fft, onesided=self.onesided)
        self.register_buffer("dft_real", dft_real)  # (n_fft, freq_bins)
        self.register_buffer("dft_imag", dft_imag)

        # Precompute iDFT for the inverse
        idft_cos, idft_sin = build_idft_matrix(self.n_fft, onesided=self.onesided)
        self.register_buffer("idft_cos", idft_cos)  # (freq_bins, n_fft)
        self.register_buffer("idft_sin", idft_sin)

    def transform(self, x):
        """
        Mimic torch.stft(..., center=True, onesided=True, normalized=False, return_complex=False).
        Returns (magnitude, phase) shaped (B, freq_bins, frames).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, orig_len = x.shape

        # center=True => reflect-pad by n_fft//2
        pad_len = self.n_fft // 2
        if self.center:
            x = F.pad(x, (pad_len, pad_len), mode="reflect")

        # Frame the signal:  (B, frames, n_fft)
        frames = x.unfold(-1, self.n_fft, self.hop_length)
        # Multiply by window
        frames = frames * self.window.view(1, 1, -1)

        # forward real + imag (B, frames, freq_bins)
        real_part = torch.matmul(frames, self.dft_real)
        imag_part = torch.matmul(frames, self.dft_imag)

        # transpose to (B, freq_bins, frames) to match torch.stft's output shape
        real_part = real_part.transpose(1, 2)
        imag_part = imag_part.transpose(1, 2)

        # magnitude, phase
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-14)
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase

    def inverse(self, magnitude, phase, length=None):
        """
        Mimic torch.istft(..., center=True, onesided=True, normalized=False).
        Reconstruction shape => (B, T).  Then we unsqueeze -> (B, 1, T).

        If length is not None, we will truncate or pad to exactly that length at the end.
        By default, we’ll do the same as PyTorch: remove the center padding of n_fft//2 from each side,
        and then clamp if it's still bigger than the original length.
        """
        # magnitude, phase => (B, freq_bins, frames)
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        # go to (B, frames, freq_bins)
        real_part = real_part.transpose(1, 2)
        imag_part = imag_part.transpose(1, 2)

        # Inverse real-valued iFFT
        # frames_time[b, t, n] = sum_{k=0..freq_bins-1}[ real_part[b,t,k]*idft_cos[k,n]
        #                                           - imag_part[b,t,k]*idft_sin[k,n] ]
        frames_time = torch.matmul(real_part, self.idft_cos) - torch.matmul(
            imag_part, self.idft_sin
        )

        # Multiply by same window, overlap-add
        frames_time = frames_time * self.window.view(1, 1, -1)
        num_frames = frames_time.shape[1]
        expected_len = (num_frames - 1) * self.hop_length + self.n_fft

        output = torch.zeros(
            frames_time.shape[0],
            expected_len,
            dtype=frames_time.dtype,
            device=frames_time.device,
        )
        window_sq = self.window * self.window
        norm = torch.zeros_like(output)

        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_length
            end = start + self.n_fft
            output[:, start:end] += frames_time[:, frame_idx, :]
            norm[:, start:end] += window_sq

        # Divide by window overlap sum
        output = output / (norm + 1e-14)

        # If center=True, remove n_fft//2 from each side
        if self.center:
            pad_len = self.n_fft // 2
            if pad_len > 0:
                output = output[..., pad_len:-pad_len]

        # If length is known (e.g. we want exactly the original # of samples),
        # clamp or pad to that length.  This matches torch.istft's behavior if
        # you pass `length=original_length`.
        if length is not None:
            if output.shape[-1] > length:
                output = output[..., :length]
            elif output.shape[-1] < length:
                pad_amount = length - output.shape[-1]
                output = F.pad(output, (0, pad_amount))

        return output.unsqueeze(1)

    def forward(self, x):
        """
        Just do transform -> inverse, returning (B, 1, T).
        By default we clamp the final length to match x.shape[-1], so we get
        exactly the same number of samples as the input (like torch.istft).
        """
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])
