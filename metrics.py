"""
Audio quality metrics for dereverberation evaluation.
"""

import torch
import numpy as np

EPSILON = 1e-8


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    Args:
        estimate:  (B, L) or (1, L)
        reference: (B, L) or (1, L)

    Returns:
        SI-SDR in dB, scalar tensor.
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    ref_pow = reference.pow(2).mean(dim=-1, keepdim=True)
    mix_pow = (estimate * reference).mean(dim=-1, keepdim=True)
    scale = mix_pow / (ref_pow + EPSILON)

    ref_scaled = scale * reference
    error = estimate - ref_scaled

    ref_scaled_pow = ref_scaled.pow(2).mean(dim=-1)
    error_pow = error.pow(2).mean(dim=-1)

    si_sdr_val = (10 * torch.log10(ref_scaled_pow + 1e-8)
                  - 10 * torch.log10(error_pow + 1e-8))
    return si_sdr_val.mean()


def compute_pesq(estimate: np.ndarray, reference: np.ndarray, sample_rate: int = 16000):
    """Compute PESQ score. Requires the `pesq` package."""
    try:
        from pesq import pesq as pesq_fn
        mode = "wb" if sample_rate >= 16000 else "nb"
        return pesq_fn(sample_rate, reference, estimate, mode)
    except ImportError:
        return float("nan")


def compute_stoi(estimate: np.ndarray, reference: np.ndarray, sample_rate: int = 16000):
    """Compute STOI score. Requires the `pystoi` package."""
    try:
        from pystoi import stoi
        return stoi(reference, estimate, sample_rate, extended=False)
    except ImportError:
        return float("nan")


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi-resolution STFT loss combining spectral convergence and log-magnitude loss.

    Computes STFT at multiple (fft_size, hop_size, win_size) resolutions and
    averages the spectral-convergence + log-magnitude L1 losses across them.
    """

    def __init__(self, fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_sizes=(512, 1024, 2048)):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_mag(self, x, fft_size, hop_size, win_size):
        window = torch.hann_window(win_size, device=x.device)
        stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_size, window=window,
                          return_complex=True, center=True)
        return stft.abs()

    def forward(self, estimate, reference):
        """
        Args:
            estimate:  (B, L) waveform
            reference: (B, L) waveform
        Returns:
            scalar loss
        """
        total_loss = 0.0
        for fft_sz, hop_sz, win_sz in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            est_mag = self._stft_mag(estimate, fft_sz, hop_sz, win_sz)
            ref_mag = self._stft_mag(reference, fft_sz, hop_sz, win_sz)

            # Spectral convergence loss
            sc_loss = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + EPSILON)

            # Log-magnitude L1 loss
            log_loss = torch.nn.functional.l1_loss(
                torch.log(est_mag + EPSILON), torch.log(ref_mag + EPSILON)
            )

            total_loss = total_loss + sc_loss + log_loss

        return total_loss / len(self.fft_sizes)
