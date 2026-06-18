"""
Audio utility functions for STFT processing and preprocessing.
"""

import random
import torch
import soundfile
import numpy as np

EPSILON = 1e-6


def load_and_stft(path_wav, sample_rate: int, window_size: int, overlap: int,
                  normalize_type: str = "max_spec", dont_use_end: int = 0,
                  test_mode: bool = False):
    """Load an audio file and compute its STFT.

    Returns:
        signal_stft: torch.Tensor of shape (F, T, 2) — real and imaginary parts.
    """
    if isinstance(path_wav, str):
        signal_wav, sr = soundfile.read(path_wav)
    else:
        signal_wav = np.array(path_wav)
        sr = sample_rate

    if len(signal_wav.shape) > 1:
        signal_wav = signal_wav.squeeze()

    if sr != sample_rate:
        import librosa
        signal_wav = librosa.resample(signal_wav, orig_sr=sr, target_sr=sample_rate)

    if dont_use_end and not test_mode and (len(signal_wav) - sample_rate) > 0:
        signal_wav = signal_wav[:len(signal_wav) - sample_rate]

    if normalize_type == "z_score":
        signal_wav = signal_wav / (signal_wav.std() + EPSILON)
    elif normalize_type == "min_max":
        signal_wav = signal_wav / (np.abs(signal_wav).max() + EPSILON)

    signal_stft = torch.stft(
        torch.tensor(signal_wav, dtype=torch.float32),
        window=torch.hamming_window(window_size),
        hop_length=overlap,
        n_fft=window_size,
        return_complex=False,
        center=False,
    ).squeeze(0)
    return signal_stft.float()


def stft_to_mag_phase(signal_stft: torch.Tensor):
    """Split STFT tensor (F, T, 2) into magnitude and phase."""
    mag = torch.sqrt(signal_stft.pow(2).sum(-1) + EPSILON)
    cmplx = signal_stft[..., 0] + 1j * signal_stft[..., 1]
    phase = torch.angle(cmplx)
    return mag.float(), phase.float()


def mag_phase_to_ri(mag: torch.Tensor, phase: torch.Tensor,
                    from_log: bool = False) -> torch.Tensor:
    """Convert magnitude + phase to real-imaginary STFT tensor.

    Args:
        mag:   (B, 1, F, T) or (1, F, T)
        phase: same shape as mag
        from_log: if True, mag = 10^mag before conversion

    Returns:
        Tensor of shape (B, 2, F, T) — channel 0=real, channel 1=imag
    """
    if from_log:
        mag = 10 ** mag
    real = torch.cos(phase) * mag
    imag = torch.sin(phase) * mag
    return torch.cat((real, imag), dim=-3)


def ri_to_wav(ri_stft: torch.Tensor, window_size: int, overlap: int,
              cut_first_freq: bool = True) -> torch.Tensor:
    """Convert real-imaginary STFT back to waveform via ISTFT.

    Args:
        ri_stft: (B, 2, F, T)

    Returns:
        waveform: (B, L)
    """
    cmplx = ri_stft[:, 0, :, :] + 1j * ri_stft[:, 1, :, :]
    if cut_first_freq:
        cmplx = torch.cat((cmplx[:, 0:1, :] * 0, cmplx), dim=1)
    wav = torch.istft(
        cmplx,
        window=torch.hamming_window(window_size).to(cmplx.device),
        hop_length=overlap,
        n_fft=window_size,
        onesided=True,
        center=False,
    )
    return wav


def preprocess_audio(path_wav: str, sample_rate: int, window_size: int, overlap: int,
                     normalize_type: str = "max_spec", cut_first_freq: bool = True,
                     dont_use_end: int = 0, test_mode: bool = False,
                     frames: int = None, train_mode: bool = True):
    """Full preprocessing pipeline for a single audio file.

    Returns:
        mag: (1, F, T) magnitude
        phase: (1, F, T) phase
    """
    stft = load_and_stft(path_wav, sample_rate, window_size, overlap,
                         normalize_type="none",
                         dont_use_end=dont_use_end, test_mode=test_mode)
    mag, phase = stft_to_mag_phase(stft)

    if normalize_type == "max_spec":
        mag = mag / (mag.max() + EPSILON)
    elif normalize_type == "z_score_spec":
        mag = mag / (mag.std() + EPSILON)

    if cut_first_freq:
        mag = mag[1:]
        phase = phase[1:]

    if train_mode and frames is not None:
        while mag.shape[-1] < frames:
            mag = torch.cat((mag, mag), dim=-1)
            phase = torch.cat((phase, phase), dim=-1)

        start = random.randint(0, mag.shape[-1] - frames)
        mag = mag[..., start:start + frames]
        phase = phase[..., start:start + frames]

    return mag.unsqueeze(0), phase.unsqueeze(0)


def preprocess_pair(input_path: str, target_path: str,
                    sample_rate: int, window_size: int, overlap: int,
                    normalize_type: str = "max_spec", cut_first_freq: bool = True,
                    dont_use_end: int = 0, frames: int = 256,
                    train_mode: bool = True):
    """Preprocess a (noisy, clean) audio pair.

    Returns:
        input_mag, input_phase, target_mag, target_phase — each (1, F, T)
    """
    in_stft = load_and_stft(input_path, sample_rate, window_size, overlap,
                            normalize_type="none", dont_use_end=dont_use_end)
    tgt_stft = load_and_stft(target_path, sample_rate, window_size, overlap,
                             normalize_type="none", dont_use_end=dont_use_end)

    in_mag, in_phase = stft_to_mag_phase(in_stft)
    tgt_mag, tgt_phase = stft_to_mag_phase(tgt_stft)

    if normalize_type == "max_spec":
        in_mag = in_mag / (in_mag.max() + EPSILON)

    # Align lengths
    min_t = min(in_mag.shape[-1], tgt_mag.shape[-1])
    in_mag = in_mag[..., :min_t]
    in_phase = in_phase[..., :min_t]
    tgt_mag = tgt_mag[..., :min_t]
    tgt_phase = tgt_phase[..., :min_t]

    if cut_first_freq:
        in_mag, in_phase = in_mag[1:], in_phase[1:]
        tgt_mag, tgt_phase = tgt_mag[1:], tgt_phase[1:]

    if train_mode and frames is not None:
        while in_mag.shape[-1] < frames:
            in_mag = torch.cat((in_mag, in_mag), dim=-1)
            in_phase = torch.cat((in_phase, in_phase), dim=-1)
            tgt_mag = torch.cat((tgt_mag, tgt_mag), dim=-1)
            tgt_phase = torch.cat((tgt_phase, tgt_phase), dim=-1)

        start = random.randint(0, in_mag.shape[-1] - frames)
        in_mag = in_mag[..., start:start + frames]
        in_phase = in_phase[..., start:start + frames]
        tgt_mag = tgt_mag[..., start:start + frames]
        tgt_phase = tgt_phase[..., start:start + frames]

    return (in_mag.unsqueeze(0), in_phase.unsqueeze(0),
            tgt_mag.unsqueeze(0), tgt_phase.unsqueeze(0))
