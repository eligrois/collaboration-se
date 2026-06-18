#!/usr/bin/env python3
"""
Test the dereverberation model on a single noisy audio file.

Takes one input audio file, runs the model, and saves both the input
and enhanced output to the results directory.

Supports stereo (2-channel) input — each channel is enhanced independently
and the result is stacked back to stereo.

Supports partial enhancement via --start_sec / --end_sec — only the specified
time range is enhanced while the rest of the signal remains unchanged.

Usage:
    python test_single.py --input_wav /path/to/noisy_audio.wav
    python test_single.py --input_wav /path/to/noisy.wav --end2end_ckpt /path/to/model.ckpt
    python test_single.py --input_wav /path/to/noisy.wav --start_sec 2.0 --end_sec 5.0
"""

import os
import sys
import argparse

import torch
import torchaudio
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from conf.config import Config
from src.model import DereverbModule
from src.audio_utils import preprocess_audio, mag_phase_to_ri, ri_to_wav


def parse_args():
    parser = argparse.ArgumentParser(description="Test dereverberation on a single file")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to noisy input audio")
    parser.add_argument("--end2end_ckpt", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--results_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--start_sec", type=float, default=None,
                        help="Start time (seconds) of region to enhance. If not set, enhance entire file.")
    parser.add_argument("--end_sec", type=float, default=None,
                        help="End time (seconds) of region to enhance. If not set, enhance entire file.")
    return parser.parse_args()


def load_model(cfg: Config, device: torch.device) -> DereverbModule:
    """Load the end-to-end model from checkpoint."""
    model = DereverbModule(
        window_size=cfg.window_size,
        overlap=cfg.overlap,
        norm_mag=cfg.norm_mag,
        norm_mag_target=cfg.norm_mag_target,
        cut_first_freq=cfg.cut_first_freq,
    )

    if cfg.end2end_ckpt and os.path.isfile(cfg.end2end_ckpt):
        ckpt = torch.load(cfg.end2end_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {cfg.end2end_ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {cfg.end2end_ckpt}")

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def enhance_mono(model: DereverbModule, wav_np: np.ndarray,
                 cfg: Config, device: torch.device) -> np.ndarray:
    """Enhance a single-channel (mono) numpy waveform.

    Args:
        wav_np: 1-D numpy array of audio samples.

    Returns:
        Enhanced 1-D numpy array.
    """
    from src.audio_utils import load_and_stft, stft_to_mag_phase, EPSILON

    # Compute STFT from numpy array (pass array directly)
    stft = load_and_stft(wav_np, cfg.sample_rate, cfg.window_size, cfg.overlap,
                         normalize_type="none", dont_use_end=0, test_mode=True)
    mag, phase = stft_to_mag_phase(stft)

    if cfg.normalize_type == "max_spec":
        mag = mag / (mag.max() + EPSILON)
    elif cfg.normalize_type == "z_score_spec":
        mag = mag / (mag.std() + EPSILON)

    if cfg.cut_first_freq:
        mag = mag[1:]
        phase = phase[1:]

    in_mag = mag.unsqueeze(0)  # (1, F, T)
    in_phase = phase.unsqueeze(0)

    in_mag_d = in_mag.unsqueeze(0).to(device)
    in_phase_d = in_phase.unsqueeze(0).to(device)
    pred_ri, _ = model((in_mag_d, in_phase_d))
    pred_ri = pred_ri.cpu()

    pred_wav = ri_to_wav(pred_ri, cfg.window_size, cfg.overlap, cfg.cut_first_freq)
    return pred_wav.squeeze().numpy()


@torch.no_grad()
def process_single_file(model: DereverbModule, input_wav_path: str,
                        cfg: Config, device: torch.device,
                        start_sec: float = None, end_sec: float = None):
    """Process a single audio file through the dereverberation model.

    Supports stereo (2-channel) input and partial enhancement.
    Returns audio at the original sample rate of the input file.

    Returns:
        input_wav: numpy array — shape (L,) for mono or (L, 2) for stereo
        pred_wav:  numpy array — same shape, enhanced
        orig_sr:   int — original sample rate of the input file
    """
    import soundfile

    # Load raw audio preserving channels
    signal_wav, orig_sr = soundfile.read(input_wav_path)

    # Keep a copy at original sample rate for output
    original_signal = signal_wav.copy()

    # Resample to model sample rate if needed
    needs_resample = orig_sr != cfg.sample_rate
    if needs_resample:
        import librosa
        if signal_wav.ndim == 1:
            signal_wav = librosa.resample(signal_wav, orig_sr=orig_sr, target_sr=cfg.sample_rate)
        else:
            channels = []
            for ch in range(signal_wav.shape[1]):
                channels.append(librosa.resample(signal_wav[:, ch], orig_sr=orig_sr, target_sr=cfg.sample_rate))
            signal_wav = np.stack(channels, axis=1)

    is_stereo = signal_wav.ndim == 2 and signal_wav.shape[1] == 2

    # Determine sample range to enhance (at model sample rate)
    total_samples = signal_wav.shape[0]
    if start_sec is not None or end_sec is not None:
        s_start = int((start_sec or 0.0) * cfg.sample_rate)
        s_end = int((end_sec or (total_samples / cfg.sample_rate)) * cfg.sample_rate)
        s_start = max(0, min(s_start, total_samples))
        s_end = max(s_start, min(s_end, total_samples))
    else:
        s_start = 0
        s_end = total_samples

    # Corresponding sample range at original sample rate
    orig_total = original_signal.shape[0]
    if start_sec is not None or end_sec is not None:
        o_start = int((start_sec or 0.0) * orig_sr)
        o_end = int((end_sec or (orig_total / orig_sr)) * orig_sr)
        o_start = max(0, min(o_start, orig_total))
        o_end = max(o_start, min(o_end, orig_total))
    else:
        o_start = 0
        o_end = orig_total

    # Build output as copy of original (unchanged regions stay intact)
    output_wav = original_signal.copy()

    if is_stereo:
        print("Stereo input detected — enhancing each channel independently.")
        for ch in range(2):
            segment = signal_wav[s_start:s_end, ch]
            enhanced_segment = enhance_mono(model, segment, cfg, device)
            # Resample enhanced segment back to original sample rate
            if needs_resample:
                import librosa
                enhanced_segment = librosa.resample(enhanced_segment, orig_sr=cfg.sample_rate, target_sr=orig_sr)
            # Match length to original segment
            orig_seg_len = o_end - o_start
            if len(enhanced_segment) >= orig_seg_len:
                enhanced_segment = enhanced_segment[:orig_seg_len]
            else:
                enhanced_segment = np.pad(enhanced_segment, (0, orig_seg_len - len(enhanced_segment)))
            output_wav[o_start:o_end, ch] = enhanced_segment
    else:
        if signal_wav.ndim > 1:
            signal_wav = signal_wav.squeeze()
        if original_signal.ndim > 1:
            original_signal = original_signal.squeeze()
            output_wav = output_wav.squeeze()
        segment = signal_wav[s_start:s_end]
        enhanced_segment = enhance_mono(model, segment, cfg, device)
        # Resample enhanced segment back to original sample rate
        if needs_resample:
            import librosa
            enhanced_segment = librosa.resample(enhanced_segment, orig_sr=cfg.sample_rate, target_sr=orig_sr)
        orig_seg_len = o_end - o_start
        if len(enhanced_segment) >= orig_seg_len:
            enhanced_segment = enhanced_segment[:orig_seg_len]
        else:
            enhanced_segment = np.pad(enhanced_segment, (0, orig_seg_len - len(enhanced_segment)))
        output_wav[o_start:o_end] = enhanced_segment

    return original_signal, output_wav, orig_sr


def main():
    args = parse_args()
    cfg = Config()

    if args.end2end_ckpt:
        cfg.end2end_ckpt = args.end2end_ckpt
    if args.results_dir:
        cfg.results_dir = args.results_dir

    if not os.path.isfile(args.input_wav):
        print(f"ERROR: Input file not found: {args.input_wav}")
        sys.exit(1)

    results_dir = os.path.join(cfg.results_dir, "single_test")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)

    print(f"Processing: {args.input_wav}")
    if args.start_sec is not None or args.end_sec is not None:
        print(f"Enhancing region: {args.start_sec or 0.0}s — {args.end_sec or 'end'}s")

    input_wav, pred_wav, orig_sr = process_single_file(model, args.input_wav, cfg, device,
                                                        start_sec=args.start_sec,
                                                        end_sec=args.end_sec)

    # Save results at the original sample rate
    basename = os.path.splitext(os.path.basename(args.input_wav))[0]
    is_stereo = input_wav.ndim == 2 and input_wav.shape[1] == 2

    for suffix, wav_np in [("noisy", input_wav), ("enhanced", pred_wav)]:
        if is_stereo:
            # (L, 2) -> (2, L) for torchaudio
            wav_t = torch.tensor(wav_np.T, dtype=torch.float32)
        else:
            wav_t = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
        peak = wav_t.abs().max() + 1e-8
        wav_t = wav_t / peak * 0.9
        out_path = os.path.join(results_dir, f"{basename}_{suffix}.wav")
        torchaudio.save(out_path, wav_t, orig_sr)
        print(f"Saved: {out_path} (sample rate: {orig_sr} Hz)")

    print("Done!")


if __name__ == "__main__":
    main()
