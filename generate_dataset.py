#!/usr/bin/env python3
"""
Generate a synthetic dereverberation training dataset.

Takes clean speech files and convolves them with Room Impulse Responses (RIRs)
at various T60/SNR combinations, producing (mix.wav, clean.wav) pairs for training.

Usage:
    python generate_dataset.py --speech_dir /path/to/clean_speech --noise_dir /path/to/noise
    python generate_dataset.py --speech_dir /path/to/speech --noise_dir /path/to/noise \
        --output_dir /path/to/output --num_samples 60000 --sample_rate 16000

Output structure:
    <output_dir>/
    ├── train/aud_files/<sample>/  {mix.wav, clean.wav}
    ├── val/aud_files/<sample>/    {mix.wav, clean.wav}
    └── dataset_summary.json
"""

import os
import re
import sys
import json
import time
import random
import argparse
import multiprocessing
from functools import partial

import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.io.wavfile import write as wav_write
from scipy.spatial import distance

EPSILON = 1e-6
NOISE_DATASETS = ["/mount/data/ajal/solan/data/whamr/",
                  "/mount/data/ajal/solan/data/dns_noise/",
]

# ── Configuration ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dereverberation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required paths
    parser.add_argument("--speech_dir", type=str, required=True,
                        help="Root directory of clean speech (e.g. LibriSpeech/train-clean-360/)")
    parser.add_argument("--noise_dirs", type=str, nargs="+",
                        default=NOISE_DATASETS,
                        help="Directories with noise files (WHAMR + DNS by default). "
                             "Pass 'none' to fall back to Gaussian noise.")
    parser.add_argument("--output_dir", type=str, default="./generated_dataset",
                        help="Output directory for the generated dataset")

    # Dataset size
    parser.add_argument("--num_samples", type=int, default=60000,
                        help="Total number of samples to generate")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Fraction used for training (rest = validation)")

    # Audio parameters
    parser.add_argument("--sample_rate", type=int, default=16000)

    # Room acoustics
    parser.add_argument("--snr_min", type=int, default=-10, help="Minimum SNR in dB")
    parser.add_argument("--snr_max", type=int, default=20, help="Maximum SNR in dB")
    parser.add_argument("--room_min", type=float, default=3.0, help="Min room dim (meters)")
    parser.add_argument("--room_max", type=float, default=10.0, help="Max room dim (meters)")
    parser.add_argument("--beta_min", type=float, default=0.2, help="Min T60 (seconds)")
    parser.add_argument("--beta_max", type=float, default=0.8, help="Max T60 (seconds)")
    parser.add_argument("--dist_mic_min", type=float, default=1.0,
                        help="Min speaker-mic distance (meters)")
    parser.add_argument("--dist_mic_max", type=float, default=6.0,
                        help="Max speaker-mic distance (meters)")
    parser.add_argument("--dist_from_wall", type=float, default=0.5,
                        help="Min distance from wall (meters)")

    # Performance
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--speech_format", type=str, default="wav",
                        choices=["flac", "wav", "mp3"],
                        help="Speech file extension to look for")

    return parser.parse_args()


# ── Audio utilities ──────────────────────────────────────────────────────────

def load_audio(path, target_sr):
    """Load audio file, convert to mono, resample, and normalize."""
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = audio / (audio.std() + EPSILON)
    # Add very light Gaussian noise for numerical stability
    audio = audio + np.random.randn(len(audio)) * 0.001
    return audio


def apply_rir(clean_signal, rir_filter):
    """Convolve clean signal with RIR, compensating for RIR delay."""
    delay = np.argmax(np.abs(rir_filter))
    reverbed = signal.fftconvolve(clean_signal, rir_filter)
    reverbed = reverbed[delay:len(clean_signal) + delay]
    return reverbed


def add_noise(reverbed_signal, noise_signal, snr_db):
    """Add noise at a specified SNR level."""
    noise = np.resize(noise_signal, reverbed_signal.shape)
    noise = noise / (noise.std() + EPSILON)
    signal_amplitude = reverbed_signal.std()
    noise_amplitude = signal_amplitude / (10 ** (snr_db / 20))
    return noise * noise_amplitude


# ── Room simulation ─────────────────────────────────────────────────────────

def random_room(room_range, dist_mic_range, dist_from_wall, beta_range):
    """Generate random room dimensions, mic/speaker positions, and T60."""
    max_retries = 50
    for _ in range(max_retries):
        L = np.array([
            random.uniform(room_range[0], room_range[1]),
            random.uniform(room_range[0], room_range[1]),
            3.0,
        ])
        mic = np.array([
            random.uniform(dist_from_wall, L[0] - dist_from_wall),
            random.uniform(dist_from_wall, L[1] - dist_from_wall),
            random.uniform(dist_from_wall, L[2] - dist_from_wall),
        ])

        for _ in range(20):
            speaker = np.array([
                random.uniform(dist_from_wall, L[0] - dist_from_wall),
                random.uniform(dist_from_wall, L[1] - dist_from_wall),
                random.uniform(1.0, 2.0),
            ])
            dist = distance.euclidean(speaker, mic)
            if dist_mic_range[0] <= dist <= dist_mic_range[1]:
                beta = random.uniform(beta_range[0], beta_range[1])
                return L, mic, speaker, beta, dist

    # Fallback with mid-range values
    L = np.array([6.0, 6.0, 3.0])
    mic = np.array([3.0, 3.0, 1.5])
    speaker = np.array([4.0, 4.0, 1.5])
    beta = random.uniform(beta_range[0], beta_range[1])
    dist = distance.euclidean(speaker, mic)
    return L, mic, speaker, beta, dist


# ── Speech & noise file collection ───────────────────────────────────────────

def collect_speech_files(speech_dir, speech_format="flac"):
    """Walk the speech directory and collect all audio file paths."""
    files = []
    for root, _, filenames in os.walk(speech_dir):
        for fname in filenames:
            if fname.endswith(f".{speech_format}"):
                files.append(os.path.join(root, fname))
    files.sort()
    if not files:
        raise FileNotFoundError(
            f"No .{speech_format} files found under: {speech_dir}"
        )
    print(f"Found {len(files)} speech files in: {speech_dir}")
    return files


def collect_noise_files(noise_dirs):
    """Collect noise file paths from multiple directories (like WHAMR + DNS)."""
    files = []
    for noise_dir in noise_dirs:
        if not os.path.isdir(noise_dir):
            print(f"WARNING: noise directory not found, skipping: {noise_dir}")
            continue
        dir_files = []
        for f in os.listdir(noise_dir):
            if f.endswith((".wav", ".flac", ".mp3")):
                dir_files.append(os.path.join(noise_dir, f))
        dir_files.sort()
        print(f"Found {len(dir_files)} noise files in: {noise_dir}")
        files.extend(dir_files)
    if files:
        print(f"Total noise files: {len(files)}")
    return files


# ── Sample generation ────────────────────────────────────────────────────────

def generate_single_sample(sample_idx, speech_files, noise_files, args, save_dir):
    """Generate a single (mix, clean) training sample."""
    # Use per-process random seeding for true randomness in multiprocessing
    np.random.seed(None)
    random.seed()

    try:
        import rir_generator as rir
    except ImportError:
        print("ERROR: rir_generator not installed. Install with: pip install rir-generator")
        return None

    # Pick random speech file
    speech_path = random.choice(speech_files)
    clean_signal = load_audio(speech_path, args.sample_rate)

    # Generate random room + RIR
    L, mic, speaker, beta, mic_dist = random_room(
        room_range=(args.room_min, args.room_max),
        dist_mic_range=(args.dist_mic_min, args.dist_mic_max),
        dist_from_wall=args.dist_from_wall,
        beta_range=(args.beta_min, args.beta_max),
    )
    h = np.squeeze(rir.generate(
        c=340, fs=args.sample_rate, r=mic, s=speaker, L=L,
        reverberation_time=beta,
    ))

    # Apply RIR
    reverbed = apply_rir(clean_signal, h)

    # Add noise
    snr_db = random.randint(args.snr_min, args.snr_max)
    if noise_files:
        noise_path = random.choice(noise_files)
        noise_signal = load_audio(noise_path, args.sample_rate)
    else:
        noise_signal = np.random.randn(len(reverbed))

    noise_scaled = add_noise(reverbed, noise_signal, snr_db)
    mix = reverbed + noise_scaled

    # Normalize and save
    sample_name = f"file_{sample_idx}_snr_{snr_db}_T60_{int(beta * 1000)}msec"
    sample_dir = os.path.join(save_dir, "aud_files", sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    max_mix = np.abs(mix).max() + EPSILON
    max_clean = np.abs(clean_signal).max() + EPSILON

    wav_write(os.path.join(sample_dir, "mix.wav"), args.sample_rate, (mix / max_mix).astype(np.float32))
    wav_write(os.path.join(sample_dir, "clean.wav"), args.sample_rate, (clean_signal / max_clean).astype(np.float32))

    metadata = {
        "id": sample_idx,
        "snr": snr_db,
        "T60_ms": int(beta * 1000),
        "mic_distance": round(mic_dist, 2),
        "speech_file": os.path.basename(speech_path),
    }

    summary_path = os.path.join(sample_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if sample_idx % 1000 == 0:
        print(f"  Generated {sample_idx} samples...")

    return metadata


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  DEREVERBERATION DATASET GENERATOR")
    print("=" * 60)
    print(f"  Speech dir   : {args.speech_dir}")
    noise_label = ', '.join(args.noise_dirs) if args.noise_dirs and args.noise_dirs != ['none'] else '(Gaussian noise only)'
    print(f"  Noise dirs   : {noise_label}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Num samples  : {args.num_samples}")
    print(f"  Sample rate  : {args.sample_rate} Hz")
    print(f"  SNR range    : [{args.snr_min}, {args.snr_max}] dB")
    print(f"  T60 range    : [{args.beta_min}, {args.beta_max}] s")
    print(f"  Room range   : [{args.room_min}, {args.room_max}] m")
    print(f"  Train ratio  : {args.train_ratio}")
    print(f"  Workers      : {args.num_workers}")
    print("=" * 60)

    # Collect files
    speech_files = collect_speech_files(args.speech_dir, args.speech_format)
    if args.noise_dirs and args.noise_dirs != ['none']:
        noise_files = collect_noise_files(args.noise_dirs)
    else:
        noise_files = []
        print("No noise directories provided — using Gaussian noise only.")

    # Split files into train / val
    num_train = int(args.num_samples * args.train_ratio)
    num_val = args.num_samples - num_train

    # Create output directories
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    for d in [train_dir, val_dir]:
        os.makedirs(os.path.join(d, "aud_files"), exist_ok=True)

    start_time = time.time()

    # Generate training samples
    print(f"\nGenerating {num_train} training samples...")
    worker_fn = partial(
        generate_single_sample,
        speech_files=speech_files,
        noise_files=noise_files,
        args=args,
        save_dir=train_dir,
    )
    with multiprocessing.Pool(args.num_workers) as pool:
        train_metadata = pool.map(worker_fn, range(num_train))
    train_metadata = [m for m in train_metadata if m is not None]

    # Generate validation samples
    print(f"\nGenerating {num_val} validation samples...")
    worker_fn = partial(
        generate_single_sample,
        speech_files=speech_files,
        noise_files=noise_files,
        args=args,
        save_dir=val_dir,
    )
    with multiprocessing.Pool(args.num_workers) as pool:
        val_metadata = pool.map(worker_fn, range(num_val))
    val_metadata = [m for m in val_metadata if m is not None]

    elapsed = time.time() - start_time

    # Save dataset summary
    summary = {
        "total_samples": len(train_metadata) + len(val_metadata),
        "train_samples": len(train_metadata),
        "val_samples": len(val_metadata),
        "sample_rate": args.sample_rate,
        "snr_range": [args.snr_min, args.snr_max],
        "t60_range": [args.beta_min, args.beta_max],
        "room_range": [args.room_min, args.room_max],
        "generation_time_sec": round(elapsed, 1),
    }
    summary_path = os.path.join(args.output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Train: {len(train_metadata)} | Val: {len(val_metadata)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {args.output_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
