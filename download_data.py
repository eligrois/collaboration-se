#!/usr/bin/env python3
"""
Download clean speech data for dereverberation training.

Supported datasets:
  1. "singapore" (default) — IMDA National Speech Corpus (Singaporean English)
     https://huggingface.co/datasets/recursal/reprocessed_singapore_national_speech_corpus
  2. "malay" — Malaya-Speech Malay STT (Bahasa Melayu)
     https://huggingface.co/datasets/mesolitica/malaya-speech-malay-stt

Output structure (LibriSpeech-like, compatible with generate_dataset.py):
    <output_dir>/
    ├── <speaker_id>/
    │   ├── <sample_key>.wav
    │   └── ...
    └── ...

Usage:
    # Singapore English (default)
    python download_singapore_data.py --output_dir ./data/singapore_speech

    # Malay
    python download_singapore_data.py --dataset malay --output_dir ./data/malay_speech

    # Both (two runs)
    python download_singapore_data.py --output_dir ./data/sg_speech --max_samples 50000
    python download_singapore_data.py --dataset malay --output_dir ./data/malay_speech --max_samples 50000

Requirements:
    pip install datasets soundfile huggingface_hub
"""

import os
import json
import argparse
import soundfile as sf
import numpy as np


# ── Dataset registry ─────────────────────────────────────────────────────────
DATASETS = {
    "singapore": {
        "hf_id": "recursal/reprocessed_singapore_national_speech_corpus",
        "revision": "refs/convert/parquet",
        "audio_col": "flac",             # column containing audio
        "speaker_col": ("json", "SpeakerID"),  # (column, key) for speaker ID
        "key_col": "__key__",            # column for sample key / filename
        "description": "IMDA National Speech Corpus — Singaporean English",
    },
    "malay": {
        "hf_id": "mesolitica/malaya-speech-malay-stt",
        "revision": None,
        "audio_col": "filename",         # column containing audio
        "speaker_col": None,             # no speaker metadata
        "key_col": None,                 # no key column — use index
        "description": "Malaya-Speech — Bahasa Melayu (1.6M samples)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download speech data for dereverberation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="singapore",
                        choices=list(DATASETS.keys()),
                        help="Which dataset to download")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./data/<dataset>_speech)")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to download")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate")
    parser.add_argument("--min_duration", type=float, default=1.0,
                        help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=15.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token (if dataset requires auth)")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"./data/{args.dataset}_speech"
    return args


def _extract_audio(sample, audio_cols):
    """Extract audio array and sample rate from a dataset sample.

    Handles both dict-style (older datasets lib) and AudioDecoder-style
    (newer datasets lib with torchcodec) column formats.

    Args:
        sample: dataset row
        audio_cols: list of column names to try for audio data

    Returns (audio_array_float32, sample_rate) or None.
    """
    for col in audio_cols:
        if col not in sample:
            continue
        obj = sample[col]

        # Dict format: {"array": np.ndarray, "sampling_rate": int}
        if isinstance(obj, dict) and "array" in obj:
            return np.array(obj["array"], dtype=np.float32), obj.get("sampling_rate", 16000)

        # AudioDecoder format: supports obj["array"], obj["sampling_rate"]
        if hasattr(obj, "__getitem__"):
            try:
                arr = np.array(obj["array"], dtype=np.float32)
                try:
                    sr = int(obj["sampling_rate"])
                except Exception:
                    sr = 16000
                return arr, sr
            except (KeyError, TypeError):
                continue

    return None


def process_sample(sample, args, saved_count, ds_cfg):
    """Extract audio from a single dataset sample. Returns (speaker_id, key, audio_array) or None."""
    audio_col = ds_cfg["audio_col"]
    result = _extract_audio(sample, [audio_col, "audio", "flac"])
    if result is None:
        return None

    audio_array, sr = result

    # Duration filter
    duration = len(audio_array) / sr
    if duration < args.min_duration or duration > args.max_duration:
        return None

    # Silence check
    if np.abs(audio_array).max() < 1e-6:
        return None

    # Resample if needed
    if sr != args.sample_rate:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=args.sample_rate)

    # Get sample key
    key_col = ds_cfg.get("key_col")
    if key_col and key_col in sample:
        key = sample[key_col]
    else:
        key = f"sample_{saved_count:06d}"

    # Get speaker ID
    speaker_spec = ds_cfg.get("speaker_col")
    speaker_id = "all"
    if speaker_spec is not None:
        col_name, field_name = speaker_spec
        metadata = sample.get(col_name, {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                metadata = {}
        if isinstance(metadata, dict):
            speaker_id = str(metadata.get(field_name, "unknown"))

    return speaker_id, key, audio_array


def main():
    args = parse_args()
    ds_cfg = DATASETS[args.dataset]

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Install with: pip install datasets soundfile huggingface_hub")
        return

    print("=" * 60)
    print("  SPEECH DATA DOWNLOADER")
    print("=" * 60)
    print(f"  Dataset     : {ds_cfg['description']}")
    print(f"  HF ID       : {ds_cfg['hf_id']}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Max samples : {args.max_samples}")
    print(f"  Sample rate : {args.sample_rate} Hz")
    print(f"  Duration    : [{args.min_duration}, {args.max_duration}] sec")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset via streaming
    print("\nConnecting to HuggingFace (streaming)...")
    load_kwargs = {"split": "train", "streaming": True}
    if ds_cfg["revision"]:
        load_kwargs["revision"] = ds_cfg["revision"]
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    try:
        dataset = load_dataset(ds_cfg["hf_id"], **load_kwargs)
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        print("Try providing --hf_token if the dataset requires authentication.")
        return

    print("  Connected!")

    saved = 0
    skipped = 0
    errors = 0

    for sample in dataset:
            if saved >= args.max_samples:
                break

            try:
                result = process_sample(sample, args, saved, ds_cfg)
                if result is None:
                    skipped += 1
                    continue

                speaker_id, key, audio_array = result

                # Save
                speaker_dir = os.path.join(args.output_dir, speaker_id)
                os.makedirs(speaker_dir, exist_ok=True)
                safe_key = key.replace("/", "_").replace("\\", "_")
                out_path = os.path.join(speaker_dir, f"{safe_key}.wav")
                sf.write(out_path, audio_array, args.sample_rate)

                saved += 1
                if saved % 500 == 0:
                    print(f"    Saved {saved}/{args.max_samples} "
                          f"(skipped: {skipped}, errors: {errors})")

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"    Error: {e}")
                elif errors == 11:
                    print("    (suppressing further error messages)")
                continue

    # Summary
    speakers = [d for d in os.listdir(args.output_dir)
                if os.path.isdir(os.path.join(args.output_dir, d))]

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Saved     : {saved} samples")
    print(f"  Speakers  : {len(speakers)}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    print(f"  Output    : {args.output_dir}")
    print(f"{'=' * 60}")
    print(f"\nNext step: Generate dereverberation dataset with:")
    print(f"  python generate_dataset.py \\")
    print(f"    --speech_dir {args.output_dir} \\")
    print(f"    --speech_format wav \\")
    print(f"    --output_dir ./data/singapore_dereverb_dataset")


if __name__ == "__main__":
    main()
