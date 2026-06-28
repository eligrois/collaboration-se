# Speech Dereverberation — Standalone Training & Evaluation

## Prerequisites — External Data to Copy

Before running any stage, copy the following into this project:

### 1. Baseline Checkpoint (required for fine-tuning & evaluation)

Already included at:
```
models/checkpoints/baseline.ckpt
```
This is the default checkpoint used for fine-tuning (`train.py`) and evaluation (`evaluate.py`, `test_single.py`).

### 2. Clean Speech Data (required for dataset generation)

You need a directory of clean speech files (`.wav`, `.flac`, or `.mp3`). Place or symlink it under `./data/`. Options:

- **Any clean speech dataset** — a directory tree of audio files
- **LibriSpeech** — download `train-clean-360` from [openslr.org](https://www.openslr.org/12)
- **Singapore NSC** — use `python download_data.py --output_dir ./data/singapore_speech`
- **Malay** — use `python download_data.py --dataset malay --output_dir ./data/malay_speech`

### 3. Noise Data (optional, for dataset generation)

To add real noise during dataset generation, provide one or more directories of noise `.wav` files via `--noise_dirs`:

- **WHAMR** noise dataset
- **DNS Challenge** noise dataset

If no noise directories are provided, Gaussian noise is used as fallback.

### 4. Test Data (required for evaluation)

For `evaluate.py`, provide a test directory via `--test_dir` with the following structure:
```
test_data/
├── clean/                        # Clean reference files (*_clean.wav)
└── dirty_samples/
    ├── IR-0.2s_SNR-[-10,-5]/     # Reverberant files (*_dirty.wav)
    ├── IR-0.2s_SNR-[-5,0]/
    └── ...
```

Or use `--legacy` mode with your generated dataset: `evaluate.py --legacy --data_root ./data/my_dataset --split val`

---

## Overview

End-to-end deep learning pipeline for **speech dereverberation** — removing room reverberation from recorded speech. The model processes a noisy/reverberant audio signal and produces a clean (enhanced) version.

### Architecture

Two-stage pipeline:

1. **Spec2Spec** (`UNetSpec2Spec`): U-Net + Transformer bottleneck — enhances magnitude spectrogram.
2. **Spec2Wav** (`UNetSpec2Wav`): U-Net — refines complex STFT (real + imaginary) to produce final waveform.

All audio: 16 kHz, 512-sample FFT, 256-sample hop.

---

## Project Structure

```
speech_enhancement/
├── conf/config.py              # All configuration parameters
├── src/
│   ├── architectures.py        # Neural network definitions
│   ├── audio_utils.py          # STFT, preprocessing, waveform reconstruction
│   ├── dataset.py              # PyTorch Dataset and Lightning DataModule
│   ├── layers.py               # Building blocks (Conv, Down, Up, etc.)
│   ├── metrics.py              # SI-SDR, PESQ, STOI evaluation metrics
│   └── model.py                # Lightning module, loss, checkpoint loading
├── train.py                    # Training script
├── evaluate.py                 # Batch evaluation with grouped results
├── test_single.py              # Single-file inference
├── generate_dataset.py         # Synthetic dataset generation
├── download_data.py            # Download speech data (Singapore / Malay)
├── models/
│   └── checkpoints/
│       └── baseline.ckpt       # Pre-trained baseline checkpoint
├── data/                       # Dataset storage (gitignored)
├── results/                    # Evaluation outputs (gitignored)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get clean speech data (one of the options above)
#    Example: download Singapore English
python download_data.py --output_dir ./data/singapore_speech --max_samples 60000

# 3. Generate dereverberation training data
python generate_dataset.py \
    --speech_dir ./data/singapore_speech \
    --speech_format wav \
    --output_dir ./data/singapore_dereverb \
    --num_samples 60000

# 4. Train (fine-tunes from baseline.ckpt by default)
python train.py --data_root ./data/singapore_dereverb

# 5. Evaluate
python evaluate.py --test_dir /path/to/test_data --end2end_ckpt ./models/checkpoints/baseline.ckpt
```

---

## Step 1: Download Clean Speech Data

### Option A: Singapore English (NSC)

```bash
python download_data.py \
    --output_dir ./data/singapore_speech \
    --max_samples 10000 \
    --min_duration 1.0 \
    --max_duration 15.0
```

If the dataset requires authentication:
```bash
python download_data.py \
    --output_dir ./data/singapore_speech \
    --hf_token YOUR_HUGGINGFACE_TOKEN
```

### Option B: LibriSpeech (English)

```bash
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzf train-clean-360.tar.gz
```

### Option C: Any Clean Speech Dataset

Any directory of clean speech files works. The generator walks the directory recursively looking for audio files (`.flac`, `.wav`, or `.mp3`).

---

## Step 2: Generate Dereverberation Dataset

```bash
python generate_dataset.py \
    --speech_dir /path/to/clean_speech \
    --output_dir ./data/my_dereverb_dataset \
    --num_samples 60000 \
    --sample_rate 16000 \
    --speech_format wav
```

With real noise:
```bash
python generate_dataset.py \
    --speech_dir /path/to/clean_speech \
    --noise_dirs /path/to/whamr_noise /path/to/dns_noise \
    --output_dir ./data/my_dereverb_dataset \
    --num_samples 60000
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--speech_dir` | (required) | Path to clean speech files |
| `--noise_dirs` | `[]` | Noise file directories (Gaussian noise if empty) |
| `--output_dir` | `./generated_dataset` | Where to save the dataset |
| `--num_samples` | 60000 | Total samples to generate |
| `--train_ratio` | 0.9 | Train/val split ratio |
| `--snr_min` / `--snr_max` | 0 / 20 | SNR range in dB |
| `--beta_min` / `--beta_max` | 0.2 / 1.0 | T60 (reverberation time) range in seconds |
| `--room_min` / `--room_max` | 3.0 / 10.0 | Room dimension range in meters |
| `--dist_mic_min` / `--dist_mic_max` | 1.0 / 10.0 | Speaker–mic distance range in meters |
| `--dist_from_wall` | 0.5 | Min distance from wall in meters |
| `--num_workers` | 8 | Parallel generation workers |
| `--speech_format` | flac | File extension to search for |

### Output Format

```
my_dereverb_dataset/
├── train/aud_files/
│   ├── file_0_snr_5_T60_300msec/
│   │   ├── mix.wav          # Reverberant + noise
│   │   ├── clean.wav        # Original clean speech
│   │   └── summary.json     # Sample metadata
│   └── ...
├── val/aud_files/
│   └── ...
└── dataset_summary.json
```

---

## Step 3: Train

```bash
# Fine-tune from baseline checkpoint (default)
python train.py --data_root ./data/my_dereverb_dataset

# Fine-tune from a specific checkpoint
python train.py --data_root ./data/my_dereverb_dataset \
    --end2end_ckpt ./models/checkpoints/baseline.ckpt

# Debug mode (fast test, 1 GPU, small batch)
python train.py --data_root ./data/my_dereverb_dataset --debug --max_epochs 1
```

Training auto-resumes from the latest checkpoint in `model_dir`. Monitor with TensorBoard:
```bash
tensorboard --logdir ./models/lightning_logs
```

### All Training Options

| CLI Argument | Config Default | Description |
|---|---|---|
| `--data_root` | `""` | Path to dataset root (required) |
| `--model_dir` | `./models` | Directory to save checkpoints |
| `--end2end_ckpt` | `baseline.ckpt` | Checkpoint to load weights from (fine-tune) |
| `--train_batch_size` | 64 | Training batch size |
| `--val_batch_size` | 64 | Validation batch size |
| `--learning_rate` | 1e-4 | Adam learning rate |
| `--max_epochs` | 1000 | Maximum training epochs |
| `--gpus` | -1 (all) | Number of GPUs to use |
| `--num_workers` | 4 | DataLoader workers |
| `--patience` | 50 | Early stopping patience (epochs) |
| `--precision` | 32 | Training precision (16 or 32) |
| `--gradient_clip_val` | 0.5 | Gradient clipping norm |
| `--accumulate_grad_batches` | 1 | Gradient accumulation steps (effective batch = batch_size × accum) |
| `--max_val_samples` | None | Limit validation set size |
| `--unfreeze_spec2spec` | True | Unfreeze spec2spec for joint fine-tuning |
| `--use_multi_res_stft_loss` | True | Add multi-resolution STFT loss |
| `--no_multi_res_stft_loss` | — | Disable multi-resolution STFT loss |
| `--multi_res_stft_spec2spec_weight` | 1.0 | Weight for multi-res STFT loss on spec2spec output |
| `--multi_res_stft_spec2wav_weight` | 0.5 | Weight for multi-res STFT loss on spec2wav output |
| `--use_ema` | False | Enable Exponential Moving Average of weights |
| `--ema_decay` | 0.999 | EMA decay rate |
| `--lr_scheduler` | `plateau` | LR scheduler: `plateau`, `cosine`, or `none` |
| `--lr_scheduler_patience` | 5 | Patience for ReduceLROnPlateau |
| `--lr_scheduler_factor` | 0.5 | Factor for ReduceLROnPlateau |
| `--warmup_epochs` | 0 | Linear warmup epochs |
| `--run_name` | auto-generated | Name for this run (subfolder under model_dir) |
| `--debug` | False | Debug mode: 1 GPU, batch=4, workers=0 |

---

## Step 4: Evaluate

### format test set (grouped by RT60 × SNR)

```bash
python evaluate.py --test_dir /path/to/test_data
```

Test directory structure:
```
test_data/
├── clean/                        # Clean reference files (*_clean.wav)
└── dirty_samples/
    ├── IR-0.2s_SNR-[-10,-5]/     # Files per condition (*_dirty.wav)
    ├── IR-0.2s_SNR-[-5,0]/
    └── ...
```

### Custom checkpoint

```bash
python evaluate.py --test_dir /path/to/test_data --end2end_ckpt ./models/checkpoints/baseline.ckpt
```

### Legacy format (data_root/split/aud_files/...)

```bash
python evaluate.py --legacy --data_root ./data/my_dataset --split val
```

### Results Output

Results are grouped in multiple summary tables:

1. **Per condition** (RT60 × SNR): average metrics for each combination
2. **Per RT60**: average across all SNR levels
3. **Per SNR**: average across all RT60 values
4. **Overall**: total mean improvement across all samples

Metrics: SI-SDR (in/out/delta), PESQ (in/out), STOI (in/out)

---

## Single-File Inference

```bash
python test_single.py --input_wav /path/to/noisy_audio.wav
python test_single.py --input_wav /path/to/noisy.wav --end2end_ckpt ./models/checkpoints/baseline.ckpt
```

Output saved to `results/single_test/`.

---

## Configuration

All settings in [`conf/config.py`](conf/config.py). CLI arguments override config defaults.

---

## Training Tips

1. **Start from baseline checkpoint** for faster convergence
2. **Monitor SI-SDR improvement** in training logs
3. **Diverse data**: vary T60 (200ms–1000ms), room sizes, SNR levels
4. **LR**: `1e-4` from scratch or for fine-tuning
5. **Batch size**: 64+ for stable training, use `--accumulate_grad_batches` if GPU memory is limited

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `--train_batch_size` or `--frames` |
| `PESQ: N/A` | `pip install pesq` |
| `STOI: N/A` | `pip install pystoi` |
| `rir-generator` missing | `pip install rir-generator` |
| Dataset auth error | Pass `--hf_token YOUR_TOKEN` to download script |
