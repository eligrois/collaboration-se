"""
Configuration for the dereverberation training / evaluation pipeline.

Edit the values below to match your setup, then run:
    python train.py
    python evaluate.py
    python test_single.py --input_wav <path>
"""

import os
from dataclasses import dataclass, field

# Resolve paths relative to the speech enhancement project root (parent of conf/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────
    data_root: str = ""  # Set via CLI or point to your generated dataset
    sample_rate: int = 16000
    window_size: int = 512
    overlap: int = 256
    normalize_type: str = "max_spec"
    cut_first_freq: bool = True
    dont_use_end: int = 0
    frames: int = 256  # number of STFT time-frames per training sample

    # ── Model ─────────────────────────────────────────────────────────────
    norm_mag: bool = False
    norm_mag_target: bool = False

    # End-to-end checkpoint (for evaluation / test / resume training)
    # Default: baseline checkpoint shipped with the repo
    # If the file does not exist and no CLI override, trains from scratch
    end2end_ckpt: str = os.path.join(_PROJECT_ROOT, "models", "checkpoints", "baseline.ckpt")

    # Unfreeze spec2spec for joint finetuning (default: frozen)
    unfreeze_spec2spec: bool = True

    # Multi-resolution STFT loss
    use_multi_res_stft_loss: bool = True
    multi_res_stft_spec2spec_weight: float = 1.0  # weight for multi-res STFT loss on spec2spec output

    # ── Training ──────────────────────────────────────────────────────────
    train_batch_size: int = 8
    val_batch_size: int = 8
    learning_rate: float = 1e-4
    gradient_clip_val: float = 0.5
    patience: int = 50
    save_top_k: int = 5
    max_epochs: int = 1000
    precision: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    gpus: int = -1  # -1 = use all available GPUs
    max_val_samples: int = None  # limit validation set size (None = all)
    accumulate_grad_batches: int = 1  # gradient accumulation steps
    use_ema: bool = False  # exponential moving average of weights
    ema_decay: float = 0.999  # EMA decay rate
    lr_scheduler: str = "plateau"  # "plateau", "cosine", or "none"
    lr_scheduler_patience: int = 5  # patience for ReduceLROnPlateau
    lr_scheduler_factor: float = 0.5  # factor for ReduceLROnPlateau
    warmup_epochs: int = 0  # linear warmup epochs

    # ── Output paths (relative to project root) ─────────────────────────────
    model_dir: str = os.path.join(_PROJECT_ROOT, "models")
    results_dir: str = os.path.join(_PROJECT_ROOT, "results")

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_max_samples: int = 20  # number of samples from val set for evaluation

    # ── Test (single file) ────────────────────────────────────────────────
    test_input_wav: str = ""
