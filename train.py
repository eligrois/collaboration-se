#!/usr/bin/env python3
"""
Train the end-to-end dereverberation model.

Usage:
    python train.py
    python train.py --gpus 1 --train_batch_size 16 --max_epochs 100
"""

import os
import sys
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from conf.config import Config
from src.model import DereverbModule, EMA, find_latest_checkpoint
from src.dataset import DereverbDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train the dereverberation model")
    parser.add_argument("--data_root", type=str, default='/mount/data/ajal/solan/data/singapore_dereverb', help="Path to dataset root")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--end2end_ckpt", type=str, default=None,
                        help="End-to-end checkpoint to load weights from. If not given, trains from scratch.")
    parser.add_argument("--unfreeze_spec2spec", action="store_true", default=None,
                        help="Unfreeze spec2spec stage for joint finetuning")
    parser.add_argument("--use_multi_res_stft_loss", action="store_true", default=None,
                        help="Add multi-resolution STFT loss (default: True)")
    parser.add_argument("--no_multi_res_stft_loss", action="store_true",
                        help="Disable multi-resolution STFT loss")
    parser.add_argument("--multi_res_stft_spec2spec_weight", type=float, default=None,
                        help="Weight for multi-resolution STFT loss on spec2spec output (default: 1.0)")
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=None,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--use_ema", action="store_true", default=None,
                        help="Use EMA of model weights")
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=["plateau", "cosine", "none"],
                        help="LR scheduler type (default: plateau)")
    parser.add_argument("--lr_scheduler_patience", type=int, default=None)
    parser.add_argument("--lr_scheduler_factor", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--run_name", type=str, default='finetune',
                        help="Name for this run (used as subfolder under model_dir). "
                             "If not given, auto-generated from config.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (1 GPU, small batch)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # Override config with command-line arguments
    for key, val in vars(args).items():
        if key in ("no_multi_res_stft_loss",):
            continue
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)

    # Handle --no_multi_res_stft_loss flag
    if args.no_multi_res_stft_loss:
        cfg.use_multi_res_stft_loss = False

    # If no end2end_ckpt provided via CLI and default doesn't exist, train from scratch
    if args.end2end_ckpt is None and not os.path.isfile(cfg.end2end_ckpt):
        cfg.end2end_ckpt = None

    if args.debug:
        cfg.gpus = 1
        cfg.train_batch_size = 4
        cfg.val_batch_size = 4
        cfg.num_workers = 0
        cfg.max_val_samples = 20

    # Build a run-specific model directory so checkpoints don't clash
    if args.run_name:
        run_name = args.run_name
    else:
        # Auto-generate from key config values
        parts = [f"lr{cfg.learning_rate}"]
        parts.append(f"bs{cfg.train_batch_size}")
        if cfg.use_multi_res_stft_loss:
            parts.append(f"stft_s2s{cfg.multi_res_stft_spec2spec_weight}")
        else:
            parts.append("no_stft")
        if cfg.unfreeze_spec2spec:
            parts.append("unfreeze")
        parts.append(cfg.lr_scheduler)
        run_name = "_".join(parts)
    cfg.model_dir = os.path.join(cfg.model_dir, run_name)
    cfg.results_dir = os.path.join(cfg.results_dir, run_name)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ── Data ──
    data_module = DereverbDataModule(
        data_root=cfg.data_root,
        sample_rate=cfg.sample_rate,
        window_size=cfg.window_size,
        overlap=cfg.overlap,
        normalize_type=cfg.normalize_type,
        cut_first_freq=cfg.cut_first_freq,
        dont_use_end=cfg.dont_use_end,
        frames=cfg.frames,
        train_batch_size=cfg.train_batch_size,
        val_batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        max_val_samples=cfg.max_val_samples,
    )

    # ── Model ──
    model = DereverbModule(
        learning_rate=cfg.learning_rate,
        gradient_clip_val=cfg.gradient_clip_val,
        window_size=cfg.window_size,
        overlap=cfg.overlap,
        norm_mag=cfg.norm_mag,
        norm_mag_target=cfg.norm_mag_target,
        cut_first_freq=cfg.cut_first_freq,
        end2end_ckpt=cfg.end2end_ckpt,
        unfreeze_spec2spec=cfg.unfreeze_spec2spec,
        use_multi_res_stft_loss=cfg.use_multi_res_stft_loss,
        multi_res_stft_spec2spec_weight=cfg.multi_res_stft_spec2spec_weight,
        lr_scheduler=cfg.lr_scheduler,
        lr_scheduler_patience=cfg.lr_scheduler_patience,
        lr_scheduler_factor=cfg.lr_scheduler_factor,
        warmup_epochs=cfg.warmup_epochs,
        max_epochs=cfg.max_epochs,
    )

    # ── Resume from checkpoint ──
    resume_ckpt = find_latest_checkpoint(cfg.model_dir)

    # ── Callbacks ──
    ckpt_dir = os.path.join(cfg.model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_last=True,
        save_top_k=cfg.save_top_k,
        mode="min",
        verbose=True,
    )
    # Always save every epoch (regardless of val_loss improvement)
    periodic_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="periodic-{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,  # keep all
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        mode="min",
    )

    callbacks = [early_stop_callback, checkpoint_callback, periodic_checkpoint]
    if cfg.use_ema:
        callbacks.append(EMA(decay=cfg.ema_decay))
        print(f"  EMA enabled (decay={cfg.ema_decay})")

    # ── Trainer ──
    num_gpus = torch.cuda.device_count() if cfg.gpus == -1 else cfg.gpus
    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs,
        default_root_dir=cfg.model_dir,
        callbacks=callbacks,
        precision=cfg.precision,
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    if num_gpus > 1:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = num_gpus
        trainer_kwargs["strategy"] = "ddp"
    elif num_gpus == 1:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = 1
    else:
        trainer_kwargs["accelerator"] = "cpu"

    trainer = Trainer(**trainer_kwargs)

    print(f"\n{'='*60}")
    print(f"  Data root:    {cfg.data_root}")
    print(f"  Model dir:    {cfg.model_dir}")
    print(f"  GPUs:         {num_gpus}")
    print(f"  Batch size:   {cfg.train_batch_size}")
    print(f"  Accum batches:{cfg.accumulate_grad_batches}  (effective batch={cfg.train_batch_size * cfg.accumulate_grad_batches})")
    print(f"  LR scheduler: {cfg.lr_scheduler}")
    print(f"  EMA:          {cfg.use_ema} (decay={cfg.ema_decay})")
    print(f"  Resume ckpt:  {resume_ckpt}")
    print(f"{'='*60}\n")

    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    print(f"\nBest model: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
