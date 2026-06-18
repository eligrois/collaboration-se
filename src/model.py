"""
End-to-end dereverberation Lightning module.

Two-stage pipeline:
  Stage 1 (frozen): Spec2Spec — enhances log-magnitude spectrogram
  Stage 2 (trainable): Spec2Wav — refines complex STFT (RI domain)

Loss: SI-SDR improvement (input SI-SDR − output SI-SDR) to maximise enhancement.
"""

import os
import glob

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from .architectures import UNetSpec2Spec, UNetSpec2Wav
from .audio_utils import mag_phase_to_ri, ri_to_wav
from .metrics import si_sdr, MultiResolutionSTFTLoss

EPSILON = 1e-6


def load_submodel_weights(model: nn.Module, ckpt_path: str, prefix: str):
    """Load a sub-model's weights from a checkpoint.

    Tries the given prefix first (e.g. 'model_spec2spec.' for end-to-end ckpts),
    then falls back to 'model.' (for standalone sub-model ckpts).

    Args:
        model: the sub-model instance (e.g. UNetSpec2Spec)
        ckpt_path: path to the Lightning checkpoint
        prefix: key prefix in state_dict, e.g. 'model_spec2spec.'
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    filtered = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not filtered:
        # Fallback for standalone sub-model checkpoints (prefix = 'model.')
        fallback = "model."
        filtered = {k[len(fallback):]: v for k, v in state_dict.items() if k.startswith(fallback)}
    model.load_state_dict(filtered, strict=False)


class DereverbModule(pl.LightningModule):
    """End-to-end dereverberation model (spec2spec + spec2wav)."""

    def __init__(self, learning_rate: float = 1e-5,
                 gradient_clip_val: float = 0.5,
                 window_size: int = 512, overlap: int = 256,
                 norm_mag: bool = False, norm_mag_target: bool = False,
                 cut_first_freq: bool = True,
                 end2end_ckpt: str = None,
                 unfreeze_spec2spec: bool = False,
                 use_multi_res_stft_loss: bool = True,
                 multi_res_stft_spec2spec_weight: float = 1.0,
                 lr_scheduler: str = "plateau",
                 lr_scheduler_patience: int = 5,
                 lr_scheduler_factor: float = 0.5,
                 warmup_epochs: int = 0,
                 max_epochs: int = 1000):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.overlap = overlap
        self.norm_mag = norm_mag
        self.norm_mag_target = norm_mag_target
        self.cut_first_freq = cut_first_freq
        self.unfreeze_spec2spec = unfreeze_spec2spec
        self.use_multi_res_stft_loss = use_multi_res_stft_loss
        self.multi_res_stft_spec2spec_weight = multi_res_stft_spec2spec_weight
        self.criterion = nn.MSELoss()

        # Stage 1: spectrogram enhancement
        self.model_spec2spec = UNetSpec2Spec()
        # Stage 2: RI refinement
        self.model_spec2wav = UNetSpec2Wav()

        # Load weights from end-to-end checkpoint if provided
        if end2end_ckpt and os.path.isfile(end2end_ckpt):
            load_submodel_weights(self.model_spec2spec, end2end_ckpt, "model_spec2spec.")
            load_submodel_weights(self.model_spec2wav, end2end_ckpt, "model_spec2wav.")
            print(f"Loaded end2end checkpoint: {end2end_ckpt}")
        else:
            print("Training from scratch (no end2end checkpoint provided)")

        # Freeze/unfreeze spec2spec
        self._apply_spec2spec_freeze(unfreeze_spec2spec)

        # Multi-resolution STFT loss
        if use_multi_res_stft_loss:
            self.multi_res_stft_loss = MultiResolutionSTFTLoss()

    def _apply_spec2spec_freeze(self, unfreeze: bool):
        for p in self.model_spec2spec.parameters():
            p.requires_grad = unfreeze
        status = "UNFROZEN" if unfreeze else "FROZEN"
        print(f"  spec2spec is {status} (requires_grad={unfreeze})")

    def forward(self, batch):
        """Forward pass through both stages.

        Args:
            batch: tuple of (input_mag, input_phase, ...)

        Returns:
            pred_ri: predicted complex STFT (B, 2, F, T)
            pred_spec: predicted magnitude spectrogram (B, 1, F, T)
        """
        if self.unfreeze_spec2spec:
            pred_spec = self.model_spec2spec(batch)
        else:
            self.model_spec2spec.eval()
            with torch.no_grad():
                pred_spec = self.model_spec2spec(batch)

        input_ri = mag_phase_to_ri(pred_spec.detach(), batch[1], from_log=True)
        pred_ri = self.model_spec2wav(input_ri)
        return pred_ri, pred_spec

    def _compute_loss(self, pred_ri, pred_spec, batch):
        """SI-SDR based loss with optional multi-resolution STFT loss."""
        input_ri = mag_phase_to_ri(batch[0], batch[1], from_log=False)
        target_ri = mag_phase_to_ri(batch[2], batch[3], from_log=False)

        input_wav = ri_to_wav(input_ri, self.window_size, self.overlap, self.cut_first_freq)
        target_wav = ri_to_wav(target_ri, self.window_size, self.overlap, self.cut_first_freq)
        pred_wav = ri_to_wav(pred_ri, self.window_size, self.overlap, self.cut_first_freq)

        # SI-SDR improvement loss (minimize = maximize improvement)
        input_sisdr = si_sdr(input_wav, target_wav, eps_log_ref=1e-20, eps_log_err=1e-12)
        pred_sisdr = si_sdr(pred_wav, target_wav, eps_log_ref=1e-20, eps_log_err=1e-12)
        loss = input_sisdr - pred_sisdr

        self.log("si_sdr_improvement", pred_sisdr - input_sisdr, prog_bar=True)

        # Multi-resolution STFT loss
        if self.use_multi_res_stft_loss:
            if self.multi_res_stft_spec2spec_weight > 0:
                spec2spec_ri = mag_phase_to_ri(pred_spec, batch[3], from_log=True)
                spec2spec_wav = ri_to_wav(spec2spec_ri, self.window_size, self.overlap, self.cut_first_freq)
                mr_stft_s2s = self.multi_res_stft_loss(spec2spec_wav, target_wav)
                self.log("mr_stft_spec2spec", mr_stft_s2s, prog_bar=True)
                loss = loss + self.multi_res_stft_spec2spec_weight * mr_stft_s2s

        return loss

    def training_step(self, batch, batch_idx):
        pred_ri, pred_spec = self(batch)
        loss = self._compute_loss(pred_ri, pred_spec, batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_ri, pred_spec = self(batch)
        loss = self._compute_loss(pred_ri, pred_spec, batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        pred_ri, pred_spec = self(batch)
        loss = self._compute_loss(pred_ri, pred_spec, batch)
        self.log("test_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0)
        config = {"optimizer": optimizer}

        scheduler_type = getattr(self.hparams, "lr_scheduler", "none")
        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=getattr(self.hparams, "lr_scheduler_factor", 0.5),
                patience=getattr(self.hparams, "lr_scheduler_patience", 5),
                min_lr=1e-7,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=getattr(self.hparams, "max_epochs", 1000),
                eta_min=1e-7,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return config


class EMA(pl.Callback):
    """Exponential Moving Average of model weights for smoother convergence."""

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def on_fit_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.backup = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def find_latest_checkpoint(folder: str):
    """Find the latest 'last.ckpt' file in a folder tree."""
    ckpts = glob.glob(os.path.join(folder, "**", "last.ckpt"), recursive=True)
    if not ckpts:
        return None
    latest = max(ckpts, key=os.path.getctime)
    print(f"Resuming from checkpoint: {latest}")
    return latest
