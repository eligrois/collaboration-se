"""
PyTorch Dataset and Lightning DataModule for pre-generated dereverberation data.

Expected data directory structure:
    <data_dir>/
        train/aud_files/
            file_0_snr_17_T60_528msec/
                mix.wav     # noisy/reverberant input
                clean.wav   # clean target
            file_1_.../
                ...
        val/aud_files/
            ...
        test/aud_files/
            ...
"""

import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .audio_utils import preprocess_pair


class DereverbDataset(Dataset):
    """Dataset that loads pre-generated (mix, clean) audio pairs."""

    def __init__(self, data_dir: str, sample_rate: int, window_size: int, overlap: int,
                 normalize_type: str = "max_spec", cut_first_freq: bool = True,
                 dont_use_end: int = 0, frames: int = 256, train_mode: bool = True,
                 max_samples: int = None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.normalize_type = normalize_type
        self.cut_first_freq = cut_first_freq
        self.dont_use_end = dont_use_end
        self.frames = frames
        self.train_mode = train_mode

        self.samples = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"DereverbDataset: {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.data_dir, self.samples[idx])
        input_path = os.path.join(sample_dir, "mix.wav")
        target_path = os.path.join(sample_dir, "clean.wav")

        in_mag, in_phase, tgt_mag, tgt_phase = preprocess_pair(
            input_path, target_path,
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            overlap=self.overlap,
            normalize_type=self.normalize_type,
            cut_first_freq=self.cut_first_freq,
            dont_use_end=self.dont_use_end,
            frames=self.frames,
            train_mode=self.train_mode,
        )
        return in_mag, in_phase, tgt_mag, tgt_phase


class DereverbDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping DereverbDataset for train/val/test splits."""

    def __init__(self, data_root: str, sample_rate: int, window_size: int, overlap: int,
                 normalize_type: str = "max_spec", cut_first_freq: bool = True,
                 dont_use_end: int = 0, frames: int = 256,
                 train_batch_size: int = 32, val_batch_size: int = 32,
                 num_workers: int = 4, pin_memory: bool = False,
                 max_val_samples: int = None):
        super().__init__()
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.normalize_type = normalize_type
        self.cut_first_freq = cut_first_freq
        self.dont_use_end = dont_use_end
        self.frames = frames
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_val_samples = max_val_samples

    def _make_dataset(self, split: str, train_mode: bool, max_samples: int = None):
        data_dir = os.path.join(self.data_root, split, "aud_files")
        return DereverbDataset(
            data_dir=data_dir,
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            overlap=self.overlap,
            normalize_type=self.normalize_type,
            cut_first_freq=self.cut_first_freq,
            dont_use_end=self.dont_use_end,
            frames=self.frames,
            train_mode=train_mode,
            max_samples=max_samples,
        )

    def train_dataloader(self):
        ds = self._make_dataset("train", train_mode=True)
        return DataLoader(ds, batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        ds = self._make_dataset("val", train_mode=True, max_samples=self.max_val_samples)
        return DataLoader(ds, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        ds = self._make_dataset("test", train_mode=False)
        return DataLoader(ds, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
