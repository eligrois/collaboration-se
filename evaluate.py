#!/usr/bin/env python3
"""
Evaluate the dereverberation model on a test set.

Supports two data layouts:
  1. layout (default):
       <test_dir>/clean/{index}_clean.wav
       <test_dir>/dirty_samples/IR-{RT60}s_SNR-[{lo},{hi}]/{index}_dirty.wav

  2. "legacy" layout (--legacy flag):
       <data_root>/{split}/aud_files/{sample_name}/{mix.wav, clean.wav}

Results are grouped by RT60 and SNR condition with per-condition and overall
summary tables, including total mean improvement.

Usage:
    python evaluate.py
    python evaluate.py --test_dir /path/to/audio_samples
    python evaluate.py --legacy --data_root /path/to/data --split val
    python evaluate.py --eval_max_samples 50
"""

import os
import re
import sys
import argparse
from collections import defaultdict

import torch
import numpy as np
import torchaudio
from tabulate import tabulate

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from conf.config import Config
from src.model import DereverbModule, load_submodel_weights
from src.audio_utils import (
    preprocess_pair, mag_phase_to_ri, ri_to_wav, preprocess_audio,
)
from src.metrics import si_sdr, compute_pesq, compute_stoi


# ── Default test set path ──────────────────────────────────────────────────
DEFAULT_TEST_DIR = '/mount/data/ajal/audio_samples/audio/'


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dereverberation model")
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR,
                        help="Path to format test directory (clean/ + dirty_samples/)")
    parser.add_argument("--end2end_ckpt", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=None,
                        help="Max samples per condition (None = all)")
    # Legacy mode
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy data layout (data_root/split/aud_files/...)")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
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
        print(f"WARNING: Checkpoint not found at {cfg.end2end_ckpt}")

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def run_inference(model: DereverbModule, batch, device: torch.device):
    """Run model inference on a single sample."""
    in_mag = batch[0].unsqueeze(0).to(device)
    in_phase = batch[1].unsqueeze(0).to(device)
    pred_ri, _ = model((in_mag, in_phase))
    return pred_ri.cpu()


def evaluate_pair(model, input_path, target_path, cfg, device):
    """Evaluate a single (dirty, clean) pair and return metrics dict."""
    in_mag, in_phase, tgt_mag, tgt_phase = preprocess_pair(
        input_path, target_path,
        sample_rate=cfg.sample_rate,
        window_size=cfg.window_size,
        overlap=cfg.overlap,
        normalize_type=cfg.normalize_type,
        cut_first_freq=cfg.cut_first_freq,
        dont_use_end=cfg.dont_use_end,
        frames=None,
        train_mode=False,
    )

    pred_ri = run_inference(model, (in_mag, in_phase), device)

    input_ri = mag_phase_to_ri(in_mag.unsqueeze(0), in_phase.unsqueeze(0))
    target_ri = mag_phase_to_ri(tgt_mag.unsqueeze(0), tgt_phase.unsqueeze(0))

    input_wav = ri_to_wav(input_ri, cfg.window_size, cfg.overlap, cfg.cut_first_freq)
    target_wav = ri_to_wav(target_ri, cfg.window_size, cfg.overlap, cfg.cut_first_freq)
    pred_wav = ri_to_wav(pred_ri, cfg.window_size, cfg.overlap, cfg.cut_first_freq)
    pred_wav = pred_wav / 1.1 / pred_wav.abs().max()  # Normalize to prevent clipping (heuristic)

    input_sisdr = si_sdr(input_wav, target_wav).item()
    pred_sisdr = si_sdr(pred_wav, target_wav).item()
    delta_sisdr = pred_sisdr - input_sisdr

    input_np = input_wav.squeeze().numpy()
    target_np = target_wav.squeeze().numpy()
    pred_np = pred_wav.squeeze().numpy()

    min_len = min(len(input_np), len(target_np), len(pred_np))
    input_np, target_np, pred_np = input_np[:min_len], target_np[:min_len], pred_np[:min_len]

    pesq_in = compute_pesq(input_np, target_np, cfg.sample_rate)
    pesq_out = compute_pesq(pred_np, target_np, cfg.sample_rate)
    stoi_in = compute_stoi(input_np, target_np, cfg.sample_rate)
    stoi_out = compute_stoi(pred_np, target_np, cfg.sample_rate)

    return {
        "SI-SDR_in": round(input_sisdr, 2),
        "SI-SDR_out": round(pred_sisdr, 2),
        "delta_SI-SDR": round(delta_sisdr, 2),
        "PESQ_in": round(pesq_in, 3) if not np.isnan(pesq_in) else float("nan"),
        "PESQ_out": round(pesq_out, 3) if not np.isnan(pesq_out) else float("nan"),
        "STOI_in": round(stoi_in, 3) if not np.isnan(stoi_in) else float("nan"),
        "STOI_out": round(stoi_out, 3) if not np.isnan(stoi_out) else float("nan"),
    }, (input_np, pred_np, target_np)


def parse_condition(condition_name):
    """Parse 'IR-0.4s_SNR-[-5,0]' into (rt60='0.4', snr='[-5,0]')."""
    m = re.match(r"IR-(.+?)s_SNR-(.+)", condition_name)
    if m:
        return m.group(1), m.group(2)
    return condition_name, "unknown"


def compute_avg(records, metric_keys):
    """Compute mean of numeric values for the given metric keys."""
    avg = {}
    for k in metric_keys:
        vals = [r[k] for r in records if not np.isnan(r[k])]
        avg[k] = round(sum(vals) / len(vals), 3) if vals else float("nan")
    return avg


def format_val(v):
    """Format a value for table display."""
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return v


# ── Collect pairs for test set ──────────────────────────────
def collect_pairs(test_dir, max_samples=None):
    """Return list of (condition_name, rt60, snr, dirty_path, clean_path)."""
    clean_dir = os.path.join(test_dir, "clean")
    dirty_dir = os.path.join(test_dir, "dirty_samples")

    if not os.path.isdir(clean_dir) or not os.path.isdir(dirty_dir):
        raise FileNotFoundError(f"Expected clean/ and dirty_samples/ under: {test_dir}")

    conditions = sorted(os.listdir(dirty_dir))
    pairs = []
    for cond in conditions:
        cond_dir = os.path.join(dirty_dir, cond)
        if not os.path.isdir(cond_dir):
            continue
        rt60, snr = parse_condition(cond)
        dirty_files = sorted([f for f in os.listdir(cond_dir) if f.endswith(".wav")])
        if max_samples:
            dirty_files = dirty_files[:max_samples]
        for df in dirty_files:
            idx = df.replace("_dirty.wav", "")
            clean_file = os.path.join(clean_dir, f"{idx}_clean.wav")
            dirty_file = os.path.join(cond_dir, df)
            if os.path.isfile(clean_file):
                pairs.append((cond, rt60, snr, dirty_file, clean_file))
    return pairs


# ── Collect pairs for legacy format ──────────────────────────────────────
def collect_legacy_pairs(data_root, split, max_samples=None):
    """Return list of (sample_name, None, None, dirty_path, clean_path)."""
    data_dir = os.path.join(data_root, split, "aud_files")
    sample_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if max_samples:
        sample_dirs = sample_dirs[:max_samples]
    pairs = []
    for sd in sample_dirs:
        mix_path = os.path.join(data_dir, sd, "mix.wav")
        clean_path = os.path.join(data_dir, sd, "clean.wav")
        if os.path.isfile(mix_path) and os.path.isfile(clean_path):
            pairs.append((sd, None, None, mix_path, clean_path))
    return pairs


def build_summary_tables(all_results):
    """Build grouped summary tables by RT60, by SNR, and overall.

    Returns (by_condition_str, by_rt60_str, by_snr_str, overall_str).
    """
    metric_keys = ["SI-SDR_in", "SI-SDR_out", "delta_SI-SDR",
                    "PESQ_in", "PESQ_out", "STOI_in", "STOI_out"]

    # ── Group by condition ──
    by_condition = defaultdict(list)
    by_rt60 = defaultdict(list)
    by_snr = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)
        if r["rt60"] is not None:
            by_rt60[r["rt60"]].append(r)
        if r["snr"] is not None:
            by_snr[r["snr"]].append(r)

    # ── Per-condition table ──
    cond_rows = []
    for cond in sorted(by_condition.keys()):
        records = by_condition[cond]
        avg = compute_avg(records, metric_keys)
        cond_rows.append([cond, len(records)] + [format_val(avg[k]) for k in metric_keys])
    cond_headers = ["Condition", "N"] + metric_keys
    cond_str = tabulate(cond_rows, headers=cond_headers, tablefmt="grid")

    # ── Per-RT60 table ──
    rt60_rows = []
    for rt60 in sorted(by_rt60.keys(), key=lambda x: float(x)):
        records = by_rt60[rt60]
        avg = compute_avg(records, metric_keys)
        rt60_rows.append([f"{rt60}s", len(records)] + [format_val(avg[k]) for k in metric_keys])
    rt60_str = tabulate(rt60_rows, headers=["RT60", "N"] + metric_keys, tablefmt="grid") if rt60_rows else ""

    # ── Per-SNR table ──
    snr_rows = []
    for snr in sorted(by_snr.keys()):
        records = by_snr[snr]
        avg = compute_avg(records, metric_keys)
        snr_rows.append([snr, len(records)] + [format_val(avg[k]) for k in metric_keys])
    snr_str = tabulate(snr_rows, headers=["SNR", "N"] + metric_keys, tablefmt="grid") if snr_rows else ""

    # ── Overall averages ──
    overall_avg = compute_avg(all_results, metric_keys)
    overall_rows = [["TOTAL", len(all_results)] + [format_val(overall_avg[k]) for k in metric_keys]]
    overall_str = tabulate(overall_rows, headers=["", "N"] + metric_keys, tablefmt="grid")

    return cond_str, rt60_str, snr_str, overall_str


def main():
    args = parse_args()
    cfg = Config()
    for key in ["end2end_ckpt", "results_dir"]:
        val = getattr(args, key, None)
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)

    os.makedirs(cfg.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(cfg, device)

    # Collect evaluation pairs
    if args.legacy:
        data_root = args.data_root or cfg.data_root
        pairs = collect_legacy_pairs(data_root, args.split, args.eval_max_samples)
        data_source = os.path.join(data_root, args.split)
    else:
        pairs = collect_pairs(args.test_dir, args.eval_max_samples)
        data_source = args.test_dir

    print(f"\nEvaluating {len(pairs)} samples from: {data_source}")
    print(f"Results will be saved to: {cfg.results_dir}\n")

    audio_results_dir = os.path.join(cfg.results_dir, "audio")
    os.makedirs(audio_results_dir, exist_ok=True)

    all_results = []
    for i, (cond, rt60, snr, dirty_path, clean_path) in enumerate(pairs):
        try:
            metrics, (input_np, pred_np, target_np) = evaluate_pair(
                model, dirty_path, clean_path, cfg, device
            )
        except Exception as e:
            print(f"  Skipping {cond}/{os.path.basename(dirty_path)}: {e}")
            continue

        sample_id = os.path.basename(dirty_path).replace(".wav", "")
        record = {"condition": cond, "rt60": rt60, "snr": snr, "file": sample_id}
        record.update(metrics)
        all_results.append(record)

        # Save audio
        safe_cond = cond.replace("/", "_").replace(" ", "_")
        for name, wav_np in [("noisy", input_np), ("enhanced", pred_np), ("clean", target_np)]:
            wav_t = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
            peak = wav_t.abs().max() + 1e-8
            wav_t = wav_t / peak * 0.9
            out_path = os.path.join(audio_results_dir, f"{safe_cond}_{sample_id}_{name}.wav")
            torchaudio.save(out_path, wav_t, cfg.sample_rate)

        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            print(f"  [{i+1}/{len(pairs)}] {cond}/{sample_id}: "
                  f"SI-SDR {metrics['SI-SDR_in']:.1f} -> {metrics['SI-SDR_out']:.1f} "
                  f"(delta={metrics['delta_SI-SDR']:+.1f})")

    if not all_results:
        print("No results produced.")
        return

    # ── Build and print summary tables ──
    cond_str, rt60_str, snr_str, overall_str = build_summary_tables(all_results)

    sep = "=" * 100

    report_lines = []
    report_lines.append(f"Checkpoint: {cfg.end2end_ckpt}")
    report_lines.append(f"Data source: {data_source}")
    report_lines.append(f"Total samples evaluated: {len(all_results)}")
    report_lines.append("")

    report_lines.append(sep)
    report_lines.append("RESULTS BY CONDITION (RT60 x SNR)")
    report_lines.append(sep)
    report_lines.append(cond_str)
    report_lines.append("")

    if rt60_str:
        report_lines.append(sep)
        report_lines.append("RESULTS BY RT60")
        report_lines.append(sep)
        report_lines.append(rt60_str)
        report_lines.append("")

    if snr_str:
        report_lines.append(sep)
        report_lines.append("RESULTS BY SNR")
        report_lines.append(sep)
        report_lines.append(snr_str)
        report_lines.append("")

    report_lines.append(sep)
    report_lines.append("OVERALL RESULTS (TOTAL MEAN IMPROVEMENT)")
    report_lines.append(sep)
    report_lines.append(overall_str)

    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    # Save to file
    report_path = os.path.join(cfg.results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
