#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FD4MM -> DiLA migration verification (the checklist from the migration doc).

Run this before starting training:

    python verify_fd4mm_dila.py --root ./data/train --seq-len 3

Steps:
  1. temporal-continuity measurement of the source indices
  2. sequence visualization (writes a PNG)
  3. value-range check of the emitted tensor
  4. real end-to-end batch through DiLA's MixedDataset / MixedBatchSampler
  5. memory-footprint probe for the configured (B, T, C, H, W)

Exit code is non-zero if any hard check fails.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./data/train", help="FD4MM data root")
    ap.add_argument("--seq-len", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sequence-mode", default="acb",
                    choices=["acb", "ab", "concat_pairs"])
    ap.add_argument("--frame-fill", default="reflect",
                    choices=["reflect", "interpolate", "repeat_last"])
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--save", default="fd4mm_sequence_check.png")
    ap.add_argument("--skip-loader", action="store_true",
                    help="skip step 4 (the real DiLA mixed-dataloader check)")
    args = ap.parse_args()

    from utils.fd4mm_dataset import (
        FD4MMDataset, measure_temporal_continuity, visualize_sequence,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failed = []

    # ---- step 1: temporal continuity --------------------------------------
    print("=" * 74)
    print("STEP 1/2  temporal continuity of the source indices")
    print("=" * 74)
    m = measure_temporal_continuity(args.root, num_samples=args.num_samples)
    for k, v in m.items():
        if not k.startswith("_"):
            print(f"  {k:>18s} = {v:9.4f}   (L2 in the transform's value range)")
    within = m["within_AB"]
    boundary = m["boundary_BA_next"]
    ratio = boundary / within
    print()
    print(f"  boundary(B_i, A_(i+1)) / within(A_i, B_i) = {ratio:.1f}x")
    if ratio > 3:
        print("  VERDICT: indices are NOT temporally continuous.")
        print("           Plan 1 (concatenating A/B pairs across indices) is INVALID;")
        print("           using the genuine within-sample A -> C -> B axis instead.")
    else:
        print("  VERDICT: continuity plausible.")

    # ---- step 3: value range ---------------------------------------------
    print()
    print("=" * 74)
    print("STEP 3  emitted tensor shape / value range")
    print("=" * 74)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    ds = FD4MMDataset(root=args.root, seq_len=args.seq_len,
                      sequence_mode=args.sequence_mode, frame_fill=args.frame_fill,
                      split="all", transform=transform, verbose=True)
    item = ds[0]["images"]
    print(f"  shape={tuple(item.shape)} dtype={item.dtype} "
          f"min={item.min():.4f} max={item.max():.4f}")
    if item.shape[0] != args.seq_len:
        failed.append(f"seq_len mismatch: {item.shape[0]} != {args.seq_len}")
    if item.shape[1] != 3:
        failed.append(f"expected 3 channels, got {item.shape[1]}")
    if not (item.min() >= -1e-6 and item.max() <= 1 + 1e-6):
        failed.append(f"value range outside [0,1]: [{item.min()}, {item.max()}]")

    # ---- step 2: visualization -------------------------------------------
    print()
    print("=" * 74)
    print("STEP 2  visual check")
    print("=" * 74)
    visualize_sequence(item, save_path=args.save,
                       title=f"FD4MM {args.sequence_mode} T={args.seq_len} ({args.frame_fill})")
    print(f"  wrote {args.save}   <-- confirm one coherent scene per sequence")

    # ---- step 4: real DiLA loader ----------------------------------------
    if not args.skip_loader:
        print()
        print("=" * 74)
        print("STEP 4  real DiLA MixedDataset / MixedBatchSampler batch")
        print("=" * 74)
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from utils.load_mixed_datasets import create_dataset_configs_from_config
        from utils.mixed_dataset import DatasetRegistry, create_mixed_dataloader

        cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = compose(config_name="train_model", overrides=["data=fd4mm"])

        md_cfg = cfg.data.multidataset
        data = {
            "dataset_names": list(md_cfg.dataset_names),
            "ratios": list(md_cfg.ratios),
            "batch_size": args.batch_size,
            "val_batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle_train": True,
            "seed": md_cfg.seed,
        }
        data["fd4mm"] = OmegaConf.to_container(md_cfg.fd4mm, resolve=True)
        data["fd4mm"]["sequence_mode"] = args.sequence_mode
        data["fd4mm"]["frame_fill"] = args.frame_fill

        DatasetRegistry.initialize_default_factories()
        configs = create_dataset_configs_from_config({"data": data})
        loader = create_mixed_dataloader(
            configs=configs, ratios=data["ratios"], batch_size=args.batch_size,
            seq_len=args.seq_len, transform=transform, mode="train",
            num_workers=0, pin_memory=False, shuffle=True, seed=data["seed"],
            logger=logging.getLogger("verify"),
        )
        batch = next(iter(loader))
        obs = batch["observations"]
        print(f"  observations: {tuple(obs.shape)} {obs.dtype} "
              f"range=[{obs.min():.4f}, {obs.max():.4f}]")
        print(f"  dataset_indices: {batch['dataset_indices'].tolist()}")
        deltas = (obs[:, 1:] - obs[:, :-1]).pow(2).mean(dim=(2, 3, 4)).sqrt()
        print(f"  adjacent-frame L2: mean={deltas.mean():.4f} max={deltas.max():.4f}")
        if deltas.max() > 0.2:
            failed.append(f"adjacent frames jump (max L2 {deltas.max():.4f}); "
                          f"sequence may be temporally incoherent")
        expected = (args.batch_size, args.seq_len, 3)
        if tuple(obs.shape[:3]) != expected:
            failed.append(f"batch shape {tuple(obs.shape)} != (B,T,C)={expected}")
        del obs, batch
    else:
        obs = None

    # ---- step 5: memory footprint ----------------------------------------
    print()
    print("=" * 74)
    print("STEP 5  memory footprint")
    print("=" * 74)
    elems = args.batch_size * args.seq_len * 3 * 256 * 256
    print(f"  one batch (B={args.batch_size}, T={args.seq_len}, 3, 256, 256) fp32 "
          f"= {elems * 4 / 2**20:.1f} MiB; fp16 = {elems * 2 / 2**20:.1f} MiB")
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  free/total: {[round(x / 2**30, 2) for x in torch.cuda.mem_get_info()]} GiB")
    else:
        print("  CUDA not available")
    print("  If OOM: lower batch_size or seq_len first (seq_len also controls T).")

    print()
    print("=" * 74)
    if failed:
        print("FAILED CHECKS:")
        for f in failed:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
