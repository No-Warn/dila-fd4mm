# -*- coding: utf-8 -*-
"""
FD4MM -> DiLA dataset adapter.

This module adapts the FD4MM / Oh et al. (ECCV 2018) motion-magnification frame
folders into the observation-sequence format expected by DiLA's mixed-dataset
loader: each item is a ``(T, C, H, W)`` float tensor with ``T = seq_len``.

--------------------------------------------------------------------------
IMPORTANT: verified temporal semantics of the FD4MM training data
--------------------------------------------------------------------------
The FD4MM training split is the *synthetic* dataset of
Oh et al., "Learning-based Video Motion Magnification" (ECCV 2018), regenerated
by FD4MM (CVPR 2024). Every index ``i`` is an **independent synthetic scene**:

  * its own random COCO background   (``meta/%06d.json -> background_file``)
  * its own randomly pasted moving objects
  * its own random magnification factor (``train_mf.txt``; all 100000 values
    are unique, so there are no per-video runs)

Within a single index ``i`` the three folders are three *temporally ordered*
frames of that one scene::

    frameA/%06d.png   t = 0.0   (start)
    frameC/%06d.png   t = 0.5   (intermediate, between A and B)
    frameB/%06d.png   t = 1.0   (end)

so the only genuine temporal axis available is ``A -> C -> B``.

Across indices there is **no** temporal continuity. Measured on this dataset
(mean pixel L2, 0-255 scale):

    ||A_i - C_i||          =  9.2      (within one genuine scene)
    ||C_i - B_i||          =  8.3
    ||A_i - B_i||          = 13.5
    ||B_i - A_(i+1)||      = 94.6      <-- same magnitude as random pairs
    ||A_i - A_(i+1)||      = 94.7

``B_i`` and ``A_(i+1)`` are never closer than ``A_i`` and ``B_i`` (0/300
samples), and 200/200 consecutive ``meta`` files reference different
backgrounds. Concatenating ``[A_1, B_1, A_2, B_2, ...]`` therefore splices
unrelated scenes and would teach DiLA that a latent action means "teleport to a
random scene", which destroys the structure/content disentanglement the model
is supposed to learn.

Consequently this adapter builds sequences from the genuine within-sample
``A -> C -> B`` axis. The concatenation strategy is still available through
``sequence_mode="concat_pairs"`` for reproduction/diagnostics, but it is not
the default and emits a warning.

--------------------------------------------------------------------------
Value range
--------------------------------------------------------------------------
The value range is whatever ``transform`` produces. DiLA's training loop
(``train.py::_setup_multidataset_loaders``) passes
``Resize((256, 256)) + ToTensor()``, i.e. ``[0, 1]``; the DINOv2/RAE encoder
applies its own normalization (``RAE.encoder_params.normalize``). Do **not**
rescale to ``[-1, 1]`` here: that would double-normalize the encoder input.
"""

from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:  # torchvision is a hard dependency of the repo, but keep the import defensive
    import torchvision.transforms as transforms
except Exception:  # pragma: no cover
    transforms = None


__all__ = [
    "FD4MMDataset",
    "measure_temporal_continuity",
    "visualize_sequence",
]

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

#: how many genuine frames each sequence_mode provides
_SEQUENCE_MODES = ("acb", "ab", "concat_pairs")
_FRAME_FILLS = ("reflect", "interpolate", "repeat_last")


def _list_frames(folder: str) -> Dict[int, str]:
    """Map integer frame index -> path for every image directly in ``folder``."""
    if not os.path.isdir(folder):
        return {}
    out: Dict[int, str] = {}
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in _IMG_EXTS:
            continue
        try:
            out[int(stem)] = os.path.join(folder, name)
        except ValueError:
            continue
    return out


class FD4MMDataset(Dataset):
    """FD4MM frame folders -> DiLA ``(T, C, H, W)`` observation sequences.

    Args:
        root: data root containing ``frameA/`` and ``frameB/`` (and normally
            ``frameC/``, ``meta/``, ``train_mf.txt``).
        seq_len: number of frames ``T`` per item. Must match DiLA's global
            ``seq_len`` so mixed-dataset batches stack.
        sequence_mode:
            * ``"acb"`` (default) -- genuine temporal order ``A -> C -> B``.
              Falls back to ``A -> B`` when ``frameC/`` is absent or
              ``use_frame_c=False``.
            * ``"ab"`` -- the minimal ``A -> B`` pair.
            * ``"concat_pairs"`` -- the FD4MM->DiLA document's plan 1:
              ``[A_s, B_s, A_(s+1), B_(s+1), ...]`` across consecutive indices.
              **Temporally invalid for this dataset**; kept for diagnostics.
        frame_fill: how to reach ``seq_len`` when the genuine axis is shorter.
            * ``"reflect"`` (default) -- palindrome extension, every emitted
              frame is a real observation (velocity reverses at the folds).
            * ``"interpolate"`` -- linear pixel interpolation. Time is
              monotone but intermediate frames are not real; measured
              ``||C - (A+B)/2|| / ||A-B|| ~= 0.5`` shows pixel interpolation
              ghosts moving objects rather than reproducing rendered frames.
            * ``"repeat_last"`` -- pad with the final frame (static tail;
              documented as discouraged because it teaches a no-op action).
        split: ``"train"``, ``"val"``/``"test"`` or ``"all"``.
        split_ratio: fraction of indices used for ``"train"``.
        seed: seed for the deterministic train/val permutation.
        num_data: optionally cap the number of source indices considered.
        transform: per-frame transform (PIL -> tensor). Defaults to
            ``Resize(image_size) + ToTensor()``.
        image_size: used only to build the default transform.
        pairs_per_sample: ``K`` for ``sequence_mode="concat_pairs"`` (``T = 2K``).
        stride: window stride for ``sequence_mode="concat_pairs"``
            (defaults to ``K``, i.e. non-overlapping).
        add_noise: add Gaussian observation noise after the transform.
        noise_std: noise standard deviation in the transform's value range.
        verbose: print index/continuity diagnostics at construction time.
    """

    def __init__(
        self,
        root: str,
        seq_len: int = 16,
        sequence_mode: str = "acb",
        frame_fill: str = "reflect",
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        num_data: Optional[int] = None,
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        use_frame_c: bool = True,
        pairs_per_sample: int = 4,
        stride: Optional[int] = None,
        add_noise: bool = False,
        noise_std: float = 0.03,
        verbose: bool = True,
    ) -> None:
        if sequence_mode not in _SEQUENCE_MODES:
            raise ValueError(
                f"sequence_mode must be one of {_SEQUENCE_MODES}, got {sequence_mode!r}"
            )
        if frame_fill not in _FRAME_FILLS:
            raise ValueError(f"frame_fill must be one of {_FRAME_FILLS}, got {frame_fill!r}")

        self.root = root
        self.sequence_mode = sequence_mode
        self.frame_fill = frame_fill
        self.split = "val" if split in ("val", "test") else split
        self.split_ratio = float(split_ratio)
        self.seed = int(seed)
        self.use_frame_c = bool(use_frame_c)
        self.pairs_per_sample = int(pairs_per_sample)
        self.stride = int(stride) if stride is not None else int(pairs_per_sample)
        self.add_noise = bool(add_noise)
        self.noise_std = float(noise_std)
        self.seq_len = int(seq_len)

        # ---- transforms -------------------------------------------------
        if transform is not None:
            self.transform = transform
        else:
            if transforms is None:  # pragma: no cover
                raise RuntimeError("torchvision is required when transform=None")
            ops: List[Callable] = []
            if image_size is not None:
                ops.append(transforms.Resize((int(image_size), int(image_size))))
            ops.append(transforms.ToTensor())
            self.transform = transforms.Compose(ops)

        # ---- discover frames --------------------------------------------
        self._paths_a = _list_frames(os.path.join(root, "frameA"))
        self._paths_b = _list_frames(os.path.join(root, "frameB"))
        self._paths_c = _list_frames(os.path.join(root, "frameC"))

        common = sorted(set(self._paths_a) & set(self._paths_b))
        if num_data is not None:
            common = common[: int(num_data)]
        if not common:
            raise FileNotFoundError(
                f"No frameA/frameB pairs found under {root!r}. Expected "
                f"{root}/frameA/%06d.png and {root}/frameB/%06d.png."
            )

        # frameC is optional; only keep indices that actually have it when used.
        self._has_frame_c = bool(self._paths_c)
        if self.sequence_mode == "acb" and self.use_frame_c:
            if not self._has_frame_c:
                warnings.warn(
                    "frameC/ not found: falling back to A -> B sequences. "
                    "The genuine 3-frame temporal axis A -> C -> B is unavailable.",
                    stacklevel=2,
                )

        if self.sequence_mode == "concat_pairs":
            warnings.warn(
                "sequence_mode='concat_pairs' concatenates A/B pairs from "
                "consecutive indices. This FD4MM split is made of independent "
                "synthetic scenes, so the resulting sequence jumps between "
                "unrelated backgrounds (measured ||B_i - A_(i+1)|| = 94.6, the "
                "same as random pairs). Use it only for reproduction/diagnostics.",
                stacklevel=2,
            )

        self.indices = self._make_split(common)
        self.samples = self._build_samples()

        if verbose:
            n_frames = 3 if (self.sequence_mode == "acb" and self._has_frame_c and self.use_frame_c) else 2
            print(f"[FD4MMDataset] root={root}")
            print(f"  sequence_mode={self.sequence_mode!r}, genuine frames/sample={n_frames}")
            print(f"  seq_len T={self.seq_len}, frame_fill={self.frame_fill!r}")
            print(f"  split={self.split!r} (ratio={self.split_ratio}), "
                  f"source indices={len(common)}, items={len(self.samples)}")

    # ------------------------------------------------------------------ #
    # indexing
    # ------------------------------------------------------------------ #
    def _make_split(self, common: Sequence[int]) -> List[int]:
        common = list(common)
        if self.split == "all":
            return common

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(common))
        n_train = int(round(len(common) * self.split_ratio))
        n_train = max(1, min(n_train, len(common) - 1)) if len(common) > 1 else len(common)

        if self.split == "train":
            chosen = perm[:n_train]
        else:
            chosen = perm[n_train:]
        return sorted(common[i] for i in chosen)

    def _build_samples(self) -> List[Tuple[int, ...]]:
        """One entry per emitted sequence, holding the source frame indices."""
        if self.sequence_mode == "concat_pairs":
            k = self.pairs_per_sample
            stride = max(1, self.stride)
            windows: List[Tuple[int, ...]] = []
            # drop-strategy: only keep complete windows (no padding)
            for s in range(0, len(self.indices) - k + 1, stride):
                windows.append(tuple(self.indices[s : s + k]))
            if not windows:
                raise ValueError(
                    f"concat_pairs produced no complete window: {len(self.indices)} "
                    f"indices, pairs_per_sample={k}. Reduce pairs_per_sample."
                )
            return windows
        return [(i,) for i in self.indices]

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    # sequence construction
    # ------------------------------------------------------------------ #
    def _genuine_frames(self, index: int) -> List[Image.Image]:
        """Load the genuine, temporally ordered frames for one source index."""
        frames: List[Image.Image] = []
        if self.sequence_mode == "ab":
            order = ("frameA", "frameB")
        elif self.sequence_mode == "acb" and self.use_frame_c and self._has_frame_c:
            order = ("frameA", "frameC", "frameB")
        else:
            order = ("frameA", "frameB")

        a_map = {"frameA": self._paths_a, "frameB": self._paths_b, "frameC": self._paths_c}
        for name in order:
            path = a_map[name].get(index)
            if path is None:
                path = self._paths_b[index]  # last-resort fallback
            frames.append(Image.open(path).convert("RGB"))
        return frames

    def _concat_pairs_frames(self, window: Sequence[int]) -> List[Image.Image]:
        frames: List[Image.Image] = []
        for idx in window:
            frames.append(Image.open(self._paths_a[idx]).convert("RGB"))
            frames.append(Image.open(self._paths_b[idx]).convert("RGB"))
        return frames

    def _fit_length(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Fit the genuine frame list to ``self.seq_len`` frames."""
        n = len(frames)
        T = self.seq_len
        if n == 0:
            raise RuntimeError("empty frame list")

        if n == T:
            return frames

        if n > T:
            # Evenly subsample, always keeping the first and last genuine frame.
            pos = np.linspace(0, n - 1, T)
            return [frames[int(round(p))] for p in pos]

        # n < T -> need to extend
        if self.frame_fill == "repeat_last":
            return frames + [frames[-1]] * (T - n)

        if self.frame_fill == "reflect":
            # Triangle wave over period 2(n-1): 0,1,..,n-1,n-2,..,1,0,1,...
            if n == 1:
                return frames * T
            period = 2 * (n - 1)
            out = []
            for t in range(T):
                m = t % period
                j = m if m <= n - 1 else period - m
                out.append(frames[j])
            return out

        # frame_fill == "interpolate" with a single genuine frame: nothing to blend.
        return frames * T

    def _load_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(f) for f in frames], dim=0)

    def _interpolate_to_len(self, frames: List[Image.Image]) -> torch.Tensor:
        """Pixel-space linear interpolation to ``seq_len`` (documented caveat)."""
        if len(frames) < 2:
            return self._load_tensor(frames * self.seq_len)
        key = self._load_tensor(frames).float()  # (n, C, H, W)
        n = key.shape[0]
        T = self.seq_len
        pos = np.linspace(0, n - 1, T)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        w = torch.from_numpy((pos - lo).astype(np.float32)).view(-1, 1, 1, 1)
        out = key[torch.from_numpy(lo)] * (1 - w) + key[torch.from_numpy(hi)] * w
        return out

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.sequence_mode == "concat_pairs":
            frames = self._concat_pairs_frames(self.samples[index])
            if len(frames) > self.seq_len:
                pos = np.linspace(0, len(frames) - 1, self.seq_len)
                frames = [frames[int(round(p))] for p in pos]
            seq = self._load_tensor(frames)
        else:
            idx = self.samples[index][0]
            frames = self._genuine_frames(idx)
            if len(frames) < self.seq_len and self.frame_fill == "interpolate":
                seq = self._interpolate_to_len(frames)
            else:
                seq = self._load_tensor(self._fit_length(frames))

        seq = seq.float()

        if self.add_noise and self.noise_std > 0:
            seq = seq + torch.randn_like(seq) * self.noise_std
            seq = seq.clamp(0.0, 1.0)

        return {"images": seq, "index": index}


# ---------------------------------------------------------------------- #
# diagnostics: the verification checklist from the migration document
# ---------------------------------------------------------------------- #
def measure_temporal_continuity(
    root: str,
    num_samples: int = 200,
    transform: Optional[Callable] = None,
) -> Dict[str, float]:
    """Quantify whether consecutive FD4MM indices are temporally continuous.

    This is step 1/2 of the migration document's verification checklist. It
    compares within-sample frame distances (genuine motion) against the
    index-boundary distance ``||B_i - A_(i+1)||`` that plan 1 relies on.

    Returns a dict of mean L2 distances. A dataset is temporally continuous
    only if ``boundary_BA_next`` is comparable to (not much larger than)
    ``within_AB``.
    """
    if transform is None:
        transform = transforms.ToTensor()
    pa, pb, pc = (
        _list_frames(os.path.join(root, "frameA")),
        _list_frames(os.path.join(root, "frameB")),
        _list_frames(os.path.join(root, "frameC")),
    )
    idxs = sorted(set(pa) & set(pb))
    if len(idxs) < 2:
        raise FileNotFoundError(f"need at least 2 frameA/frameB pairs under {root!r}")

    def load(m: Dict[int, str], i: int) -> torch.Tensor:
        return transform(Image.open(m[i]).convert("RGB")).float()

    def l2(x: torch.Tensor, y: torch.Tensor) -> float:
        return float((x - y).pow(2).mean().sqrt())

    acc: Dict[str, List[float]] = {k: [] for k in
                                   ("within_AB", "within_AC", "within_CB",
                                    "boundary_BA_next", "boundary_AA_next")}
    n = min(num_samples, len(idxs) - 1)
    for k in range(n):
        i, j = idxs[k], idxs[k + 1]
        A, B = load(pa, i), load(pb, i)
        A2 = load(pa, j)
        acc["within_AB"].append(l2(A, B))
        acc["boundary_BA_next"].append(l2(B, A2))
        acc["boundary_AA_next"].append(l2(A, A2))
        if i in pc:
            C = load(pc, i)
            acc["within_AC"].append(l2(A, C))
            acc["within_CB"].append(l2(C, B))

    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    out["_num_samples"] = float(n)
    return out


def visualize_sequence(
    seq: torch.Tensor,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
) -> None:
    """Save/plot a ``(T, C, H, W)`` observation sequence (checklist step 2)."""
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu()
    T = seq.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(2.2 * T, 2.6))
    if T == 1:
        axes = [axes]
    for t in range(T):
        img = seq[t].permute(1, 2, 0).numpy()
        if img.min() < -0.01:  # stored in [-1, 1]
            img = (img + 1.0) / 2.0
        axes[t].imshow(np.clip(img, 0, 1))
        axes[t].set_title(labels[t] if labels and t < len(labels) else f"t={t}", fontsize=8)
        axes[t].axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:  # pragma: no cover
        plt.show()


# ---------------------------------------------------------------------- #
# CLI: the migration document's pre-training verification checklist
# ---------------------------------------------------------------------- #
def _main() -> None:  # pragma: no cover
    """CLI: run the migration document's pre-training verification checklist."""
    import argparse

    ap = argparse.ArgumentParser(description="FD4MM -> DiLA data verification")
    ap.add_argument("--root", default="./data/train", help="FD4MM data root")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--save", default="fd4mm_sequence_check.png")
    args = ap.parse_args()

    print("=" * 72)
    print("STEP 1/2: temporal continuity of FD4MM indices")
    print("=" * 72)
    m = measure_temporal_continuity(args.root, num_samples=args.num_samples)
    for k, v in m.items():
        if not k.startswith("_"):
            print(f"  {k:>18s} = {v:9.4f}   (L2 in the transform's value range)")
    within = m.get("within_AB", float("nan"))
    boundary = m.get("boundary_BA_next", float("nan"))
    print()
    if boundary > 3 * within:
        print(f"  VERDICT: NOT temporally continuous "
              f"(boundary/within = {boundary / within:.1f}x).")
        print("  Plan 1 (concatenating A/B pairs across indices) is INVALID.")
    else:
        print(f"  VERDICT: continuity plausible "
              f"(boundary/within = {boundary / within:.1f}x).")

    print()
    print("=" * 72)
    print("STEP 3: value range of the emitted tensor")
    print("=" * 72)
    ds = FD4MMDataset(root=args.root, seq_len=args.seq_len, split="all", verbose=True)
    item = ds[0]["images"]
    print(f"  shape={tuple(item.shape)} dtype={item.dtype} "
          f"min={item.min():.4f} max={item.max():.4f}")

    print()
    print("=" * 72)
    print("STEP 2: visual check ->", args.save)
    print("=" * 72)
    visualize_sequence(
        item, save_path=args.save,
        title=f"FD4MM {ds.sequence_mode} T={args.seq_len} ({ds.frame_fill})",
    )
    print("  wrote", args.save)


if __name__ == "__main__":  # pragma: no cover
    _main()
