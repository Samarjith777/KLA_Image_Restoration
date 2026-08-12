from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KLARestorationDataset(Dataset):
    """
    KLA paired image-restoration dataset.

    Degraded input:
        128 x 128
        1 channel
        float32

    Ground truth:
        256 x 256
        1 channel
        float32
    """

    def __init__(
        self,
        degraded_dir,
        gt_dir,
        split_file=None,
    ):
        self.degraded_dir = Path(degraded_dir)
        self.gt_dir = Path(gt_dir)

        if not self.degraded_dir.exists():
            raise FileNotFoundError(
                f"Degraded directory not found: {self.degraded_dir}"
            )

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Ground-truth directory not found: {self.gt_dir}"
            )

        # --------------------------------------------------
        # Determine which files to use
        # --------------------------------------------------

        if split_file is not None:

            split_file = Path(split_file)

            if not split_file.exists():
                raise FileNotFoundError(
                    f"Split file not found: {split_file}"
                )

            with open(split_file, "r") as f:
                filenames = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

        else:

            filenames = sorted(
                file.name
                for file in self.degraded_dir.glob("*.npy")
            )

        # --------------------------------------------------
        # Build paired file list
        # --------------------------------------------------

        self.pairs = []

        for filename in filenames:

            degraded_path = (
                self.degraded_dir / filename
            )

            gt_path = (
                self.gt_dir / filename
            )

            if not degraded_path.exists():
                raise FileNotFoundError(
                    f"Degraded file not found: {degraded_path}"
                )

            if not gt_path.exists():
                raise FileNotFoundError(
                    f"Ground-truth file not found: {gt_path}"
                )

            self.pairs.append(
                (degraded_path, gt_path)
            )

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No paired samples found."
            )

        print(
            f"Loaded {len(self.pairs)} paired samples."
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        degraded_path, gt_path = self.pairs[index]

        # --------------------------------------------------
        # Load NumPy arrays
        # --------------------------------------------------

        degraded = np.load(degraded_path)
        ground_truth = np.load(gt_path)

        # --------------------------------------------------
        # Validate dimensions
        # --------------------------------------------------

        if degraded.shape != (128, 128):
            raise ValueError(
                f"Unexpected degraded shape "
                f"{degraded.shape} in {degraded_path.name}"
            )

        if ground_truth.shape != (256, 256):
            raise ValueError(
                f"Unexpected GT shape "
                f"{ground_truth.shape} in {gt_path.name}"
            )

        # --------------------------------------------------
        # Convert to float32
        # --------------------------------------------------

        degraded = degraded.astype(
            np.float32,
            copy=False
        )

        ground_truth = ground_truth.astype(
            np.float32,
            copy=False
        )

        # --------------------------------------------------
        # NumPy → PyTorch
        #
        # [H, W] → [1, H, W]
        # --------------------------------------------------

        degraded = torch.from_numpy(
            degraded
        ).unsqueeze(0)

        ground_truth = torch.from_numpy(
            ground_truth
        ).unsqueeze(0)

        return degraded, ground_truth