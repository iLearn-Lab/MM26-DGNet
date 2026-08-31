from pathlib import Path

import numpy as np
import torch
from PIL import Image


DATASET_NORMALIZATION = {
    "NUAA-SIRST": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "SIRST": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "NUDT-SIRST": {"mean": 107.80905151367188, "std": 33.02274703979492},
    "IRSTD-1K": {"mean": 87.4661865234375, "std": 39.71953201293945},
    "NUDT-SIRST-Sea": {"mean": 43.62403869628906, "std": 18.91838264465332},
    "SIRST4": {"mean": 62.10432052612305, "std": 23.96998405456543},
    "IRDST-real": {"mean": 101.54053497314453, "std": 56.49856185913086},
}


class TrainingObjective:
    """Compose the epoch-dependent objective used by DGNet training."""

    _AUXILIARY_START_EPOCH = 5
    _AUXILIARY_HALF_WEIGHT_EPOCH = 300
    _AUXILIARY_QUARTER_WEIGHT_EPOCH = 400
    _AUXILIARY_STOP_EPOCH = 500

    def __init__(self, primary_loss, auxiliary_loss):
        self.primary_loss = primary_loss
        self.auxiliary_loss = auxiliary_loss

    @classmethod
    def auxiliary_weight(cls, epoch):
        if epoch < cls._AUXILIARY_START_EPOCH:
            return 0.0
        if epoch < cls._AUXILIARY_HALF_WEIGHT_EPOCH:
            return 1.0
        if epoch < cls._AUXILIARY_QUARTER_WEIGHT_EPOCH:
            return 0.5
        if epoch < cls._AUXILIARY_STOP_EPOCH:
            return 0.25
        return 0.0

    def __call__(self, epoch, prediction, labels, data):
        loss = self.primary_loss(prediction, labels)
        auxiliary_weight = self.auxiliary_weight(epoch)
        if auxiliary_weight > 0:
            loss = loss + auxiliary_weight * self.auxiliary_loss(
                prediction, labels, data
            )
        return loss


def normalize_image(image, config):
    return (image - config['mean']) / config['std']


def _find_image(image_dir, image_id):
    for suffix in (".png", ".jpg", ".bmp"):
        path = image_dir / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image found for '{image_id}' in {image_dir}")


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name in DATASET_NORMALIZATION:
        return DATASET_NORMALIZATION[dataset_name].copy()

    dataset_dir = Path(dataset_dir)
    index_dir = dataset_dir / "img_idx"
    image_ids = []
    for split in ("train", "test"):
        index_path = index_dir / f"{split}_{dataset_name}.txt"
        if index_path.is_file():
            image_ids.extend(
                line.strip()
                for line in index_path.read_text().splitlines()
                if line.strip()
            )
    if not image_ids:
        raise FileNotFoundError(
            f"No train/test index files found for dataset '{dataset_name}' in {index_dir}"
        )

    means, stds = [], []
    for image_id in image_ids:
        image = Image.open(_find_image(dataset_dir / "images", image_id)).convert("I")
        image = np.asarray(image, dtype=np.float32)
        means.append(image.mean())
        stds.append(image.std())

    config = {"mean": float(np.mean(means)), "std": float(np.mean(stds))}
    print(f"Computed normalization for {dataset_name}: {config}")
    return config


def get_optimizer(
    network, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings
):
    if optimizer_name != "Adam":
        raise ValueError("This release supports the Adam optimizer used in the paper.")
    optimizer = torch.optim.Adam(network.parameters(), lr=optimizer_settings["lr"])

    if scheduler_name != "MultiStepLR":
        raise ValueError(
            "This release supports the MultiStepLR schedule used in the paper."
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=scheduler_settings["step"],
        gamma=scheduler_settings["gamma"],
    )
    return optimizer, scheduler
