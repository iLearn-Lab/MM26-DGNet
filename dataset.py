import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, gaussian_filter, label
from torch.utils.data import Dataset

from utils import get_img_norm_cfg, normalize_image


MINIMUM_COMPONENT_RETENTION = 0.25


def maximum_component_retention(target_region, crop_box):
    """Return the largest crop-retention ratio among target components."""
    target_region = np.asarray(target_region, dtype=bool)
    components, component_count = label(
        target_region,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count == 0:
        return 0.0

    x_start, y_start, x_end, y_end = (int(value) for value in crop_box)
    maximum_retention = 0.0
    for component_index in range(1, component_count + 1):
        component = components == component_index
        component_area = int(np.count_nonzero(component))
        retained_area = int(
            np.count_nonzero(component[y_start:y_end, x_start:x_end])
        )
        maximum_retention = max(
            maximum_retention,
            retained_area / max(component_area, 1),
        )
    return float(maximum_retention)


def target_protected_gaussian_blur(image, mask, sigma, protection_radius=1):
    """Blur the background while keeping targets and nearby pixels unchanged."""
    image_array = np.asarray(image, dtype=np.float32)
    blurred_array = gaussian_filter(image_array, sigma=sigma)

    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = mask_array[:, :, 0]
    mask_threshold = 0.5 if mask_array.size and mask_array.max() <= 1.0 else 127.5
    target_region = mask_array > mask_threshold
    protected_region = binary_dilation(
        target_region,
        structure=np.ones((3, 3), dtype=bool),
        iterations=protection_radius,
    )
    blurred_array[protected_region] = image_array[protected_region]
    return Image.fromarray(blurred_array)


class InfraredSmallTargetDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_name,
        patch_size,
        mode,
        img_norm_cfg=None,
        use_aug=True,
    ):
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.dataset_name = dataset_name
        dataset_root = Path(dataset_dir).expanduser()
        storage_name = dataset_name
        if dataset_name == "SIRST" and not (dataset_root / "SIRST").is_dir():
            legacy_name = "NUAA-SIRST"
            if (dataset_root / legacy_name).is_dir():
                storage_name = legacy_name
        self.storage_name = storage_name
        self.dataset_dir = dataset_root / storage_name
        self.patch_size = patch_size
        self.base_size = 256
        self.use_sync_transform = dataset_name == "IRSTD-1K"
        self.mode = mode
        self.use_aug = use_aug

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.dataset_dir}"
            )

        split_names = ["train"] if mode == "train" else ["test"]
        if mode == "test":
            split_names = ["train", "test"]
        self.files = []
        for split_name in split_names:
            index_path = (
                self.dataset_dir / "img_idx" / f"{split_name}_{self.storage_name}.txt"
            )
            if not index_path.is_file():
                raise FileNotFoundError(f"Missing dataset split file: {index_path}")
            self.files.extend(
                line.strip()
                for line in index_path.read_text().splitlines()
                if line.strip()
            )
        print(f"{len(self.files)} samples from {dataset_name} for {mode}")

        self.img_norm_cfg = (
            get_img_norm_cfg(dataset_name, self.dataset_dir)
            if img_norm_cfg is None
            else img_norm_cfg
        )
        self.transform = Augmentation()


    def _resolve_image(self, folder, image_id):
        for suffix in (".png", ".bmp", ".jpg"):
            path = self.dataset_dir / folder / f"{image_id}{suffix}"
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"No .png, .bmp, or .jpg file for '{image_id}' in "
            f"{self.dataset_dir / folder}"
        )

    def _load_pair(self, image_id):
        image = Image.open(self._resolve_image("images", image_id)).convert("I")
        mask = Image.open(self._resolve_image("masks", image_id))
        return image, mask

    def __getitem__(self, index):
        image_id = self.files[index]
        image, mask = self._load_pair(image_id)

        apply_statistics_augmentation = False
        if self.mode == "train" and self.use_aug:
            is_dark_intensity = (
                np.asarray(image, dtype=np.float32).mean() <= self.img_norm_cfg["mean"]
            )
            apply_statistics_augmentation = bool(is_dark_intensity)

        if self.mode == "train" and self.use_sync_transform:
            image, mask = self._sync_transform(image, mask)
        else:
            image = image.resize((256, 256), Image.Resampling.NEAREST)
            mask = mask.resize((256, 256), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32)
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        if mask_array.ndim > 2:
            mask_array = mask_array[:, :, 0]

        if self.mode == "train":
            if apply_statistics_augmentation:
                fused_array = controlled_background_statistics_augmentation(
                    image_array,
                    mask_array,
                )
            else:
                fused_array = image_array.copy()

            image_stack = np.stack(
                [
                    normalize_image(image_array, self.img_norm_cfg),
                    normalize_image(fused_array, self.img_norm_cfg),
                ],
                axis=0,
            )
            image_stack, mask_array = self.transform(image_stack, mask_array)
            image_tensor = torch.from_numpy(np.ascontiguousarray(image_stack[0:1]))
            fused_tensor = torch.from_numpy(np.ascontiguousarray(image_stack[1:2]))
            mask_tensor = torch.from_numpy(
                np.ascontiguousarray(mask_array[np.newaxis, :])
            )
            return image_tensor, fused_tensor, mask_tensor

        height, width = image_array.shape
        image_array = normalize_image(image_array, self.img_norm_cfg)[np.newaxis, :]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_array))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_array[np.newaxis, :]))
        return image_tensor, mask_tensor, [height, width], image_id

    def _sync_transform(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        crop_size = self.patch_size or self.base_size
        long_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.2))
        width, height = image.size
        if height > width:
            output_height = long_size
            output_width = int(width * long_size / height + 0.5)
            short_size = output_width
        else:
            output_width = long_size
            output_height = int(height * long_size / width + 0.5)
            short_size = output_height

        size = (output_width, output_height)
        image = image.resize(size, Image.Resampling.NEAREST)
        mask = mask.resize(size, Image.Resampling.NEAREST)

        if short_size < crop_size:
            pad_height = max(0, crop_size - output_height)
            pad_width = max(0, crop_size - output_width)
            border = (0, 0, pad_width, pad_height)
            image = ImageOps.expand(image, border=border, fill=0)
            mask = ImageOps.expand(mask, border=border, fill=0)

        width, height = image.size
        x_start = random.randint(0, width - crop_size)
        y_start = random.randint(0, height - crop_size)
        crop_box = (
            x_start,
            y_start,
            x_start + crop_size,
            y_start + crop_size,
        )
        resized_mask_array = np.asarray(mask)
        if resized_mask_array.ndim > 2:
            resized_mask_array = resized_mask_array[:, :, 0]
        threshold = (
            0.5
            if resized_mask_array.size and resized_mask_array.max() <= 1.0
            else 127.5
        )
        target_region = resized_mask_array > threshold
        target_coordinates = np.argwhere(target_region)
        candidate_mask = mask.crop(crop_box)
        retained_component_ratio = maximum_component_retention(
            target_region,
            crop_box,
        )
        if (
            target_coordinates.size
            and retained_component_ratio < MINIMUM_COMPONENT_RETENTION
        ):
            target_y, target_x = target_coordinates[
                random.randrange(len(target_coordinates))
            ]
            x_min = max(0, int(target_x) - crop_size + 1)
            x_max = min(int(target_x), width - crop_size)
            y_min = max(0, int(target_y) - crop_size + 1)
            y_max = min(int(target_y), height - crop_size)
            x_start = random.randint(x_min, x_max)
            y_start = random.randint(y_min, y_max)
            crop_box = (
                x_start,
                y_start,
                x_start + crop_size,
                y_start + crop_size,
            )
            candidate_mask = mask.crop(crop_box)
        image = image.crop(crop_box)
        mask = candidate_mask

        if random.random() < 0.5:
            sigma = random.random()
            image = target_protected_gaussian_blur(
                image,
                mask,
                sigma=sigma,
                protection_radius=1,
            )

        return image, mask

    def __len__(self):
        return len(self.files)


class Augmentation:
    def __call__(self, input_array, target):
        if random.random() < 0.5:
            input_array = input_array[:, ::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input_array = input_array[:, :, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input_array = input_array.transpose(0, 2, 1)
            target = target.transpose(1, 0)
        return input_array, target


def _minimum_local_contrast_strength(
    source_image,
    augmented_image,
    target_region,
    protection_radius,
    local_contrast_radius,
    minimum_contrast_ratio,
):
    """Return one safe strength that protects every positive-contrast target."""
    components, component_count = label(target_region)
    safe_strength = 1.0
    all_targets = target_region.astype(bool)

    for component_index in range(1, component_count + 1):
        component = components == component_index
        protected_component = binary_dilation(
            component,
            iterations=protection_radius,
        )
        local_region = binary_dilation(
            component,
            iterations=local_contrast_radius,
        )
        local_background = local_region & ~protected_component & ~all_targets
        if not np.any(local_background):
            continue

        target_mean = float(source_image[component].mean())
        source_contrast = target_mean - float(
            source_image[local_background].mean()
        )
        if source_contrast <= 0.0:
            continue

        augmented_contrast = target_mean - float(
            augmented_image[local_background].mean()
        )
        minimum_contrast = minimum_contrast_ratio * source_contrast
        if augmented_contrast >= minimum_contrast:
            continue

        contrast_change = source_contrast - augmented_contrast
        component_strength = (
            source_contrast - minimum_contrast
        ) / max(contrast_change, 1e-6)
        safe_strength = min(safe_strength, component_strength)

    return float(np.clip(safe_strength, 0.0, 1.0))


def controlled_background_statistics_augmentation(
    source_image,
    mask,
    mean_shift_range=(7.0, 21.0),
    std_ratio_range=(0.85, 1.30),
    protection_radius=3,
    local_contrast_radius=12,
    minimum_contrast_ratio=0.8,
    feather_sigma=1.0,
):
    source_image = np.asarray(source_image, dtype=np.float32)

    target_region = np.asarray(mask) > 0.5
    protected_region = binary_dilation(
        target_region,
        iterations=protection_radius,
    )
    background_region = ~protected_region
    if not np.any(background_region):
        return source_image.copy()

    source_background = source_image[background_region]
    source_mean = float(source_background.mean())
    source_std = float(source_background.std())
    if source_std < 1e-6:
        return source_image.copy()

    mean_shift = random.triangular(
        mean_shift_range[0],
        mean_shift_range[1],
        12.0,
    )
    log_std_ratio = random.uniform(
        np.log(std_ratio_range[0]),
        np.log(std_ratio_range[1]),
    )
    std_ratio = float(np.exp(log_std_ratio))
    augmented_image = (
        (source_image - source_mean) * std_ratio
        + source_mean
        + mean_shift
    )
    augmented_image = np.clip(augmented_image, 0.0, 255.0).astype(np.float32)

    safe_strength = _minimum_local_contrast_strength(
        source_image,
        augmented_image,
        target_region,
        protection_radius,
        local_contrast_radius,
        minimum_contrast_ratio,
    )
    augmented_image = source_image + safe_strength * (
        augmented_image - source_image
    )

    protection_weight = gaussian_filter(
        protected_region.astype(np.float32),
        sigma=feather_sigma,
    )
    protection_weight[protected_region] = 1.0
    protection_weight = np.clip(protection_weight, 0.0, 1.0)
    result = (
        protection_weight * source_image
        + (1.0 - protection_weight) * augmented_image
    )
    result[protected_region] = source_image[protected_region]

    return np.clip(result, 0.0, 255.0).astype(np.float32)
