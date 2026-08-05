"""Compute target-level ROC points from DGNet ``predict_map`` MAT files."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage import measure


class ROCMetric:
    """Accumulate pixel-level FPR and target-level TPR over score thresholds."""

    def __init__(self, bins=100):
        if bins < 1:
            raise ValueError("bins must be positive")
        self.thresholds = np.linspace(0.0, 1.0, bins + 1)
        self.false_positive_pixels = np.zeros(bins + 1, dtype=np.float64)
        self.detected_targets = np.zeros(bins + 1, dtype=np.float64)
        self.background_pixels = 0
        self.targets = 0

    def update(self, prediction, mask):
        prediction = np.asarray(prediction, dtype=np.float32)
        prediction = np.clip(prediction, 0.0, 1.0)
        mask = np.asarray(mask) > 0
        components = measure.label(mask, connectivity=2)
        regions = measure.regionprops(components)
        background = ~mask

        self.background_pixels += int(background.sum())
        self.targets += len(regions)
        for index, threshold in enumerate(self.thresholds):
            predicted = prediction >= threshold
            self.false_positive_pixels[index] += np.logical_and(
                background, predicted
            ).sum()
            for region in regions:
                coordinates = region.coords
                if predicted[coordinates[:, 0], coordinates[:, 1]].any():
                    self.detected_targets[index] += 1

    def get(self):
        fpr = self.false_positive_pixels / max(self.background_pixels, 1)
        tpr = self.detected_targets / max(self.targets, 1)
        return fpr, tpr


def _load_prediction(path):
    content = loadmat(path)
    if "predict_map" not in content:
        raise KeyError(f"MAT file does not contain 'predict_map': {path}")
    prediction = np.squeeze(content["predict_map"])
    if prediction.ndim != 2:
        raise ValueError(
            f"Expected a 2-D predict_map in {path}, got shape {prediction.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError(f"predict_map contains NaN or Inf values: {path}")
    return prediction


def _find_mask(mask_dir, image_id):
    for suffix in (".png", ".bmp", ".jpg"):
        path = mask_dir / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No ground-truth mask found for {image_id}")


def evaluate_roc(prediction_dir, mask_dir, bins=100):
    prediction_dir = Path(prediction_dir).expanduser()
    mask_dir = Path(mask_dir).expanduser()
    prediction_files = sorted(prediction_dir.glob("*.mat"))
    if not prediction_files:
        raise FileNotFoundError(f"No .mat prediction maps found in {prediction_dir}")

    metric = ROCMetric(bins=bins)
    for prediction_path in prediction_files:
        prediction = _load_prediction(prediction_path)
        mask_path = _find_mask(mask_dir, prediction_path.stem)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if prediction.shape != mask.shape:
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_image = mask_image.resize(
                (prediction.shape[1], prediction.shape[0]),
                Image.Resampling.NEAREST,
            )
            mask = np.asarray(mask_image) > 0
        metric.update(prediction, mask)

    fpr, tpr = metric.get()
    # Thresholds run from low to high, so reverse both axes for integration.
    auc = float(np.trapz(tpr[::-1], fpr[::-1]))
    return metric.thresholds, fpr, tpr, auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--output", default="./roc_metrics.npz")
    args = parser.parse_args()

    thresholds, fpr, tpr, auc = evaluate_roc(
        args.prediction_dir, args.mask_dir, args.bins
    )
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        thresholds=thresholds,
        fpr=fpr,
        tpr=tpr,
        auc=np.asarray(auc),
    )
    print(f"ROC AUC: {auc:.6f}")
    print(f"Saved ROC data to {output_path}")


if __name__ == "__main__":
    main()
