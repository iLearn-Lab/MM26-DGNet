import argparse
import random
import time
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.io import savemat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cda_loss import ConsensusKnowledgeDirectionalAlignmentLoss
from dataset import InfraredSmallTargetDataset
from metrics import DetectionMetric, IoUMetric
from model import DGNetModel
from utils import TrainingObjective, get_optimizer


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = {
    "IRSTD-1K": "best_IRSTD-1K.pth.tar",
    "NUDT-SIRST": "best_NUDT-SIRST.pth.tar",
    # The upstream dataset is canonically named SIRST. The local folder name
    # NUAA-SIRST is retained for compatibility with the experiment splits.
    "NUAA-SIRST": "best_SIRST.pth.tar",
    "SIRST": "best_SIRST.pth.tar",
}

def save_checkpoint(state, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)

def load_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)

def load_model_state(model, checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

def load_prior_text_features(path, device):
    """Load the two fixed CLIP text features used by all PWM modules."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Prior text features not found: {path}. Run "
            "'python prepare_text_features.py' as described in README.md."
        )
    features = torch.load(path, map_location="cpu", weights_only=True)
    target = features["target"].float().reshape(1, -1)
    background = features["background"].float().reshape(1, -1)
    if target.shape[1] != 512 or background.shape[1] != 512:
        raise ValueError(
            f"Expected two 512-dimensional CLIP features in {path}, got "
            f"{tuple(target.shape)} and {tuple(background.shape)}"
        )
    return target.to(device), background.to(device)


class DGNetTrainer:
    def __init__(self, options):
        self.opt = options
        self.mode = options.mode
        self._set_seed(options.seed)
        self.device, self.gpu_ids = self._get_device(options.gpu_ids)

        generator = torch.Generator().manual_seed(options.seed)
        self.train_loader = None
        if self.mode == "train":
            train_set = InfraredSmallTargetDataset(
                dataset_dir=options.dataset_dir,
                dataset_name=options.trainset,
                patch_size=options.patch_size,
                mode="train",
                img_norm_cfg=options.img_norm_cfg,
                use_fft_aug=not options.disable_fft_aug,
            )
            self.train_loader = DataLoader(
                train_set,
                batch_size=options.batch_size,
                shuffle=True,
                num_workers=options.num_workers,
                worker_init_fn=self._seed_worker,
                generator=generator,
                pin_memory=True,
            )

        self.val_loaders = {}
        self.best_metric_miou = {}
        for dataset_name in options.testset:
            evaluation_mode = "val"
            if self.mode == "train" and dataset_name != options.trainset:
                evaluation_mode = "test"
            val_set = InfraredSmallTargetDataset(
                dataset_dir=options.dataset_dir,
                dataset_name=dataset_name,
                patch_size=None,
                mode=evaluation_mode,
                img_norm_cfg=options.img_norm_cfg,
                use_fft_aug=False,
            )
            self.val_loaders[dataset_name] = DataLoader(
                val_set,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=options.num_workers,
                worker_init_fn=self._seed_worker,
                generator=generator,
                pin_memory=True,
            )
            self.best_metric_miou[dataset_name] = 0.0

        base_model = DGNetModel(model_name=options.model).to(self.device)
        self.epoch_state = 0
        resume_checkpoint = None
        if options.pretrained:
            load_model_state(base_model, load_checkpoint(options.pretrained))
            print(f"Loaded pretrained model: {options.pretrained}")
        if options.resume:
            resume_checkpoint = load_checkpoint(options.resume)
            load_model_state(base_model, resume_checkpoint)
            self.epoch_state = int(resume_checkpoint.get("epoch", 0))
            print(f"Resuming from epoch {self.epoch_state}: {options.resume}")

        self.model = nn.DataParallel(
            base_model,
            device_ids=self.gpu_ids,
            output_device=self.gpu_ids[0],
        )

        self.target_text, self.background_text = load_prior_text_features(
            options.text_features, self.device
        )

        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.training_objective = None
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.save_name = f"{options.name}-{timestamp}-{options.trainset}"

        if self.mode == "train":
            auxiliary_loss = ConsensusKnowledgeDirectionalAlignmentLoss(
                self.device, ratio=0.8, model_name=options.clip_model
            ).to(self.device)
            self.training_objective = TrainingObjective(
                primary_loss=self.model.module.loss,
                auxiliary_loss=auxiliary_loss,
            )
            optimizer_settings = {"lr": options.learning_rate}
            scheduler_settings = {
                "step": options.lr_steps,
                "gamma": options.lr_gamma,
            }
            self.optimizer, self.scheduler = get_optimizer(
                self.model,
                "Adam",
                "MultiStepLR",
                optimizer_settings,
                scheduler_settings,
            )
            if resume_checkpoint is not None:
                self._restore_best_miou(resume_checkpoint)
                self._restore_training_state(resume_checkpoint)

            self.save_folder = Path(options.weights_dir) / self.save_name
            self.save_folder.mkdir(parents=True, exist_ok=True)
            log_folder = Path(options.log_dir) / self.save_name
            self.writer = SummaryWriter(log_folder)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _seed_worker(self, worker_id):
        # Preserve the original training behavior: DataLoader passes the worker
        # id directly to the former seed callback, so workers use seeds 0..N-1.
        random.seed(worker_id)
        np.random.seed(worker_id)
        torch.manual_seed(worker_id)
        torch.cuda.manual_seed(worker_id)
        torch.cuda.manual_seed_all(worker_id)

    @staticmethod
    def _get_device(gpu_ids_value):
        gpu_ids = [int(item) for item in str(gpu_ids_value).split(",") if item.strip()]
        if not gpu_ids:
            raise ValueError("--gpu_ids must contain at least one GPU id")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but no CUDA device is available")
        if min(gpu_ids) < 0 or max(gpu_ids) >= torch.cuda.device_count():
            raise ValueError(
                f"GPU ids {gpu_ids} are invalid for "
                f"{torch.cuda.device_count()} visible CUDA device(s)"
            )
        torch.cuda.set_device(gpu_ids[0])
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids

    def _text_features(self, batch_size):
        return (
            self.target_text.repeat(batch_size, 1),
            self.background_text.repeat(batch_size, 1),
        )

    def _restore_training_state(self, checkpoint):
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            return

        # Published checkpoints predate optimizer/scheduler serialization.
        completed_epochs = self.epoch_state
        decay_count = sum(step <= completed_epochs for step in self.opt.lr_steps)
        resumed_lr = self.opt.learning_rate * (self.opt.lr_gamma**decay_count)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = resumed_lr
        self.scheduler.last_epoch = completed_epochs - 1
        self.scheduler._last_lr = [resumed_lr] * len(self.optimizer.param_groups)

    def _restore_best_miou(self, checkpoint):
        saved_best = checkpoint.get("best_mIoU")
        if isinstance(saved_best, dict):
            for dataset_name, value in saved_best.items():
                if dataset_name in self.best_metric_miou:
                    self.best_metric_miou[dataset_name] = float(value)
        elif saved_best is not None and len(self.best_metric_miou) == 1:
            dataset_name = next(iter(self.best_metric_miou))
            self.best_metric_miou[dataset_name] = float(saved_best)
        elif "eval_mIoU" in checkpoint and self.opt.trainset in self.best_metric_miou:
            self.best_metric_miou[self.opt.trainset] = float(checkpoint["eval_mIoU"])

        restored = ", ".join(
            f"{name}={value * 100:.4f}"
            for name, value in self.best_metric_miou.items()
        )
        print(f"Restored best mIoU values: {restored}")

    def train_epoch(self, epoch):
        start = timeit.default_timer()
        self.model.train()
        total_loss = 0.0

        for image, augmented_image, mask in self.train_loader:
            data = torch.cat([image, augmented_image], dim=0).to(
                self.device, non_blocking=True
            )
            labels = torch.cat([mask, mask], dim=0).to(self.device, non_blocking=True)
            target_text, background_text = self._text_features(data.shape[0])
            prediction = self.model(data, target_text, background_text)
            loss = self.training_objective(epoch, prediction, labels, data)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().item()

        self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss", average_loss, epoch)
        elapsed = timeit.default_timer() - start
        print(
            f"Train epoch {epoch:03d} on {self.opt.trainset}, "
            f"loss: {average_loss:.4f}, time: {int(elapsed // 60)}m "
            f"{int(elapsed % 60)}s"
        )

    @torch.no_grad()
    def evaluate(self, epoch=None):
        self.model.eval()
        results = {}
        for dataset_name, loader in self.val_loaders.items():
            start = timeit.default_timer()
            miou_metric = IoUMetric()
            detection_metric = DetectionMetric()

            for data, mask, size, _ in loader:
                data = data.to(self.device, non_blocking=True)
                target_text, background_text = self._text_features(data.shape[0])
                prediction = self.model(data, target_text, background_text)
                height, width = int(size[0]), int(size[1])
                prediction = prediction[:, :, :height, :width]
                mask = mask[:, :, :height, :width]
                binary_prediction = prediction > self.opt.threshold
                miou_metric.update(binary_prediction.cpu(), mask)
                detection_metric.update(
                    binary_prediction[0, 0].cpu(), mask[0, 0], (height, width)
                )

            pixel_accuracy, mean_iou = miou_metric.get()
            probability_detection, false_alarm = detection_metric.get()
            metrics = {
                "pixAcc": pixel_accuracy,
                "mIoU": float(mean_iou),
                "Pd": float(probability_detection),
                "Fa": float(false_alarm),
            }
            results[dataset_name] = metrics

            if self.mode == "train":
                self._record_evaluation(dataset_name, epoch, metrics)

            elapsed = timeit.default_timer() - start
            print(
                f"Eval on {dataset_name}: pixAcc={pixel_accuracy * 100:.4f}, "
                f"mIoU={mean_iou * 100:.4f}, "
                f"best_mIoU={self.best_metric_miou[dataset_name] * 100:.4f}, "
                f"Pd={probability_detection * 100:.4f}, "
                f"Fa={false_alarm * 1e6:.4f}e-6, "
                f"time={int(elapsed // 60)}m {int(elapsed % 60)}s"
            )
        return results

    def _checkpoint_state(self, epoch, metrics):
        return {
            "epoch": epoch + 1,
            "state_dict": self.model.module.state_dict(), # 模型权重，约20MB
            "optimizer": self.optimizer.state_dict(), # 可以注释，约40.72MB
            "scheduler": self.scheduler.state_dict(),
            "best_mIoU": dict(self.best_metric_miou),
            "eval_pixAcc": metrics["pixAcc"],
            "eval_mIoU": metrics["mIoU"],
            "eval_Pd": metrics["Pd"],
            "eval_Fa": metrics["Fa"],
        }

    def _record_evaluation(self, dataset_name, epoch, metrics):
        improved = metrics["mIoU"] > self.best_metric_miou[dataset_name]
        if improved:
            self.best_metric_miou[dataset_name] = metrics["mIoU"]
        best_miou = self.best_metric_miou[dataset_name]

        log_line = (
            f"{time.strftime('%Y-%m-%d-%H-%M-%S')} - {epoch:04d} "
            f"pixAcc {metrics['pixAcc'] * 100:.4f} "
            f"mIoU {metrics['mIoU'] * 100:.4f} "
            f"best_mIoU {best_miou * 100:.4f} "
            f"PD {metrics['Pd'] * 100:.4f} "
            f"FA {metrics['Fa'] * 1e6:.4f}\n"
        )
        with (self.save_folder / f"log_on_{dataset_name}.txt").open("a") as log_file:
            log_file.write(log_line)

        for metric_name, value in metrics.items():
            scale = 1e6 if metric_name == "Fa" else 100
            self.writer.add_scalar(
                f"eval/{dataset_name}/{metric_name}", value * scale, epoch
            )
        self.writer.add_scalar(
            f"eval/{dataset_name}/best_mIoU", best_miou * 100, epoch
        )

        if improved:
            checkpoint_path = (
                self.save_folder / f"best_mIoU_on_{dataset_name}.pth.tar"
            )
            save_checkpoint(
                self._checkpoint_state(epoch, metrics),
                checkpoint_path,
            )
            best_log_line = (
                f"{time.strftime('%Y-%m-%d-%H-%M-%S')} - {epoch:04d} "
                f"best_mIoU {best_miou * 100:.4f} "
                f"checkpoint {checkpoint_path.name}\n"
            )
            with (
                self.save_folder / f"best_mIoU_log_on_{dataset_name}.txt"
            ).open("a") as best_log_file:
                best_log_file.write(best_log_line)

    def _checkpoint_for_dataset(self, dataset_name):
        if self.opt.checkpoint:
            return Path(self.opt.checkpoint).expanduser()
        if dataset_name not in DEFAULT_WEIGHTS:
            raise ValueError(
                f"No default checkpoint is registered for {dataset_name}; "
                "pass --checkpoint explicitly."
            )
        return Path(self.opt.weights_dir) / DEFAULT_WEIGHTS[dataset_name]

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        all_results = {}
        for dataset_name, loader in self.val_loaders.items():
            checkpoint_path = self._checkpoint_for_dataset(dataset_name)
            checkpoint = load_checkpoint(checkpoint_path)
            load_model_state(self.model.module, checkpoint)
            print(f"Loaded checkpoint for {dataset_name}: {checkpoint_path}")

            miou_metric = IoUMetric()
            detection_metric = DetectionMetric()
            output_path = Path(self.opt.output_dir) / dataset_name
            mat_output_path = output_path / "mats"
            if self.opt.save_output:
                output_path.mkdir(parents=True, exist_ok=True)
            if self.opt.save_mat:
                mat_output_path.mkdir(parents=True, exist_ok=True)

            for data, mask, size, filenames in loader:
                data = data.to(self.device, non_blocking=True)
                target_text, background_text = self._text_features(data.shape[0])
                prediction = self.model(data, target_text, background_text)
                height, width = int(size[0]), int(size[1])
                prediction = prediction[:, :, :height, :width]
                mask = mask[:, :, :height, :width]
                binary_prediction = prediction > self.opt.threshold
                miou_metric.update(binary_prediction.cpu(), mask)
                detection_metric.update(
                    binary_prediction[0, 0].cpu(), mask[0, 0], (height, width)
                )

                if self.opt.save_output:
                    for batch_index, filename in enumerate(filenames):
                        output = (
                            binary_prediction[batch_index, 0]
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                            * 255
                        )
                        Image.fromarray(output).save(output_path / f"{filename}.png")
                if self.opt.save_mat:
                    for batch_index, filename in enumerate(filenames):
                        predict_map = (
                            prediction[batch_index, 0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        savemat(
                            mat_output_path / f"{filename}.mat",
                            {"predict_map": predict_map},
                            do_compression=True,
                        )

            pixel_accuracy, mean_iou = miou_metric.get()
            probability_detection, false_alarm = detection_metric.get()
            all_results[dataset_name] = {
                "pixAcc": pixel_accuracy,
                "mIoU": float(mean_iou),
                "Pd": float(probability_detection),
                "Fa": float(false_alarm),
            }
            print(
                f"Inference on {dataset_name}: pixAcc={pixel_accuracy * 100:.4f}, "
                f"mIoU={mean_iou * 100:.4f}, "
                f"Pd={probability_detection * 100:.4f}, "
                f"Fa={false_alarm * 1e6:.4f}e-6"
            )
            if self.opt.save_mat:
                print(
                    f"Saved {len(loader.dataset)} ROC prediction maps to "
                    f"{mat_output_path}"
                )
        return all_results

    def close(self):
        if self.writer is not None:
            self.writer.close()


def build_parser(default_mode="train"):
    parser = argparse.ArgumentParser(description="DGNet training and evaluation")
    parser.add_argument("--mode", choices=("train", "test"), default=default_mode)
    parser.add_argument("--model", default="DGNet")
    parser.add_argument("--name", default="DGNet", help="experiment name")
    parser.add_argument("--trainset", default="IRSTD-1K")
    parser.add_argument(
        "--testset",
        default="IRSTD-1K",
        help="one dataset or slash-separated datasets",
    )
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument(
        "--text_features",
        default="./weights/dgnet_text_features.pth",
        help="precomputed fixed CLIP features for the two PWM prompts",
    )
    parser.add_argument(
        "--clip_model",
        default="ViT-B/32",
        help="OpenAI CLIP model name or local .pt path for CDA Loss",
    )
    parser.add_argument("--weights_dir", default="./weights")
    parser.add_argument("--log_dir", default="./tf-logs")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--batch_size", "--batchSize", type=int, default=16)
    parser.add_argument("--patch_size", "--patchSize", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_steps", type=int, nargs="+", default=[300, 450])
    parser.add_argument("--lr_gamma", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ids", default="4")
    parser.add_argument("--test_freq", type=int, default=1)
    parser.add_argument("--img_norm_cfg_mean", type=float, default=None)
    parser.add_argument("--img_norm_cfg_std", type=float, default=None)
    parser.add_argument("--disable_fft_aug", action="store_true")
    parser.add_argument("--save_output", dest="save_output", action="store_true")
    parser.add_argument("--no_save_output", dest="save_output", action="store_false")
    parser.add_argument(
        "--save_mat",
        action="store_true",
        help=(
            "save unthresholded prediction maps as compressed .mat files "
            "for ROC evaluation"
        ),
    )
    parser.set_defaults(save_output=True)
    return parser


def prepare_options(options):
    options.testset = [
        item.strip()
        for item in options.testset.replace(",", "/").split("/")
        if item.strip()
    ]
    if not options.testset:
        raise ValueError("--testset must specify at least one dataset")
    if (options.img_norm_cfg_mean is None) != (options.img_norm_cfg_std is None):
        raise ValueError(
            "--img_norm_cfg_mean and --img_norm_cfg_std must be provided together"
        )
    options.img_norm_cfg = None
    if options.img_norm_cfg_mean is not None:
        options.img_norm_cfg = {
            "mean": options.img_norm_cfg_mean,
            "std": options.img_norm_cfg_std,
        }
    return options


def main():
    options = prepare_options(build_parser().parse_args())
    trainer = DGNetTrainer(options)
    try:
        if options.mode == "test":
            trainer.inference()
            return

        print("========== Training ==========")
        for epoch in range(trainer.epoch_state, options.epochs):
            trainer.train_epoch(epoch)
            if (epoch + 1) % options.test_freq == 0:
                trainer.evaluate(epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
