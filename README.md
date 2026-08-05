<div align="center">

<h2 align="center">
  <b>DGNet: Dual-knowledge Guided Network for Infrared Small Target Detection</b>
</h2>

PyTorch implementation for the ACM Multimedia 2026 submission.

<div align="center">
  <a href="./paper/2026_MM_DGNet_Submit.pdf">
    <img src="https://img.shields.io/badge/📄%20Paper-PDF-blue?style=flat-square" alt="Paper">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/ACM%20MM-2026-purple?style=flat-square" alt="ACM MM 2026">
  </a>
  <a href="./weights">
    <img src="https://img.shields.io/badge/🏆%20Models-Checkpoints-yellow?style=flat-square" alt="Model checkpoints">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/💻%20Code-Repository-lightgrey?style=flat-square" alt="Code repository">
  </a>
</div>

</div>

## 📢 Updates

- 💻 **[08/2026]** Training, inference, and ROC evaluation code is available.
- 🏆 **[08/2026]** DGNet checkpoints for all three datasets are included.
- 📄 **[08/2026]** The anonymous ACM Multimedia 2026 submission is included as a local PDF.

## 📖 Introduction

InfRared Small Target Detection (IRSTD) aims to segment weak and tiny targets from complex infrared backgrounds. Existing text-guided approaches often use one image-specific description for both the target and background. This can entangle two different objectives—target enhancement and background suppression—and can require an external vision-language model during inference.

DGNet addresses these limitations with multiple generalizable texts and two complementary forms of knowledge:

- **Prior knowledge:** the Prior-knowledge Wavelet Modulation (**PWM**) module uses separate fixed descriptions for large, smooth backgrounds and small, sparse targets. Background-Knowledge Guided Modulation (**B-KGM**) suppresses low-frequency background features, while Target-Knowledge Guided Modulation (**T-KGM**) enhances high-frequency target features.
- **Consensus knowledge:** the Consensus-knowledge Directional Alignment (**CDA**) loss constructs a shared optimization direction from a cluttered source state to an ideal target-enhanced state in the frozen CLIP embedding space.

DGNet uses a four-stage encoder-decoder with PWM modules on its skip connections. The training objective follows Eq. (12) of the paper:

```text
L = L_CDA + L_IoU
```

The fixed PWM prompt features are precomputed. Therefore, neither the CLIP text encoder nor the CLIP image encoder is executed during inference.

This repository provides:

- 💻 DGNet training and inference code
- 🏆 Released checkpoints for three public datasets
- 📈 IoU, Pd, Fa, ROC, and AUC evaluation
- 🧠 CDA Loss and PWM text-feature preparation code

## 🧠 Method / Framework

<p align="center">
  <img src="./Figs/Overview.jpeg" alt="Overall framework of DGNet" width="100%">
</p>

<p align="center">
  <b>Figure 1. Overall architecture of DGNet.</b>
</p>

<p align="center">
  <img src="./Figs/CDA-Loss.jpeg" alt="Consensus-knowledge Directional Alignment Loss" width="100%">
</p>

<p align="center">
  <b>Figure 2. Consensus-knowledge Directional Alignment Loss.</b>
</p>

### Code-to-paper naming

| Paper component | Python class | Source file |
| --- | --- | --- |
| Dual-knowledge Guided Network (DGNet) | `DGNet` | `model/dgnet/dgnet.py` |
| Prior-knowledge Wavelet Modulation (PWM) | `PriorKnowledgeWaveletModulation` | `model/dgnet/pwm.py` |
| Background-Knowledge Guided Modulation (B-KGM) | `BackgroundKnowledgeGuidedModulation` | `model/dgnet/pwm.py` |
| Target-Knowledge Guided Modulation (T-KGM) | `TargetKnowledgeGuidedModulation` | `model/dgnet/pwm.py` |
| Consensus-knowledge Directional Alignment Loss | `ConsensusKnowledgeDirectionalAlignmentLoss` | `cda_loss.py` |
| IoU Loss | `IoULoss` | `losses.py` |

The internal parameter attribute names are retained so that the released checkpoints remain strictly compatible with the renamed public-facing files and classes.

> **Implementation note:** The figures and equations in the paper describe the main design at a conceptual level and omit some low-level implementation details for readability. The released source code and checkpoints are therefore the authoritative implementation. Tensor transformations, feature-routing operations, normalization behavior, module ordering, and checkpoint-facing parameter names in the working code should not be simplified or reconstructed solely from the paper diagrams.

## 📁 Project Structure

```text
DGNet-MM26/
├── Figs/
│   ├── Overview.jpeg
│   └── CDA-Loss.jpeg
├── model/
│   ├── dgnet/
│   │   ├── dgnet.py                # DGNet encoder-decoder
│   │   ├── pwm.py                  # PWM, B-KGM, and T-KGM
│   │   └── wavelet.py              # DWT and inverse DWT
│   └── wrapper.py                  # DGNet and IoU-loss wrapper
├── paper/
│   └── 2026_MM_DGNet_Submit.pdf
├── weights/
│   ├── best_IRSTD-1K.pth.tar
│   ├── best_SIRST.pth.tar
│   ├── best_NUDT-SIRST.pth.tar
│   └── dgnet_text_features.pth
├── cda_loss.py                     # CDA Loss
├── dataset.py                      # Dataset loader and augmentation
├── losses.py                       # IoU Loss
├── metrics.py                      # IoU, Pd, and Fa metrics
├── prepare_text_features.py        # Fixed PWM feature exporter
├── roc.py                          # ROC and AUC evaluation
├── train.py                        # Training entry point
├── test.py                         # Evaluation entry point
├── requirements.txt
├── README.md
└── LICENSE
```

## ⚙️ Installation

### 📥 1. Obtain the Repository

The public repository URL will be added after release. From the extracted source package, enter the project directory:

```bash
cd DGNet-MM26
```

### 🧪 2. Create the Environment

The code was validated in the CodeV environment with Python 3.10.20, PyTorch 2.8.0, torchvision 0.23.0, and CUDA 12.8.

```bash
conda create -n dgnet python=3.10 -y
conda activate dgnet

# This command reproduces the PyTorch build used by the CodeV environment.
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

A CUDA-capable GPU is required by the provided training and evaluation entry points.

### 📦 3. Install OpenAI CLIP for CDA Loss

CDA Loss uses the original [OpenAI CLIP](https://github.com/openai/CLIP) implementation during training:

```bash
git clone https://github.com/openai/CLIP.git
cd CLIP
python setup.py install
cd ..
```

By default, `clip.load("ViT-B/32")` downloads and caches the OpenAI ViT-B/32 checkpoint. On an offline machine, pass a local checkpoint using `--clip_model /path/to/ViT-B-32.pt`.

OpenAI CLIP is required for training because CDA Loss uses its frozen image and text encoders. It is not required for evaluation.

### 🤗 4. Prepare the Fixed PWM Text Features

The included `weights/dgnet_text_features.pth` contains the two fixed 512-dimensional CLIP text features used by PWM. Evaluation loads this file directly.

To regenerate it, download the [ModelScope CLIP ViT-B/32 mirror](https://modelscope.cn/models/openai-mirror/clip-vit-base-patch32):

```bash
git lfs install
git clone https://www.modelscope.cn/openai-mirror/clip-vit-base-patch32.git

python prepare_text_features.py \
  --model_path ./clip-vit-base-patch32 \
  --output ./weights/dgnet_text_features.pth
```

The prior prompts defined in the paper are:

```text
an infrared image occupied by large and smooth background regions
an infrared image containing small and sparse thermal targets
```

## 🗂️ Datasets

All experiments in the paper use three public infrared small-target datasets. IRSTD-1K and NUAA-SIRST are divided into training and testing sets at an approximately 4:1 ratio. NUDT-SIRST is divided approximately equally.

| Dataset | Description | Images | Train / test | Source |
| --- | --- | ---: | ---: | --- |
| IRSTD-1K | Real infrared scenes with diverse targets and complex backgrounds | 1,001 | 800 / 201 | [ISNet repository](https://github.com/RuiZhang97/ISNet) |
| NUAA-SIRST | Real single-frame infrared images; also referred to as SIRST | 427 | 341 / 86 | [SIRST repository](https://github.com/YimianDai/sirst) |
| NUDT-SIRST | Synthesized 256×256 infrared images with varied target characteristics | 1,327 | 663 / 664 | [DNANet repository](https://github.com/YeRen123455/Infrared-Small-Target-Detection) |

The datasets are not redistributed in this repository. Please download them from their public sources and follow their original licenses and citation requirements.

Arrange the files as follows:

```text
datasets/
├── IRSTD-1K/
│   ├── images/
│   ├── masks/
│   └── img_idx/
│       ├── train_IRSTD-1K.txt
│       ├── test_IRSTD-1K.txt
│       └── enhance_IRSTD-1K.txt     # Optional augmentation image index
├── NUAA-SIRST/
│   ├── images/
│   ├── masks/
│   └── img_idx/
│       ├── train_NUAA-SIRST.txt
│       ├── test_NUAA-SIRST.txt
│       └── enhance_NUAA-SIRST.txt   # Optional augmentation image index
└── NUDT-SIRST/
    ├── images/
    ├── masks/
    └── img_idx/
        ├── train_NUDT-SIRST.txt
        ├── test_NUDT-SIRST.txt
        └── enhance_NUDT-SIRST.txt   # Optional augmentation image index
```

Each index file contains one image identifier per line without an extension. Images and masks may use PNG, BMP, or JPG and must have matching identifiers.

The optional `enhance_<dataset>.txt` contains image identifiers selected from the corresponding training split for frequency-domain augmentation. For a training sample whose mean intensity is no greater than the dataset mean, one entry is sampled from this list and its Fourier amplitude is combined with the source image phase. Other samples retain the original image as their augmented view. When migrating an existing dataset, copy the entries previously stored in `high_<dataset>.txt` into `enhance_<dataset>.txt`; `low_<dataset>.txt` is no longer used. If the enhancement index is absent or empty, the augmented view equals the original image.

The paper refers to the 427-image dataset as SIRST in its dataset description and as NUAA-SIRST in the quantitative comparison. The code accepts both names. If `--testset SIRST` is used and `datasets/SIRST/` is absent, `datasets/NUAA-SIRST/` is selected automatically.

## 🏆 Released Checkpoints / Models

The following results are reported in Table 1 of the paper. The corresponding checkpoints are included in `weights/`.

| Dataset | IoU (%) ↑ | Pd (%) ↑ | Fa (×10⁻⁶) ↓ | Checkpoint |
| :---: | ---: | ---: | ---: | :---: |
| IRSTD-1K | 72.72 | 93.88 | 4.25 | [`best_IRSTD-1K.pth.tar`](./weights/best_IRSTD-1K.pth.tar) |
| NUAA-SIRST | 82.68 | 100.00 | 1.24 | [`best_SIRST.pth.tar`](./weights/best_SIRST.pth.tar) |
| NUDT-SIRST | 95.78 | 99.37 | 1.19 | [`best_NUDT-SIRST.pth.tar`](./weights/best_NUDT-SIRST.pth.tar) |

At 256×256 resolution, the paper reports **5.34M** parameters, **8.06G** FLOPs, and **75.61 FPS** for DGNet.

## 🚀 Usage

### ✅ Select the Correct Entry Point

| Goal | Entry point | Model initialization | OpenAI CLIP required? |
| --- | --- | --- | :---: |
| Retrain DGNet from scratch | `train.py` | Random initialization | Yes |
| Test the released checkpoints | `test.py` | A checkpoint from `weights/` | No |
| Export MAT files for ROC | `test.py --save_mat` | A checkpoint from `weights/` | No |

Do not pass `--pretrained` or `--resume` when training from scratch. Evaluation should normally use `test.py`; it selects test mode automatically and does not initialize CDA Loss.

### 🏋️ A. Retrain DGNet from Scratch

The default configuration follows Section 4.2 of the paper:

- 600 training epochs with batch size 16
- Adam optimizer with an initial learning rate of `5e-4`
- Learning rate reduced by 10% at epochs 300 and 450 (multiplied by 0.9 at each milestone)
- Inputs resized or cropped to 256×256
- CLIP ViT-B/32 with ratio slider `r=0.8`
- IoU Loss throughout training; CDA Loss is additionally enabled for internal epoch indices 5 through 300 (inclusive)

Before training, install OpenAI CLIP, prepare the datasets, and ensure that `weights/dgnet_text_features.pth` is present. The following command starts a new IRSTD-1K training run with randomly initialized DGNet parameters:

```bash
python train.py \
  --dataset_dir ./datasets \
  --trainset IRSTD-1K \
  --testset IRSTD-1K \
  --batch_size 16 \
  --epochs 600 \
  --seed 42 \
  --gpu_ids 0
```

To train separately on the other two datasets:

```bash
# NUAA-SIRST
python train.py \
  --dataset_dir ./datasets \
  --trainset NUAA-SIRST \
  --testset NUAA-SIRST \
  --batch_size 16 \
  --epochs 600 \
  --seed 42 \
  --gpu_ids 0

# NUDT-SIRST
python train.py \
  --dataset_dir ./datasets \
  --trainset NUDT-SIRST \
  --testset NUDT-SIRST \
  --batch_size 16 \
  --epochs 600 \
  --seed 42 \
  --gpu_ids 0
```

For multiple GPUs, pass a comma-separated list such as `--gpu_ids 0,1`. The default seed is 42; keeping the same seed and software environment makes data shuffling and augmentation behavior reproducible.

Checkpoints and evaluation logs are written to `weights/<experiment>/`, while TensorBoard events are written to `tf-logs/<experiment>/`. At every validation epoch, `log_on_<dataset>.txt` records the current mIoU and the best mIoU reached so far. Whenever mIoU improves, `best_mIoU_on_<dataset>.pth.tar` is overwritten and the improvement is appended to `best_mIoU_log_on_<dataset>.txt`. The current and best mIoU curves are also available in TensorBoard. Use `--pretrained /path/to/checkpoint` to initialize DGNet parameters or `--resume /path/to/checkpoint` to continue a training run.

- `--pretrained` loads model parameters but starts a new optimizer, scheduler, and training run.
- `--resume` restores model parameters, best mIoU, and, for checkpoints produced by this code, optimizer and scheduler states.

### 🔍 B. Test the Released Checkpoints

OpenAI CLIP and the downloaded Transformers model are not needed for inference. Only the included fixed text-feature file is loaded.

To evaluate all three released checkpoints in one command without saving binary masks:

```bash
python test.py \
  --dataset_dir ./datasets \
  --testset IRSTD-1K/NUAA-SIRST/NUDT-SIRST \
  --weights_dir ./weights \
  --no_save_output \
  --gpu_ids 0
```

Alternatively, evaluate one checkpoint explicitly:

```bash
# IRSTD-1K
python test.py \
  --dataset_dir ./datasets \
  --testset IRSTD-1K \
  --checkpoint ./weights/best_IRSTD-1K.pth.tar \
  --gpu_ids 0

# NUAA-SIRST
python test.py \
  --dataset_dir ./datasets \
  --testset NUAA-SIRST \
  --checkpoint ./weights/best_SIRST.pth.tar \
  --gpu_ids 0

# NUDT-SIRST
python test.py \
  --dataset_dir ./datasets \
  --testset NUDT-SIRST \
  --checkpoint ./weights/best_NUDT-SIRST.pth.tar \
  --gpu_ids 0
```

For these three datasets, `--checkpoint` can be omitted because `test.py` automatically maps each dataset to its corresponding file in `weights/`. When an explicit `--checkpoint` is supplied, test only the matching dataset in that command.

Binary prediction masks are saved under `outputs/<dataset>/` by default. Pass `--no_save_output` to calculate metrics without saving masks. Testing only reads the released checkpoint and does not overwrite it.

### 📈 ROC and AUC Evaluation

ROC must be calculated from continuous probabilities rather than thresholded binary masks. Pass `--save_mat` during inference to save the sigmoid output before thresholding:

```bash
python test.py \
  --dataset_dir ./datasets \
  --testset IRSTD-1K \
  --checkpoint ./weights/best_IRSTD-1K.pth.tar \
  --save_mat --no_save_output \
  --gpu_ids 0

python roc.py \
  --prediction_dir ./outputs/IRSTD-1K/mats \
  --mask_dir ./datasets/IRSTD-1K/masks \
  --bins 100 \
  --output ./outputs/IRSTD-1K/roc_metrics.npz
```

Each MAT file contains a two-dimensional `float32` array named `predict_map`. MAT files and ground-truth masks are matched by file stem. With `--bins 100`, `roc.py` evaluates 101 thresholds over `[0, 1]`.

The resulting NPZ file contains:

- `thresholds`: evaluated probability thresholds
- `fpr`: pixel-level false-positive rate
- `tpr`: target-level detection rate
- `auc`: trapezoidal area under the ROC curve

Keep the default `--save_output` behavior if both binary PNG masks and MAT files are needed. Use `--no_save_output` when only ROC inputs are required.

## 📚 Citation

If you find this project useful, please consider citing the paper. The BibTeX entry will be added after publication:

```bibtex

```

## 📄 License

This project is released under the [Apache License 2.0](./LICENSE).

## 🙏 Acknowledgements

This project builds on [OpenAI CLIP](https://github.com/openai/CLIP), [Hugging Face Transformers](https://github.com/huggingface/transformers), PyWavelets, and the public IRSTD-1K, NUAA-SIRST, and NUDT-SIRST datasets.
