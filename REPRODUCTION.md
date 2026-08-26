# Reproduction Guide

This repository contains the implementation code for material image
segmentation comparison experiments. Licensed source files from selected
official upstream projects are retained together with the experiment-specific
training, testing, configuration, metric, and plotting code. MatSAM and
Swin-Unet are installed from separate upstream clones.

The research datasets and model weights are not included directly in this
repository. The CPU demo uses three deterministic simulated image-mask pairs.

## 1. Repository Layout

| Path | Content |
| --- | --- |
| `CNN/U-net` | U-Net baseline |
| `CNN/DeepLabV3+` | DeepLabV3+ baseline |
| `matsam` | MatSAM experiment scripts; the upstream runtime is cloned separately into `matsam/upstream` |
| `hqsam` | HQ-SAM upstream runtime plus original, binary LoRA, and multi-class class-token experiment scripts |
| `sam2` | SAM2 upstream runtime plus original, binary LoRA, and multi-class class-token experiment scripts |
| `microsam` | micro-sam upstream runtime plus original, binary LoRA, and multi-class class-token experiment scripts |
| `Swin-Unet` | Swin-Unet experiment scripts; the upstream runtime is cloned separately into `Swin-Unet/upstream` |
| `mmsegmentation` | SegFormer configs based on MMSegmentation |
| `Myutils` | Metric calculation, prediction visualization, and paper figure utilities |
| `WEIGHTS.md` | External weight/checkpoint placement notes |

## 2. Environment

The experiments were run on Ubuntu 22.04.3 LTS with separate environments for
different model families. U-Net and DeepLabV3+ originally reused an existing
environment named `sam` for convenience. The public reproduction environment
is named `cnn` because these baselines only require
`segmentation_models_pytorch` and do not depend on a SAM model. Exact tested
package versions are stored in [`environment_reports`](environment_reports/README.md).

Use the official environment requirements shipped with each upstream project first, then install the extra packages used by the experiment scripts.

### CNN Baselines

```bash
conda create -n cnn python=3.9
conda activate cnn

pip install torch torchvision
pip install segmentation-models-pytorch albumentations opencv-python numpy pandas matplotlib tqdm pillow tifffile openpyxl scikit-image scikit-learn scipy
```

### MatSAM

```bash
conda create -n matsam python=3.9
conda activate matsam
git clone https://github.com/USTB-AI3DVIP/matsam.git matsam/upstream
git -C matsam/upstream checkout cbea7edaada991d88d7dfee656bd7e3dac09863f
pip install -r matsam/upstream/requirements.txt
pip install peft pandas openpyxl tifffile
```

### HQ-SAM

Follow the official requirements in `hqsam/requirements.txt`.

```bash
conda create -n hqsam python=3.9
conda activate hqsam
pip install torch torchvision
pip install -r hqsam/requirements.txt
cd hqsam
pip install -e .
```

### SAM2

Follow `sam2/INSTALL.md`: Linux, Python >= 3.10, PyTorch >= 2.5.1, matching `torchvision`, and a CUDA toolkit matching the PyTorch CUDA version.

```bash
conda create -n sam2 python=3.10
conda activate sam2
pip install torch torchvision
cd sam2
pip install -e ".[notebooks]"
pip install peft segment-anything opencv-python pandas openpyxl tifffile
```

If the SAM2 CUDA extension fails to build on the server, it can be skipped:

```bash
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

### micro-sam

Use `microsam/environment.yaml` and the upstream editable install.

```bash
conda env create -f microsam/environment.yaml -n microsam
conda activate microsam
conda install -c conda-forge vigra
cd microsam
pip install -e .
pip install peft pandas openpyxl
```

`vigra` is distributed through conda-forge and is required by the bundled
`torch-em`/`elf` dependency chain. Install it with conda rather than pip if it
is not already resolved from `environment.yaml`.

### Swin-Unet

```bash
conda create -n swinunet python=3.9
conda activate swinunet
git clone https://github.com/HuCaoFighting/Swin-Unet.git Swin-Unet/upstream
git -C Swin-Unet/upstream checkout f48f623e226e25b6e395c37207915c50aaa9c776
pip install torch torchvision
pip install -r Swin-Unet/upstream/requirements.txt
pip install albumentations pandas openpyxl matplotlib opencv-python
```

### SegFormer / MMSegmentation

Use the official MMSegmentation dependency chain.

```bash
conda create -n segformer python=3.9
conda activate segformer
pip install torch torchvision
cd mmsegmentation
pip install -U openmim
mim install mmengine==0.10.7 mmcv==2.1.0
pip install -e .
```

The tested editable checkout reports MMSegmentation `1.2.2`. `mmcv` must
include its compiled operators; a plain or incompatible installation that
cannot import `mmcv._ext` is not sufficient.

## 3. Dataset Structure

The seven datasets used in the experiments are:

```text
MetalDAM
UHCS
EBC
Super
Aachen-Heerlen
EMPS
Grain
```

The complete research dataset archive is available from Zenodo:

- [Dataset archive](https://doi.org/10.5281/zenodo.22039777)

After downloading the archive, arrange each dataset using the split structure
below.

Each dataset should use this structure:

```text
DATASET_NAME/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Mask files are read as label maps. Binary experiments use foreground/background labels, and multi-class experiments use integer class IDs.

### Command-Line Input Convention

`--dataset-root` always points to one `DATASET_NAME` directory, not to the
parent directory containing all datasets. Training scripts consume the
`train` and `val` splits, while test scripts consume the `test` split. Use the
same filename stem for each image-mask pair whenever possible. The loaders
also support common mask suffixes such as `RGMask`, `_mask`, `_label`, and
`_seg` for compatibility with the original datasets.

The common path arguments are:

| Argument | Meaning |
| --- | --- |
| `--dataset-root` | One dataset directory with the documented split structure |
| `--output-dir` | Directory created for checkpoints, metrics, and predictions |
| `--checkpoint` | Official base weight for original inference, or the trained model weight for CNN/Swin-Unet testing |
| `--finetuned-checkpoint` | LoRA/decoder or class-token checkpoint produced by a SAM-family fine-tuning script |
| `--num-classes` | Total number of label IDs, including background |
| `--device` | PyTorch device such as `cuda:0` or `cpu` |

All experiment test scripts save raw class-index or instance-index masks under
`OUTPUT_DIR/plots/` and export image-level metrics to
`OUTPUT_DIR/metrics_summary.xlsx`. Swin-Unet adds a dataset-name subdirectory
below `OUTPUT_DIR`. The raw masks retain the original image dimensions and are
intended for quantitative analysis; comparison figures can be generated
separately with `Myutils/visualizer.py`.

CNN, Swin-Unet, and SegFormer training use a batch size of `32`. The
SAM-family fine-tuning scripts process one image per optimizer step.

## 4. Model Overview

The model descriptions below follow the manuscript text. If an implementation detail in the current script differs from the manuscript description, use the manuscript description for reporting.

| Model | Description |
| --- | --- |
| U-Net | Uses the U-Net implementation from `segmentation_models_pytorch` with a ResNet-50 encoder. Input images are resized to `512 x 512` before inference, and predicted masks are resized back to the original resolution using nearest-neighbor interpolation. |
| DeepLabV3+ | Uses the DeepLabV3+ implementation from `segmentation_models_pytorch` with a ResNet-50 backbone. Input images are resized to `512 x 512`, and nearest-neighbor interpolation is used to recover the original mask resolution. |
| SegFormer | Uses MMSegmentation's SegFormer implementation. Inference uses the MiT-B0 architecture with input images resized to `512 x 512`. |
| Swin-Unet | Uses the Swin-Unet implementation from HuCaoFighting/Swin-Unet. Input images are resized to `224 x 224` and normalized with ImageNet statistics; outputs are resized back to the original resolution with nearest-neighbor interpolation. |
| SAM2 | Uses `sam2.1_hiera_base_plus` through `SAM2AutomaticMaskGenerator`, with `points_per_side=32`, `points_per_batch=64`, `pred_iou_thresh=0.8`, `stability_score_thresh=0.8`, and `crop_n_layers=0`. |
| HQ-SAM | Uses the HQ-SAM `vit_b` backbone through `SamAutomaticMaskGenerator`, with `points_per_side=32`, `pred_iou_thresh=0.85`, `stability_score_thresh=0.8`, and `crop_n_layers=0`. |
| micro-sam | Uses `vit_b_lm` in automatic instance segmentation mode. Parameters include `min_size=10`, `center_distance_threshold=0.5`, and `boundary_distance_threshold=0.5`. |
| MatSAM | Uses the MatSAM `vit_h` model with a custom `PromptGenerator`; `method_type=1` for grain data and `method_type=2` for metallographic data, with `n_per_side_base=54`, `pred_iou_thresh=0.88`, `stability_score_thresh=0.9`, and `box_nms_thresh=0.7`. |

The SAM-family experiments use two fine-tuning routes. Aachen-Heerlen, EMPS,
and Grain are binary datasets and use image-encoder LoRA plus a fully
trainable mask decoder with point prompts. EBC, Super, MetalDAM, and UHCS use
class tokens for multiclass semantic prediction. In both routes the prompt
encoder remains frozen and LoRA uses `rank=8`, `alpha=16`, and
`dropout=0.05`, targeting the `qkv` and `proj` modules.

### Hyperparameter Summary

| Model | Hyperparameters and values |
| --- | --- |
| U-Net | encoder `ResNet-50`; encoder weights `ImageNet`; input size `512 x 512`; batch size `32`; epochs `500`; optimizer `AdamW`; learning rate `0.0003`; weight decay `0.001`; scheduler `CosineAnnealingLR`, minimum learning rate `0.000001`; binary loss `Dice + BCEWithLogits`; multi-class loss `Dice + cross-entropy`; early-stopping patience `50` |
| DeepLabV3+ | encoder `ResNet-50`; encoder weights `ImageNet`; input size `512 x 512`; batch size `32`; epochs `500`; optimizer `AdamW`; learning rate `0.0003`; weight decay `0.001`; scheduler `CosineAnnealingLR`, minimum learning rate `0.000001`; binary loss `Dice + BCEWithLogits`; multi-class loss `Dice + cross-entropy`; early-stopping patience `50` |
| SegFormer | backbone `MiT-B0`; input size `512 x 512`; training batch size `32`; validation/test batch size `16`; epochs `200`; optimizer `AdamW`; learning rate `0.00006`; Adam betas `(0.9, 0.999)`; weight decay `0.01`; scheduler `5`-epoch linear warm-up followed by polynomial decay |
| Swin-Unet | backbone `Swin-Tiny`; ImageNet pretrained weights; input size `224 x 224`; patch size `4`; window size `7`; batch size `32`; epochs `500`; optimizer `AdamW`; learning rate `0.0001`; weight decay `0.0001`; scheduler `CosineAnnealingLR`, minimum learning rate `0.000001`; loss `0.5 x cross-entropy + 0.5 x Dice`; early-stopping patience `50` |
| MatSAM | model `ViT-H`; training batch size `1`; epochs `200`; optimizer `AdamW`; learning rate `0.0001`; weight decay `0.0001`; multiclass scheduler `CosineAnnealingLR`, minimum learning rate `0.000001`; multiclass loss `cross-entropy + Dice`; binary loss `BCE + Dice + IoU MSE`; early-stopping patience `50`; LoRA rank `8`, alpha `16`, dropout `0.05`, targets `qkv` and `proj`; `method_type=1` for grain data and `method_type=2` for metallographic data; `n_per_side_base=54`; original inference `pred_iou_thresh=0.88`, `stability_score_thresh=0.9`; binary fine-tuned inference `pred_iou_thresh=0.7`, `stability_score_thresh=0.7`; `box_nms_thresh=0.7`; `crop_n_layers=0` |
| HQ-SAM | model `ViT-B`; training batch size `1`; epochs `200`; optimizer `AdamW`; learning rate `0.0001`; weight decay `0.0001`; multiclass scheduler `CosineAnnealingLR`, minimum learning rate `0.000001`; multiclass loss `cross-entropy + Dice`; binary loss `BCE + Dice + IoU MSE`; early-stopping patience `50`; LoRA rank `8`, alpha `16`, dropout `0.05`, targets `qkv` and `proj`; `points_per_side=32`; original inference `pred_iou_thresh=0.85`; binary fine-tuned inference `pred_iou_thresh=0.78`; `stability_score_thresh=0.8`; `crop_n_layers=0` |
| SAM2 | model `sam2.1_hiera_base_plus`; training batch size `1`; epochs `200`; optimizer `AdamW`; learning rate `0.0001`; weight decay `0.0001`; scheduler `CosineAnnealingLR`, minimum learning rate `0.000001`; multiclass loss `cross-entropy + Dice`; binary loss `BCE + Dice + IoU MSE`; early-stopping patience `50`; LoRA rank `8`, alpha `16`, dropout `0.05`, targets `qkv` and `proj`; `points_per_side=32`; `points_per_batch=64`; `pred_iou_thresh=0.8`; `stability_score_thresh=0.8`; `crop_n_layers=0` |
| micro-sam | model `ViT-B-LM`; training batch size `1`; epochs `200`; optimizer `AdamW`; learning rate `0.0001`; weight decay `0.0001`; multiclass scheduler `CosineAnnealingLR`, minimum learning rate `0.000001`; multiclass loss `cross-entropy + Dice`; binary loss `BCE + Dice + IoU MSE`; early-stopping patience `50`; LoRA rank `8`, alpha `16`, dropout `0.05`, targets `qkv` and `proj`; `min_size=10`; `center_distance_threshold=0.5`; `boundary_distance_threshold=0.5` |

## 5. Weights and Checkpoints

Weights are not stored in this repository. Official download links, expected
filenames, and checkpoint compatibility notes are listed in
[`WEIGHTS.md`](WEIGHTS.md). Public scripts accept checkpoint locations through
command-line arguments; no fixed checkpoint directory is required.

Model weights for the evaluated model-dataset combinations are available from
the [Hugging Face model collection](https://huggingface.co/collections/WUT-AI-AI4Mat/segmentation-methods-for-quantitative-microstructure-analysi).
The collection contains trained CNN and Transformer weights as well as
fine-tuned foundation-model weights.

## 6. CNN Baselines

The experiment-specific CNN entry points are:

### U-Net

Folder:

```text
CNN/U-net
```

| Script | Purpose |
| --- | --- |
| `train.py` | Binary U-Net training |
| `predict.py` | Binary U-Net testing, metric export, and raw-mask saving |
| `train_multi.py` | Multi-class U-Net training |
| `test_mutil.py` | Multi-class U-Net testing, metric export, and raw-mask saving |

Experiment settings:

| Parameter | Value |
| --- | --- |
| `ENCODER` | `resnet50` |
| `ENCODER_WEIGHTS` | `imagenet` |
| `INPUT_SIZE` | `512` |
| `BATCH_SIZE` | `32` |
| `NUM_WORKERS` | `8` |
| `EPOCHS` | `500` |
| `LEARNING_RATE` | `0.0003` |
| optimizer | AdamW |
| weight decay | `0.001` |
| learning-rate scheduler | cosine annealing, minimum learning rate `0.000001` |
| binary loss | Dice + BCE with logits |
| multi-class loss | Dice + cross-entropy |
| `PATIENCE` | `50` |
| `CLASSES` | dataset-specific for multi-class experiments |

Example:

```bash
python CNN/U-net/train.py --dataset-root /path/to/MetalDAM --output-dir outputs/unet
python CNN/U-net/predict.py --dataset-root /path/to/MetalDAM --checkpoint outputs/unet/Best_Model.pth --output-dir results/unet
```

### DeepLabV3+

Folder:

```text
CNN/DeepLabV3+
```

| Script | Purpose |
| --- | --- |
| `train.py` | Binary DeepLabV3+ training |
| `test.py` | Binary DeepLabV3+ testing, metric export, and raw-mask saving |
| `train_multi.py` | Multi-class DeepLabV3+ training |
| `test_multi.py` | Multi-class DeepLabV3+ testing, metric export, and raw-mask saving |

Experiment settings:

| Parameter | Value |
| --- | --- |
| `ENCODER` | `resnet50` |
| `ENCODER_WEIGHTS` | `imagenet` |
| `INPUT_SIZE` | `512` |
| `BATCH_SIZE` | `32` |
| `NUM_WORKERS` | `8` |
| `EPOCHS` | `500` |
| `LEARNING_RATE` | `0.0003` |
| optimizer | AdamW |
| weight decay | `0.001` |
| learning-rate scheduler | cosine annealing, minimum learning rate `0.000001` |
| binary loss | Dice + BCE with logits |
| multi-class loss | Dice + cross-entropy |
| `PATIENCE` | `50` |
| `NUM_CLASSES` / `CLASSES` | dataset-specific |

Example:

```bash
python "CNN/DeepLabV3+/train_multi.py" --dataset-root /path/to/EBC --output-dir outputs/deeplab_ebc --num-classes 3
python "CNN/DeepLabV3+/test_multi.py" --dataset-root /path/to/EBC --checkpoint outputs/deeplab_ebc/DeepLabV3P_Best_Model.pth --output-dir results/deeplab_ebc --num-classes 3
```

## 7. SAM-Family Experiments

For MatSAM, HQ-SAM, SAM2, and micro-sam, the custom experiment entry points
are:

- one original test script for the base model;
- binary LoRA+decoder training and test scripts for Aachen-Heerlen, EMPS, and
  Grain;
- class-token training and test scripts for EBC, Super, MetalDAM, and UHCS.

The architecture descriptions follow the manuscript. The experiment settings below follow the current public training and testing functions.

### MatSAM

Folder:

```text
matsam
```

Clone the official MatSAM repository into `matsam/upstream` before running
these scripts. This directory is ignored by Git and is not redistributed by
this repository.

| Script | Purpose |
| --- | --- |
| `datasets_test.py` | Original MatSAM testing |
| `train_lora_decoder.py` | Binary LoRA+decoder fine-tuning |
| `test_lora_decoder.py` | Binary LoRA+decoder testing |
| `train_semantic_matsam.py` | Multi-class LoRA+decoder fine-tuning with semantic class tokens |
| `test_semantic_matsam.py` | Testing with the fine-tuned semantic class-token model |

Main settings:

| Parameter | Value |
| --- | --- |
| `MODEL_TYPE` | `vit_h` |
| `NUM_CLASSES` | dataset-specific |
| training batch size | `1` |
| `EPOCHS` | `200` |
| `LR` | `0.0001` |
| optimizer | AdamW |
| weight decay | `0.0001` |
| learning-rate scheduler | cosine annealing, minimum learning rate `0.000001` |
| binary loss | BCE + Dice + IoU MSE |
| multi-class loss | cross-entropy + Dice |
| `PATIENCE` | `50` |
| LoRA | rank `8`, alpha `16`, dropout `0.05`; targets `qkv` and `proj` |
| prompt generation | `method_type=1` for grain data; `method_type=2` for metallographic data; `n_per_side_base=54` |
| original inference mask filtering | `pred_iou_thresh=0.88`, `stability_score_thresh=0.9`, `box_nms_thresh=0.7`, `crop_n_layers=0` |

Example:

```bash
python matsam/datasets_test.py --dataset-root /path/to/MetalDAM --checkpoint weights/sam_vit_h_4b8939.pth --output-dir results/matsam_original --method-type 2
python matsam/train_lora_decoder.py --dataset-root /path/to/EMPS --checkpoint weights/sam_vit_h_4b8939.pth --output-dir outputs/matsam_emps
python matsam/test_lora_decoder.py --dataset-root /path/to/EMPS --checkpoint weights/sam_vit_h_4b8939.pth --finetuned-checkpoint outputs/matsam_emps/matsam_lora_decoder_best.pth --output-dir results/matsam_emps --method-type 2
python matsam/train_semantic_matsam.py --dataset-root /path/to/MetalDAM --checkpoint weights/sam_vit_h_4b8939.pth --output-dir outputs/matsam --num-classes 5
python matsam/test_semantic_matsam.py --dataset-root /path/to/MetalDAM --checkpoint weights/sam_vit_h_4b8939.pth --finetuned-checkpoint outputs/matsam/semantic_matsam_best.pth --output-dir results/matsam --num-classes 5
```

### HQ-SAM

Folder:

```text
hqsam
```

| Script | Purpose |
| --- | --- |
| `test_hqsam.py` | Original HQ-SAM testing |
| `train_lora_decoder.py` | Binary LoRA+decoder fine-tuning |
| `test_lora_decoder.py` | Binary LoRA+decoder testing |
| `train_semantic_hqsam.py` | Multi-class LoRA+decoder fine-tuning with semantic class tokens |
| `test_semantic_hqsam.py` | Testing with the fine-tuned semantic class-token model |

Main settings:

| Parameter | Value |
| --- | --- |
| `MODEL_TYPE` | `vit_b` |
| `NUM_CLASSES` | dataset-specific |
| training batch size | `1` |
| `EPOCHS` | `200` |
| `LR` | `0.0001` |
| optimizer | AdamW |
| weight decay | `0.0001` |
| learning-rate scheduler | cosine annealing, minimum learning rate `0.000001` |
| binary loss | BCE + Dice + IoU MSE |
| multi-class loss | cross-entropy + Dice |
| `PATIENCE` | `50` |
| LoRA | rank `8`, alpha `16`, dropout `0.05`; targets `qkv` and `proj` |
| original inference automatic mask generation | `points_per_side=32`, `pred_iou_thresh=0.85`, `stability_score_thresh=0.8`, `crop_n_layers=0` |

Example:

```bash
python hqsam/test_hqsam.py --dataset-root /path/to/EBC --checkpoint weights/sam_hq_vit_b.pth --output-dir results/hqsam_original
python hqsam/train_lora_decoder.py --dataset-root /path/to/EMPS --checkpoint weights/sam_hq_vit_b.pth --output-dir outputs/hqsam_emps
python hqsam/test_lora_decoder.py --dataset-root /path/to/EMPS --checkpoint weights/sam_hq_vit_b.pth --finetuned-checkpoint outputs/hqsam_emps/hqsam_lora_decoder_best.pth --output-dir results/hqsam_emps
python hqsam/train_semantic_hqsam.py --dataset-root /path/to/EBC --checkpoint weights/sam_hq_vit_b.pth --output-dir outputs/hqsam --num-classes 3
python hqsam/test_semantic_hqsam.py --dataset-root /path/to/EBC --checkpoint weights/sam_hq_vit_b.pth --finetuned-checkpoint outputs/hqsam/semantic_hqsam_best.pth --output-dir results/hqsam --num-classes 3
```

### SAM2

Folder:

```text
sam2
```

| Script | Purpose |
| --- | --- |
| `test_sam2.py` | Original SAM2 testing |
| `train_lora_decoder.py` | Binary LoRA+decoder fine-tuning |
| `test_lora_decoder.py` | Binary LoRA+decoder testing |
| `train_semantic_sam2.py` | Multi-class LoRA+decoder fine-tuning with semantic class tokens |
| `test_semantic_sam2.py` | Testing with the fine-tuned semantic class-token model |

Main settings:

| Parameter | Value |
| --- | --- |
| `MODEL_CFG` | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| `NUM_CLASSES` | dataset-specific |
| training batch size | `1` |
| `NUM_EPOCHS` | `200` |
| `LEARNING_RATE` | `0.0001` |
| optimizer | AdamW |
| weight decay | `0.0001` |
| learning-rate scheduler | cosine annealing, minimum learning rate `0.000001` |
| binary loss | BCE + Dice + IoU MSE |
| multi-class loss | cross-entropy + Dice |
| `PATIENCE` | `50` |
| LoRA | rank `8`, alpha `16`, dropout `0.05`; targets `qkv` and `proj` |
| automatic mask generation | `points_per_side=32`, `points_per_batch=64`, `pred_iou_thresh=0.8`, `stability_score_thresh=0.8`, `crop_n_layers=0` |

The SAM2 training and fine-tuned test scripts both resize the longest side to
`1024`, pad to `1024 x 1024`, remove the padded region from logits, and then
restore the original image size.

Example:

```bash
python sam2/test_sam2.py --dataset-root /path/to/EBC --checkpoint weights/sam2.1_hiera_base_plus.pt --output-dir results/sam2_original
python sam2/train_lora_decoder.py --dataset-root /path/to/EMPS --checkpoint weights/sam2.1_hiera_base_plus.pt --output-dir outputs/sam2_emps
python sam2/test_lora_decoder.py --dataset-root /path/to/EMPS --checkpoint weights/sam2.1_hiera_base_plus.pt --finetuned-checkpoint outputs/sam2_emps/sam2_lora_decoder_best.pth --output-dir results/sam2_emps
python sam2/train_semantic_sam2.py --dataset-root /path/to/EBC --checkpoint weights/sam2.1_hiera_base_plus.pt --output-dir outputs/sam2 --num-classes 3
python sam2/test_semantic_sam2.py --dataset-root /path/to/EBC --checkpoint weights/sam2.1_hiera_base_plus.pt --finetuned-checkpoint outputs/sam2/semantic_sam2_best.pth --output-dir results/sam2 --num-classes 3
```

### micro-sam

Folder:

```text
microsam
```

| Script | Purpose |
| --- | --- |
| `data_test.py` | Original micro-sam testing |
| `train_lora_decoder.py` | Binary LoRA+decoder fine-tuning |
| `test_lora_decoder.py` | Binary LoRA+decoder testing |
| `train_semantic_microsam.py` | Multi-class LoRA+decoder fine-tuning with semantic class tokens |
| `test_semantic_microsam.py` | Testing with the fine-tuned semantic class-token model |

Main settings:

| Parameter | Value |
| --- | --- |
| `MODEL_TYPE` | `vit_b_lm` |
| `NUM_CLASSES` | dataset-specific |
| training batch size | `1` |
| `EPOCHS` | `200` |
| `LR` | `0.0001` |
| optimizer | AdamW |
| weight decay | `0.0001` |
| learning-rate scheduler | cosine annealing, minimum learning rate `0.000001` |
| binary loss | BCE + Dice + IoU MSE |
| multi-class loss | cross-entropy + Dice |
| `PATIENCE` | `50` |
| LoRA | rank `8`, alpha `16`, dropout `0.05`; targets `qkv` and `proj` |
| automatic instance segmentation | `min_size=10`, `center_distance_threshold=0.5`, `boundary_distance_threshold=0.5` |

Example:

```bash
python microsam/data_test.py --dataset-root /path/to/EBC --checkpoint weights/vit_b.pt --output-dir results/microsam_original
python microsam/train_lora_decoder.py --dataset-root /path/to/EMPS --output-dir outputs/microsam_emps
python microsam/test_lora_decoder.py --dataset-root /path/to/EMPS --finetuned-checkpoint outputs/microsam_emps/microsam_lora_decoder_best.pth --output-dir results/microsam_emps
python microsam/train_semantic_microsam.py --dataset-root /path/to/EBC --output-dir outputs/microsam --num-classes 3
python microsam/test_semantic_microsam.py --dataset-root /path/to/EBC --finetuned-checkpoint outputs/microsam/semantic_microsam_best.pth --output-dir results/microsam --num-classes 3
```

## 8. Swin-Unet

Folder:

```text
Swin-Unet
```

The experiment-specific entry scripts are:

| Script | Purpose |
| --- | --- |
| `train_standalone.py` | Swin-Unet training for the material dataset structure |
| `test_data.py` | Swin-Unet testing and metric export |

Experiment settings:

| Parameter | Value |
| --- | --- |
| `IMG_SIZE` | `224` |
| `BATCH_SIZE` | `32` |
| `NUM_WORKERS` | `8` |
| `EPOCHS` | `500` |
| `BASE_LR` | `0.0001` |
| `WEIGHT_DECAY` | `0.0001` |
| optimizer | AdamW |
| learning-rate scheduler | cosine annealing, minimum learning rate `0.000001` |
| loss | `0.5 x` cross-entropy + `0.5 x` Dice |
| `PATIENCE` | `50` |
| pretrained model | ImageNet-pretrained Swin-Tiny, patch size `4`, window size `7` |

Clone the official Swin-Unet repository into `Swin-Unet/upstream` before
running these scripts. The experiment entry scripts import `config.py` and
`networks/` from that ignored directory; dataset loading remains in the two
experiment scripts.

Example:

```bash
python Swin-Unet/train_standalone.py --dataset-root /path/to/EMPS --num-classes 2 --pretrained-checkpoint weights/swin_tiny_patch4_window7_224.pth --output-dir outputs/swin
python Swin-Unet/test_data.py --dataset-root /path/to/EMPS --num-classes 2 --checkpoint outputs/swin/EMPS/best_model.pth --output-dir results/swin
```

## 9. SegFormer / MMSegmentation

SegFormer is reproduced through MMSegmentation configs.

Custom configs:

```text
mmsegmentation/configs/segformer/segformer_mit-b0_1xb2-200e_aachen-heerlen-512x512.py
mmsegmentation/configs/segformer/segformer_mit-b0_1xb2-200e_ebc-512x512.py
mmsegmentation/configs/segformer/segformer_mit-b0_1xb2-200e_emps-512x512.py
mmsegmentation/configs/segformer/segformer_mit-b0_1xb2-200e_grain-512x512.py
mmsegmentation/configs/segformer/segformer_mit-b0_1xb2-200e_super-512x512.py
mmsegmentation/configs/segformer/segformer_mit-b0_1xb2-200e_uhcs-512x512.py
mmsegmentation/configs/segformer/segformer_mit-b0_1xb2-2k_metaldam-512x512.py
```

Shared experiment settings:

| Parameter | Value |
| --- | --- |
| backbone/head | SegFormer MiT-B0 |
| crop size | `512 x 512` |
| max epochs | `200` |
| train batch size | `32` |
| validation/test batch size | `16` |
| optimizer | AdamW |
| learning rate | `0.00006` |
| Adam betas | `(0.9, 0.999)` |
| weight decay | `0.01` |
| learning-rate scheduler | 5-epoch linear warm-up followed by polynomial decay |
| number of classes | dataset-specific |

SegFormer testing is aligned with the other models through:

```text
mmsegmentation/tools/test_segformer_dataset.py
```

This script loads a SegFormer config/checkpoint, runs inference on the test images, saves raw predicted masks, calls `Myutils.metrics.Metric.compute_all` for metric calculation, and exports `metrics_summary.xlsx`. Dataset-specific wrappers call the same shared testing logic:

```text
mmsegmentation/tools/test_aachen_heerlen_segformer.py
mmsegmentation/tools/test_ebc_segformer.py
mmsegmentation/tools/test_emps_segformer.py
mmsegmentation/tools/test_grain_segformer.py
mmsegmentation/tools/test_metaldam_segformer.py
mmsegmentation/tools/test_super_segformer.py
mmsegmentation/tools/test_uhcs_segformer.py
```

Run examples:

```bash
cd mmsegmentation
export DATASET_ROOT=/path/to/UHCS
python tools/train.py configs/segformer/segformer_mit-b0_1xb2-200e_uhcs-512x512.py
python tools/test_uhcs_segformer.py --dataset-root /path/to/UHCS --checkpoint path/to/checkpoint.pth
```

## 10. Utilities

Folder:

```text
Myutils
```

| File | Purpose |
| --- | --- |
| `metrics.py` | Accuracy, mIoU, Dice, precision, recall, HD, and HD95 calculation |
| `visualizer.py` | Prediction mask saving and image-mask comparison visualization |
| `drawn_Fig_3.py` | Generates the CE-versus-quantitative-analysis-error scatter plot used in Figure 3 |
| `drawn_mul.py` | Generates grouped bar charts for metric gaps, top-k overlap, correlation comparisons, and ablation results |
| `drawn_sup_2.py` | Evaluates ranking stability after removing combinations of methods and exports the Supplementary Figure 2 results |
| `drawn_sup_3.py` | Evaluates metric-ranking robustness under repeated random sample removal using a fixed reference ranking |
| `drawn_sup_3_new.py` | Evaluates metric-ranking robustness under repeated random sample removal using a dynamically recomputed reference ranking |

The test scripts call these utilities to export metrics and save raw predicted
masks. Comparison figures are optional post-processing outputs.

### Paper Figure and Robustness Analysis

The `drawn*.py` scripts are post-processing utilities used after model
evaluation. They are not required for model training or inference. Install the
additional plotting dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scipy openpyxl
```

| Script | Input | Processing | Output |
| --- | --- | --- | --- |
| `drawn_Fig_3.py` | Per-model Excel evaluation files | Reads CE from the 11th column and quantitative analysis error from the final column, then draws a symmetric-log scatter plot | `Fig_3d.pdf` |
| `drawn_mul.py` | Figure values stored in the selected plotting function | Draws grouped bar charts through `drawnHistogram`; the available entry functions cover metric gaps, Spearman correlations, top-3 overlap, and ablation comparisons | `Fig_2c.pdf`, `fig_2d.pdf`, `Fig_4.pdf`, or `top_Ablation_vs.pdf`, depending on the called function |
| `drawn_sup_2.py` | One summary Excel file containing model names, metric values, and the reference score in the final column | Removes every one-, two-, and three-method combination and calculates the Spearman correlation between each metric ranking and the remaining reference ranking | `Supp_2g.xlsx` and `Supp_2g.pdf` |
| `drawn_sup_3.py` | One per-model Excel file containing image-level metric results | Repeats 100 trials with 30% of samples removed, ranks models by the mean of each metric, and compares the rankings with a fixed reference ranking | `Supp_3g.xlsx` and `Supp_3g.pdf` |
| `drawn_sup_3_new.py` | One per-model Excel file containing image-level metric results | Repeats 100 trials with 30% of samples removed and recomputes the reference ranking from the final Excel column in every trial | `Supp_3a.xlsx` and `Supp_3a.pdf` |

The robustness scripts expect the metric columns `miou`, `dice`, `precision`,
`recall`, `acc`, `hd95`, `mae`, and `nsd`. Because lower values are better for
HD95 and MAE/CE, these columns are sign-inverted before ranking so that all
metrics can be ordered in the same direction.

The figure utilities accept input and output paths from the command line:

```bash
python Myutils/drawn_Fig_3.py --input-dir results/excels --output figures/Fig_3d.pdf
python Myutils/drawn_mul.py --figure fig2d --output figures/Fig_2d.pdf
python Myutils/drawn_sup_2.py --input results/summary.xlsx --output-xlsx figures/Supp_2g.xlsx --output-figure figures/Supp_2g.pdf
python Myutils/drawn_sup_3.py --input-dir results/per_model --output-xlsx figures/Supp_3g.xlsx --output-figure figures/Supp_3g.pdf --seed 42
```

## 11. Suggested Reproduction Order

1. Create the Python environment for the selected model family.
2. Install the corresponding official requirements.
3. Prepare datasets with the required `DATASET_NAME/{train,val,test}/{images,masks}` structure.
4. Download pretrained weights and place them according to `WEIGHTS.md`.
5. Run training scripts.
6. Run the corresponding test scripts.
7. Run the required `Myutils/drawn*.py` post-processing scripts to reproduce the paper figures.

## 12. Notes Before Public Release

- Large model weights and generated checkpoints are intentionally excluded from the repository.
- MatSAM and Swin-Unet upstream source is intentionally excluded. Clone each
  project as described above and review its terms before use or redistribution;
  the recorded upstream repositories do not provide an explicit source-code
  license.
