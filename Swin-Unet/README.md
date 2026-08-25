# Swin-Unet Baseline

This directory contains the Swin-Unet files used for the material image
segmentation experiments. The retained public entry points are:

- `train_standalone.py`: binary or multiclass training;
- `test_data.py`: checkpoint evaluation and metric export.

Both scripts use `datasets/dataset_material.py`, the network implementation in
`networks/`, and `configs/swin_tiny_patch4_window7_224_lite.yaml`. They expect
the dataset layout documented in the repository root README.

## Environment

Install the tested dependencies from the repository root documentation and
place the ImageNet-pretrained Swin-Tiny checkpoint as described in
[`../WEIGHTS.md`](../WEIGHTS.md).

## Usage

```bash
python Swin-Unet/train_standalone.py \
  --dataset-root /path/to/DATASET_NAME \
  --num-classes 2 \
  --pretrained-checkpoint /path/to/swin_tiny_patch4_window7_224.pth \
  --output-dir outputs/swin

python Swin-Unet/test_data.py \
  --dataset-root /path/to/DATASET_NAME \
  --num-classes 2 \
  --checkpoint outputs/swin/DATASET_NAME/best_model.pth \
  --output-dir results/swin
```

The implementation is derived from
[HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet). See
[`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) for redistribution
status and attribution.
