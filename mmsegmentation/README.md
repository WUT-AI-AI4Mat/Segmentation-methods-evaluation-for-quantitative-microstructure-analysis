# SegFormer Baseline

This directory contains the MMSegmentation 1.2.2 runtime code and the
SegFormer-B0 configuration files used in the material image segmentation
experiments. Unrelated model configurations, converters, deployment tools,
benchmarks, tests, and project-maintenance files are excluded from the public
benchmark repository.

## Retained Experiment Files

- `configs/segformer/segformer_mit-b0_1xb2-200e_*`: one configuration for
  each of the seven datasets;
- `configs/segformer/segformer_mit-b0_1xb2-200e_multisuffix_base.py`: shared
  200-epoch configuration;
- `tools/train.py`: MMSegmentation training entry point;
- `tools/test_segformer_dataset.py`: shared batch inference, metric export,
  and raw-mask saving implementation;
- `tools/test_*_segformer.py`: dataset-specific test wrappers.

## Installation

```bash
conda create -n segformer python=3.9
conda activate segformer
pip install torch torchvision
cd mmsegmentation
pip install -U openmim
mim install mmengine mmcv
pip install -e .
```

Exact versions from the tested server are listed in
[`../environment_reports/segformer.txt`](../environment_reports/segformer.txt).

## Usage

Set `DATASET_ROOT` before training because the dataset configurations read this
environment variable:

```bash
cd mmsegmentation
export DATASET_ROOT=/path/to/DATASET_NAME
python tools/train.py \
  configs/segformer/segformer_mit-b0_1xb2-200e_uhcs-512x512.py

python tools/test_uhcs_segformer.py \
  --dataset-root /path/to/UHCS \
  --checkpoint /path/to/checkpoint.pth
```

Change the configuration and test wrapper for the selected dataset. The
complete commands and hyperparameters are documented in
[`../REPRODUCTION.md`](../REPRODUCTION.md).

The runtime is derived from
[OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and
remains subject to the Apache-2.0 license in [`LICENSE`](LICENSE).
