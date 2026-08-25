# Swin-Unet Experiment Scripts

This directory contains only the experiment-specific training and testing
scripts used in this benchmark. The Swin-Unet upstream source is not
redistributed in this repository.

## Upstream Setup

From the benchmark repository root, clone Swin-Unet into the ignored
`upstream` directory:

```bash
git clone https://github.com/HuCaoFighting/Swin-Unet.git Swin-Unet/upstream
git -C Swin-Unet/upstream checkout f48f623e226e25b6e395c37207915c50aaa9c776
```

Create the environment and install the upstream requirements plus the packages
used by the experiment scripts:

```bash
conda create -n swinunet python=3.9
conda activate swinunet
pip install torch torchvision
pip install -r Swin-Unet/upstream/requirements.txt
pip install albumentations pandas openpyxl matplotlib opencv-python
```

Download the official Swin-Tiny pretrained checkpoint as described in
`../WEIGHTS.md`.

## Experiment Entry Points

| Script | Purpose |
| --- | --- |
| `train_standalone.py` | Binary or multi-class Swin-Unet training |
| `test_data.py` | Binary or multi-class checkpoint evaluation |

Run either script with `--help` after cloning the upstream repository. Complete
dataset, checkpoint, and command examples are provided in
`../REPRODUCTION.md`.

The recorded Swin-Unet upstream repository does not publish an explicit
source-code license. Review its terms before using or redistributing the
upstream source.
