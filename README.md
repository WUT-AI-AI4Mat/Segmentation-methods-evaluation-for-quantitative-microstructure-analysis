# Material Image Segmentation Benchmark

This repository contains the code used to compare CNN, Transformer, and
Segment Anything based models for material image segmentation.

The evaluated datasets are MetalDAM, UHCS, EBC, Super, Aachen-Heerlen, EMPS,
and Grain. Dataset images, annotations, pretrained weights, and fine-tuned
checkpoints are not included.

## Models

| Family | Models |
| --- | --- |
| CNN | U-Net, DeepLabV3+ |
| Transformer | SegFormer, Swin-Unet |
| Segment Anything | MatSAM, HQ-SAM, SAM2, micro-sam |

## System Requirements

### CPU Demo

The included smoke test runs on a standard desktop without a GPU. It was
validated with the following configuration:

| Component | Tested version |
| --- | --- |
| Operating system | Windows 11, 64-bit |
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| OpenCV | 4.13.0 |
| SciPy | 1.15.3 |
| Memory | 4 GB or more |
| GPU | Not required |

Linux and macOS are also expected to work because the demo uses only portable
Python packages, but the tested desktop configuration above is the reference
configuration.

### Full Benchmark

The model experiments were run on Linux cloud servers. A separate conda
environment is used for each model family, except that U-Net and DeepLabV3+
share the CNN environment. The reference Python versions and dependency files
are:

| Model family | Operating system | Python | Dependency specification |
| --- | --- | ---: | --- |
| U-Net and DeepLabV3+ | Linux | 3.10 | [CNN environment](REPRODUCTION.md#cnn-baselines) |
| MatSAM | Linux | 3.8 | [`matsam/requirements.txt`](matsam/requirements.txt) |
| HQ-SAM | Linux | 3.10 | [`hqsam/requirements.txt`](hqsam/requirements.txt) |
| SAM2 | Linux | 3.10 or later | [`sam2/INSTALL.md`](sam2/INSTALL.md) |
| micro-sam | Linux | 3.10 | [`microsam/environment.yaml`](microsam/environment.yaml) |
| Swin-Unet | Linux | 3.8 | [`Swin-Unet/requirements.txt`](Swin-Unet/requirements.txt) |
| SegFormer | Linux | 3.10 | [`mmsegmentation/requirements`](mmsegmentation/requirements) |

Full training and GPU inference require an NVIDIA CUDA-capable GPU. At least
24 GB of GPU memory and 32 GB of system memory are recommended for the
SAM-family fine-tuning scripts at batch size 1. CNN, Swin-Unet, and SegFormer
memory use depends on the configured batch size; reduce the batch size if the
available GPU memory is lower. Model checkpoints require several additional
gigabytes of disk space and are not stored in this repository.

Exact operating-system, driver, CUDA, GPU, PyTorch, and installed-package
versions should be captured from every cloud environment with
[`scripts/collect_environment.sh`](scripts/collect_environment.sh). The
commands for all seven environments are listed in
[`environment_reports/README.md`](environment_reports/README.md).

## Installation Guide

### Demo Installation

From the repository root:

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r demo/requirements.txt
```

On Windows, activate the environment with
`.venv-demo\Scripts\Activate.ps1`. Installation of the three pinned demo
dependencies took approximately 44 seconds on the tested desktop with packages
available from PyPI. Network speed and an empty package cache can increase this
time.

The full model environments typically require approximately 10-40 minutes per
environment, excluding checkpoint downloads. The exact time depends on the
PyTorch/CUDA build, network speed, and whether CUDA extensions are compiled.
Complete model-specific commands are provided in
[REPRODUCTION.md](REPRODUCTION.md#2-environment).

## Demo

The repository includes three deterministic simulated microscopy-like images
and masks. They are independent of the seven research datasets and can be
redistributed with this repository.

Run the CPU-only smoke test:

```bash
python demo/run_demo.py --verify
```

Expected terminal output reports that three images were processed and that the
computed values match `demo/expected_metrics.csv`. The command creates:

```text
demo/output/
  metrics.csv
  predicted_masks/
    synthetic_01.png
    synthetic_02.png
    synthetic_03.png
```

The demo completed in approximately 3 seconds in a clean environment on the
tested desktop. It validates data loading, mask generation, and the shared
metric implementation; it does not reproduce a manuscript result. See
[`demo/README.md`](demo/README.md) for details.

## Documentation

- [Reproduction guide](REPRODUCTION.md): environments, datasets, entry
  scripts, configurations, hyperparameters, and evaluation workflow.
- [Model weights](WEIGHTS.md): official download sources and expected
  checkpoint names.
- [Third-party software](THIRD_PARTY_LICENSES.md): upstream repositories,
  revisions, and license status.

## Dataset Layout

Each dataset must use the following structure:

```text
DATASET_NAME/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Instructions for Use

1. Create the environment for the selected model family by following
   [REPRODUCTION.md](REPRODUCTION.md).
2. Download the required pretrained checkpoint from [WEIGHTS.md](WEIGHTS.md).
3. Train the model with the corresponding public training entry script.
4. Evaluate the saved checkpoint with the corresponding test script.
5. Use `Myutils/metrics.py` and the model test scripts to export the same
   metrics used in the comparison.

Run an entry script with `--help` to see its required paths and optional
parameters. Paths to datasets, checkpoints, and output directories are
provided at runtime and are not tied to a particular workstation.

The seven research datasets use the directory layout shown above. Detailed
commands for running every model on a user-provided dataset are listed in
[REPRODUCTION.md](REPRODUCTION.md). Official pretrained checkpoint links and
expected filenames are listed in [WEIGHTS.md](WEIGHTS.md).

## Reproduction

The optional full reproduction workflow, model hyperparameters, training and
test entry points, and paper-figure utilities are documented in
[REPRODUCTION.md](REPRODUCTION.md). The full research datasets and fine-tuned
checkpoints are intentionally excluded from Git; publish their permanent data
and checkpoint links separately when available.

## Scope

The public experiment interface is limited to:

- binary and multiclass training/testing for U-Net and DeepLabV3+;
- original inference and class-token fine-tuning/inference for the
  Segment Anything based models;
- `train_standalone.py` and `test_data.py` for Swin-Unet;
- the custom SegFormer configs and `tools/test_segformer_dataset.py`;
- metric, visualization, and paper figure utilities in `Myutils`.

Some model directories contain code derived from upstream research
repositories. Their original notices remain in those directories. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) before redistribution.

## Citation

Citation information for the associated paper should be added here after the
paper metadata is public. When using this repository, also cite the original
papers for the compared models and follow their license and checkpoint terms.

## License

The original experiment code in this repository is released under the
[MIT License](LICENSE). Third-party components remain subject to their
respective upstream terms; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
