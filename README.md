# Material Image Segmentation Benchmark

This repository contains the code used to compare CNN, Transformer, and
Segment Anything based models for material image segmentation.

The evaluated datasets are MetalDAM, UHCS, EBC, Super, Aachen-Heerlen, EMPS,
and Grain. The full datasets, pretrained weights, and fine-tuned checkpoints
are not included. The CPU demo contains one test image-mask pair from each
dataset.

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

The model experiments were run on an Ubuntu 22.04.3 LTS cloud server with two
NVIDIA GeForce RTX 4090 GPUs. A separate conda environment was used for each
model family, except that U-Net and DeepLabV3+ shared the CNN environment. The
following versions were read directly from the tested environments:

| Models | Conda environment | Python | PyTorch | PyTorch CUDA | Main framework | Complete package list |
| --- | --- | ---: | ---: | ---: | --- | --- |
| U-Net and DeepLabV3+ | `sam` | 3.9.25 | 2.8.0 | 12.8 | SMP 0.5.0 | [`sam.txt`](environment_reports/sam.txt) |
| MatSAM | `matsam` | 3.9.25 | 2.5.1 | 12.1 | PEFT 0.17.1 | [`matsam.txt`](environment_reports/matsam.txt) |
| HQ-SAM | `hqsam` | 3.9.25 | 2.8.0 | 12.8 | PEFT 0.17.1 | [`hqsam.txt`](environment_reports/hqsam.txt) |
| SAM2 | `sam2` | 3.10.19 | 2.10.0 | 12.8 | PEFT 0.18.1 | [`sam2.txt`](environment_reports/sam2.txt) |
| micro-sam | `microsam` | 3.10.19 | 2.9.1 | 12.8 | micro-sam 1.6.2 | [`microsam.txt`](environment_reports/microsam.txt) |
| Swin-Unet | `swinunet` | 3.9.25 | 2.8.0 | 12.8 | timm 1.0.24 | [`swinunet.txt`](environment_reports/swinunet.txt) |
| SegFormer | `segformer` | 3.9.25 | 2.1.2 | 12.1 | MMSegmentation 1.2.2 | [`segformer.txt`](environment_reports/segformer.txt) |

Full training and GPU inference require an NVIDIA CUDA-capable GPU. At least
24 GB of GPU memory and 32 GB of system memory are recommended for the
SAM-family fine-tuning scripts at batch size 1. CNN, Swin-Unet, and SegFormer
memory use depends on the configured batch size; reduce the batch size if the
available GPU memory is lower. Model checkpoints require several additional
gigabytes of disk space and are not stored in this repository.

The complete tested server specification is recorded in
[`environment_reports/SERVER.md`](environment_reports/SERVER.md). Exact
installed-package versions for all seven environments are available in
[`environment_reports`](environment_reports/README.md). Use
[`scripts/collect_environment.sh`](scripts/collect_environment.sh) to capture
a fresh report after changing an environment.

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

The repository includes one test image-mask pair from each of the seven
evaluation datasets. The pairs are converted losslessly to PNG and their
original relative paths are recorded in `demo/data/test/samples.csv`.

Run the CPU-only smoke test:

```bash
python demo/run_demo.py --verify
```

Expected terminal output reports that seven images were processed and that the
computed values match `demo/expected_metrics.csv`. The command creates:

```text
demo/output/
  metrics.csv
  predicted_masks/
    aachen_heerlen_IMG_00004.png
    ebc_010417_4_S2480009.png
    emps_01ac659240.png
    grain_A_01_04.png
    metaldam_micrograph1.png
    super_45kx_SE_15kV-etched_0.png
    uhcs_A_micrograph1006.png
```

The demo completed in approximately 2 seconds in a clean environment on the
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

`--dataset-root` must point to one dataset directory, for example
`/data/datasets/UHCS`, rather than the parent directory containing all seven
datasets. Training scripts read `train/images`, `train/masks`, `val/images`,
and `val/masks`; test scripts read `test/images` and `test/masks`.

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
- original inference, binary LoRA+decoder fine-tuning/inference, and
  multiclass class-token fine-tuning/inference for the Segment Anything based
  models;
- `train_standalone.py` and `test_data.py` for Swin-Unet;
- the custom SegFormer configs and `tools/test_segformer_dataset.py`;
- metric, visualization, and paper figure utilities in `Myutils`.

Some model directories contain code derived from upstream research
repositories. The selected upstream source files and original notices are
retained alongside the experiment-specific training and test scripts. Logs,
checkpoints, datasets, and generated results are excluded. See
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
