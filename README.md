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

## Quick Start

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
