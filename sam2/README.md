# SAM2 Experiments

This directory contains the SAM2.1 runtime required by the material image
segmentation experiments and the following public entry points:

- `test_sam2.py`: original SAM2 automatic mask inference;
- `train_lora_decoder.py` and `test_lora_decoder.py`: binary LoRA and mask
  decoder fine-tuning for Aachen-Heerlen, EMPS, and Grain;
- `train_semantic_sam2.py` and `test_semantic_sam2.py`: multiclass class-token
  fine-tuning for EBC, Super, MetalDAM, and UHCS.

The experiments use `sam2.1_hiera_base_plus` and the configuration at
`sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`. Follow [`INSTALL.md`](INSTALL.md)
for the package installation and download the official checkpoint listed in
[`../WEIGHTS.md`](../WEIGHTS.md). Exact commands, dataset structure,
hyperparameters, and output files are documented in
[`../REPRODUCTION.md`](../REPRODUCTION.md).

The model implementation is derived from
[facebookresearch/sam2](https://github.com/facebookresearch/sam2) and remains
subject to the Apache-2.0 license in [`LICENSE`](LICENSE). The connected
components CUDA source keeps the additional terms in
[`LICENSE_cctorch`](LICENSE_cctorch).
