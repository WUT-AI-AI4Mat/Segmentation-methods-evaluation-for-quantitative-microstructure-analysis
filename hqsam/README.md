# HQ-SAM Experiments

This directory contains the HQ-SAM runtime required by the material image
segmentation experiments and the following public entry points:

- `test_hqsam.py`: original HQ-SAM inference;
- `train_lora_decoder.py` and `test_lora_decoder.py`: binary LoRA and mask
  decoder fine-tuning for Aachen-Heerlen, EMPS, and Grain;
- `train_semantic_hqsam.py` and `test_semantic_hqsam.py`: multiclass
  class-token fine-tuning for EBC, Super, MetalDAM, and UHCS.

Install the dependencies in `requirements.txt`, install this directory in
editable mode, and download the official ViT-B checkpoint listed in
[`../WEIGHTS.md`](../WEIGHTS.md). Exact commands, dataset structure,
hyperparameters, and output files are documented in
[`../REPRODUCTION.md`](../REPRODUCTION.md).

The model implementation is derived from
[SysCV/sam-hq](https://github.com/SysCV/sam-hq) and remains subject to the
Apache-2.0 license in [`LICENSE`](LICENSE).
