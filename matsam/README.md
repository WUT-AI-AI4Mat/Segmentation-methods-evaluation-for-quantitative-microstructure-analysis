# MatSAM Experiments

This directory contains the MatSAM runtime required by the material image
segmentation experiments and the following public entry points:

- `datasets_test.py`: original MatSAM automatic mask inference;
- `train_lora_decoder.py` and `test_lora_decoder.py`: binary LoRA and mask
  decoder fine-tuning for Aachen-Heerlen, EMPS, and Grain;
- `train_semantic_matsam.py` and `test_semantic_matsam.py`: multiclass
  class-token fine-tuning for EBC, Super, MetalDAM, and UHCS.

Install the dependencies in `requirements.txt` and download the official SAM
ViT-H checkpoint listed in [`../WEIGHTS.md`](../WEIGHTS.md). Exact commands,
dataset structure, MatSAM prompt parameters, hyperparameters, and output files
are documented in [`../REPRODUCTION.md`](../REPRODUCTION.md).

The implementation is derived from
[USTB-AI3DVIP/matsam](https://github.com/USTB-AI3DVIP/matsam). The recorded
upstream source did not include a license file; consult
[`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) before
redistributing this directory.
