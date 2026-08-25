# micro-sam Experiments

This directory contains the micro-sam runtime required by the material image
segmentation experiments and the following public entry points:

- `data_test.py`: original micro-sam automatic instance segmentation;
- `train_lora_decoder.py` and `test_lora_decoder.py`: binary LoRA and mask
  decoder fine-tuning for Aachen-Heerlen, EMPS, and Grain;
- `train_semantic_microsam.py` and `test_semantic_microsam.py`: multiclass
  class-token fine-tuning for EBC, Super, MetalDAM, and UHCS.

Create the environment from `environment.yaml`, install this directory in
editable mode, and obtain the ViT-B-LM checkpoint as described in
[`../WEIGHTS.md`](../WEIGHTS.md). Exact commands, dataset structure,
hyperparameters, and output files are documented in
[`../REPRODUCTION.md`](../REPRODUCTION.md).

The runtime is derived from
[computational-cell-analytics/micro-sam](https://github.com/computational-cell-analytics/micro-sam)
and remains subject to the MIT license in [`LICENSE`](LICENSE). The bundled
`torch-em` runtime keeps its license in [`torch-em/LICENSE`](torch-em/LICENSE).
