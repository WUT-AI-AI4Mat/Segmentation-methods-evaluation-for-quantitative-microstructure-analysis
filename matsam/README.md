# MatSAM Experiment Scripts

This directory contains only the experiment-specific training and testing
scripts used in this benchmark. The MatSAM upstream source is not redistributed
in this repository.

## Upstream Setup

From the benchmark repository root, clone MatSAM into the ignored `upstream`
directory:

```bash
git clone https://github.com/USTB-AI3DVIP/matsam.git matsam/upstream
git -C matsam/upstream checkout cbea7edaada991d88d7dfee656bd7e3dac09863f
```

Create the MatSAM environment and install the upstream requirements plus the
packages used by the experiment scripts:

```bash
conda create -n matsam python=3.9
conda activate matsam
pip install -r matsam/upstream/requirements.txt
pip install peft pandas openpyxl tifffile
```

Download the official SAM ViT-H checkpoint as described in `../WEIGHTS.md`.

## Experiment Entry Points

| Script | Purpose |
| --- | --- |
| `datasets_test.py` | Original MatSAM inference and evaluation |
| `train_lora_decoder.py` | Binary LoRA and mask-decoder fine-tuning |
| `test_lora_decoder.py` | Binary fine-tuned model evaluation |
| `train_semantic_matsam.py` | Multi-class class-token fine-tuning |
| `test_semantic_matsam.py` | Multi-class class-token evaluation |

Run any script with `--help` after cloning the upstream repository. Complete
dataset, checkpoint, and command examples are provided in
`../REPRODUCTION.md`.

MatSAM did not publish an explicit source-code license in the recorded upstream
revision. Review the upstream terms before using or redistributing its source.
