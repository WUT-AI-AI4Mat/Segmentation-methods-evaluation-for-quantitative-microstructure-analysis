# CPU Demo

This smoke test validates image loading, mask generation, and the shared metric
implementation without downloading model checkpoints or research datasets. It
uses three deterministic, simulated microscopy-like images and binary masks.
The demo is not used to report any manuscript result.

## Requirements

- Windows, Linux, or macOS
- Python 3.10
- 4 GB RAM
- CPU only; no GPU is required

Create a clean environment from the repository root:

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r demo/requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv-demo\Scripts\Activate.ps1
```

## Run

```bash
python demo/run_demo.py --verify
```

The command reads `demo/data/test/images`, compares predictions with
`demo/data/test/masks`, and creates:

```text
demo/output/
  metrics.csv
  predicted_masks/
    synthetic_01.png
    synthetic_02.png
    synthetic_03.png
```

The final row of `metrics.csv` is the average over the three images. With
`--verify`, the values are compared with `demo/expected_metrics.csv` using a
tolerance of `0.000001`.

In a clean Python 3.10 environment on the tested Windows desktop, dependency
installation took approximately 44 seconds and the demo took approximately 3
seconds. Runtime excludes environment creation and package download time. A
GPU is not used.

To regenerate the included simulated dataset:

```bash
python demo/generate_demo_data.py
```
