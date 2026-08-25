# CPU Demo

This smoke test validates image loading, mask generation, and the shared metric
implementation without downloading model checkpoints or the full research
datasets. It
uses one test image-mask pair from each of UHCS, Super, EBC, MetalDAM,
Aachen-Heerlen, EMPS, and Grain. The included PNG files preserve the original
image dimensions and mask label IDs. The demo is not used to report a
manuscript result.

The exact source filenames, included filenames, and class counts are recorded
in `demo/data/test/samples.csv`. Confirm that the source dataset terms permit
redistribution before making the repository public.

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
    aachen_heerlen_IMG_00004.png
    ebc_010417_4_S2480009.png
    emps_01ac659240.png
    grain_A_01_04.png
    metaldam_micrograph1.png
    super_45kx_SE_15kV-etched_0.png
    uhcs_A_micrograph1006.png
```

The final row of `metrics.csv` is the average over the seven images. With
`--verify`, the values are compared with `demo/expected_metrics.csv` using a
tolerance of `0.000001`.

In a clean Python 3.10 environment on the tested Windows desktop, dependency
installation took approximately 44 seconds and the demo took approximately 2
seconds. Runtime excludes environment creation and package download time. A
GPU is not used.

To rebuild the included sample set from the full datasets:

```bash
python demo/prepare_demo_data.py --datasets-root /path/to/datasets
```
