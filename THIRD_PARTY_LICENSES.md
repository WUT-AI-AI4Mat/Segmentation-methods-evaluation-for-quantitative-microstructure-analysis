# Third-Party Software

This repository combines experiment code with source code from several
research repositories. The root MIT License applies only to code for which the
repository authors hold the relevant rights. It does not replace the terms of
third-party components. MatSAM and Swin-Unet upstream source is intentionally
not redistributed.

| Local path | Upstream project | Recorded revision | License status |
| --- | --- | --- | --- |
| `hqsam/` | [SysCV/sam-hq](https://github.com/SysCV/sam-hq) | Upstream source copy | Apache License 2.0; local `hqsam/LICENSE` retained |
| `sam2/` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Upstream source copy | Apache License 2.0; local `sam2/LICENSE` retained |
| `microsam/` | [computational-cell-analytics/micro-sam](https://github.com/computational-cell-analytics/micro-sam) | Upstream source copy | MIT License; local `microsam/LICENSE` retained |
| `microsam/torch-em/` | [constantinpape/torch-em](https://github.com/constantinpape/torch-em) | Upstream source copy | MIT License; local `microsam/torch-em/LICENSE` retained |
| `mmsegmentation/` | [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | Upstream source copy | Apache License 2.0; local `mmsegmentation/LICENSE` retained |
| `matsam/upstream/` (not tracked) | [USTB-AI3DVIP/matsam](https://github.com/USTB-AI3DVIP/matsam) | External clone; experiments were based on `cbea7edaada991d88d7dfee656bd7e3dac09863f` | Not redistributed; no license file was present in the recorded upstream revision, so review the upstream terms before use |
| MatSAM `gala` dependency | [janelia-flyem/gala](https://github.com/janelia-flyem/gala) | External dependency | BSD-style license; the dependency source is excluded from the public experiment repository |
| `Swin-Unet/upstream/` (not tracked) | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) | External clone; experiments were based on `f48f623e226e25b6e395c37207915c50aaa9c776` | Not redistributed; no license file is published in the recorded upstream repository, so review its terms before use |

## External Python Dependencies

The repository also uses packages such as PyTorch, torchvision,
segmentation-models-pytorch, Albumentations, OpenCV, NumPy, pandas,
scikit-image, and Matplotlib. These packages are installed separately and keep
their own licenses.

## Checkpoints

Model checkpoints are not covered by the source-code license. Download and use
them under the terms provided by their publishers. See [WEIGHTS.md](WEIGHTS.md)
for official sources.
