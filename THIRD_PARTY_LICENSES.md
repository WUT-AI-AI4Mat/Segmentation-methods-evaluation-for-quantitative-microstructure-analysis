# Third-Party Software

This repository combines experiment code with source code derived from several
research repositories. The root MIT License applies only to code for which the
repository authors hold the relevant rights. It does not replace the terms of
third-party components.

| Local path | Upstream project | Recorded revision | License status |
| --- | --- | --- | --- |
| `hqsam/` | [SysCV/sam-hq](https://github.com/SysCV/sam-hq) | Upstream source copy | Apache License 2.0; local `hqsam/LICENSE` retained |
| `sam2/` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Upstream source copy | Apache License 2.0; local `sam2/LICENSE` retained |
| `microsam/` | [computational-cell-analytics/micro-sam](https://github.com/computational-cell-analytics/micro-sam) | Upstream source copy | MIT License; local `microsam/LICENSE` retained |
| `microsam/torch-em/` | [constantinpape/torch-em](https://github.com/constantinpape/torch-em) | Upstream source copy | MIT License; local `microsam/torch-em/LICENSE` retained |
| `mmsegmentation/` | [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | Upstream source copy | Apache License 2.0; local `mmsegmentation/LICENSE` retained |
| `matsam/` | [USTB-AI3DVIP/matsam](https://github.com/USTB-AI3DVIP/matsam) | `cbea7edaada991d88d7dfee656bd7e3dac09863f` plus experiment changes | No license file was present in the recorded upstream copy; obtain permission or clarification before redistributing this source |
| MatSAM `gala` dependency | [janelia-flyem/gala](https://github.com/janelia-flyem/gala) | External dependency | BSD-style license; the dependency source is excluded from the public experiment repository |
| `Swin-Unet/` | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) | `f48f623e226e25b6e395c37207915c50aaa9c776` plus experiment changes | No license file is published in the recorded upstream repository; obtain permission or publish changes through an authorized upstream fork |

## External Python Dependencies

The repository also uses packages such as PyTorch, torchvision,
segmentation-models-pytorch, Albumentations, OpenCV, NumPy, pandas,
scikit-image, and Matplotlib. These packages are installed separately and keep
their own licenses.

## Checkpoints

Model checkpoints are not covered by the source-code license. Download and use
them under the terms provided by their publishers. See [WEIGHTS.md](WEIGHTS.md)
for official sources.
