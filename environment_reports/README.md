# Tested Cloud Environments

This directory records the exact software environments used on the tested
cloud server. See [`SERVER.md`](SERVER.md) for the operating system and
hardware configuration.

| Models | Conda environment | Complete package list |
| --- | --- | --- |
| U-Net and DeepLabV3+ | `cnn` | [`cnn.txt`](cnn.txt) |
| MatSAM | `matsam` | [`matsam.txt`](matsam.txt) |
| HQ-SAM | `hqsam` | [`hqsam.txt`](hqsam.txt) |
| SAM2 | `sam2` | [`sam2.txt`](sam2.txt) |
| micro-sam | `microsam` | [`microsam.txt`](microsam.txt) |
| Swin-Unet | `swinunet` | [`swinunet.txt`](swinunet.txt) |
| SegFormer | `segformer` | [`segformer.txt`](segformer.txt) |

The CNN package list was captured from the original environment named `sam`;
the public reproduction environment is named `cnn` to reflect its actual
purpose. The package lists were generated with
`python -m pip list --format=freeze` on
25 August 2026. They contain package names and installed versions only; local
paths, usernames, hostnames, dataset paths, and package-index credentials are
not included.

To capture a fresh report after changing an environment, activate it and run:

```bash
bash scripts/collect_environment.sh ENVIRONMENT_NAME
```

Review generated reports for private paths or package indexes before
committing them.
