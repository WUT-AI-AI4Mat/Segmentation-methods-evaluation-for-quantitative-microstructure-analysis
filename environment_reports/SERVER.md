# Tested Cloud Server

The full benchmark was run and checked on the following Linux cloud server.
The values below describe the tested platform; they are not minimum hardware
requirements.

| Component | Tested configuration |
| --- | --- |
| Operating system | Ubuntu 22.04.3 LTS (Jammy), x86_64 |
| CPU | Intel Xeon Platinum 8368Q at 2.60 GHz, 152 logical CPUs |
| System memory | 251 GiB |
| GPU | 2 x NVIDIA GeForce RTX 4090, 24,564 MiB each |
| NVIDIA driver | 580.82.09 |
| Driver-supported CUDA version | 13.0 |

The login shell did not expose an `nvcc` executable. The conda environments
use CUDA-enabled PyTorch wheels with CUDA 12.1 or 12.8, as listed in the
environment table in the repository README. The driver-supported CUDA version
reported by `nvidia-smi` is therefore not the CUDA runtime used by every model.

The server information was collected on 25 August 2026. Current memory, swap,
and disk utilization were intentionally omitted because they describe server
load at collection time rather than software requirements.
