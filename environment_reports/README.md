# Tested Cloud Environments

The software checklist requires the exact operating system, GPU, CUDA, Python,
PyTorch, and dependency versions used for the experiments. Generate one report
inside each activated cloud environment:

```bash
conda activate cnn_seg
bash scripts/collect_environment.sh cnn

conda activate matsam
bash scripts/collect_environment.sh matsam

conda activate hqsam
bash scripts/collect_environment.sh hqsam

conda activate sam2
bash scripts/collect_environment.sh sam2

conda activate microsam
bash scripts/collect_environment.sh microsam

conda activate swin_unet
bash scripts/collect_environment.sh swin_unet

conda activate mmseg
bash scripts/collect_environment.sh segformer
```

Each command writes a text report to this directory. Review reports for local
paths or private package indexes before committing them. The report does not
collect dataset paths, usernames, or the machine hostname.
