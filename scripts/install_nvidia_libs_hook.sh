#!/bin/bash
# Add nvidia sub-package library paths so torch can find libcudart.so.13
# etc. on first import. Conda pip-installs nvidia-* into
# site-packages/nvidia/<sub>/lib/ but does NOT update LD_LIBRARY_PATH.
#
# Idempotent: safe to re-run. Run once after the first
# `pip install torch torchvision torchaudio` in any conda env that has
# a CUDA-enabled torch wheel. Without this, every fresh shell / SSH
# session fails `import torch` with:
#   OSError: libcudart.so.13: cannot open shared object file
for sub in cu13 cuda_runtime cudnn cublas cufft cusparse curand cusolver cudnn_frontend nvjitlink; do
    p="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/$sub/lib"
    [ -d "$p" ] && export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
done