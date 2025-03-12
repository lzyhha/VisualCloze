# Installation

Downloading VisualCloze repo from github:

```bash
git clone xxx
```

### 1. Create a conda environment and install PyTorch

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

```bash
conda create -n visualcloze -y
conda activate visualcloze
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install flash-attn

```bash
pip install flash-attn --no-build-isolation
```