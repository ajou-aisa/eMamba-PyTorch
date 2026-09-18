# eMamba PyTorch

PyTorch reimplementation of the MARS configuration from  
**eMamba: Efficient Acceleration Framework for Mamba Models in Edge Computing**.

The goal of this project is to reproduce the FP32 eMamba model on the
public MARS dataset and compare the result with the accuracy reported in
the paper.

## Model Configuration

MARS configuration reported in the eMamba paper:

| Parameter | Value |
|---|---:|
| Model dimension (`D`) | 20 |
| Expansion factor (`E`) | 2 |
| Patch size (`P`) | 2 |
| Number of Mamba blocks (`M`) | 2 |
| State dimension (`N`) | 8 |
| Output dimension | 57 |

High-level architecture:

```text
MARS Input [8, 8, 5]
        │
        ▼
Patch Embedding (P=2)
        │
        ▼
16 Tokens × D=20
        │
        ▼
eMamba Block × 2
        │
        ▼
Output Head
        │
        ▼
57 values
(19 joints × XYZ)
````

The paper reports an FP32 MARS RMSE of approximately **7.85 cm**. 

## Repository Structure

```text
.
├── datasets/
│   └── mars.py
├── models/
│   ├── emamba.py
│   ├── emamba_block.py
│   ├── range_norm.py
│   └── selective_ssm.py
├── third_party/
│   └── MARS/
├── requirements.txt
├── README.md
└── train.py
```

## Setup

### 1. Clone the repository

Clone this repository together with the MARS dataset submodule:

```bash
git clone --recurse-submodules <repository-url>
cd eMamba
````

If you already cloned the repository without the submodule, initialize it manually:

```bash
git submodule update --init --recursive
```

After this step, the MARS dataset should be available under:

```text
third_party/MARS/
```

You can check that the dataset files exist with:

```bash
ls third_party/MARS/feature
```

You should see files such as:

```text
featuremap_train.npy
featuremap_validate.npy
featuremap_test.npy
labels_train.npy
labels_validate.npy
labels_test.npy
```

### 2. Create a Python virtual environment

Python 3.11 is recommended.

Check your Python version:

```bash
python3.11 --version
```

Create a virtual environment:

```bash
python3.11 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, your shell prompt should show something like:

```text
(.venv)
```

You can verify that the virtual environment is active with:

```bash
which python
python --version
```

### 3. Install dependencies

Upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Verify the environment

Run the following command:

```bash
python - <<'PY'
import torch
import numpy as np

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
PY
```

On Apple Silicon Macs, `MPS available: True` means PyTorch can use the Apple GPU.

### 5. Verify the MARS dataset loader

Run a simple check:

```bash
python - <<'PY'
from datasets.mars import MARSDataset

dataset = MARSDataset(
    "third_party/MARS/feature/featuremap_train.npy",
    "third_party/MARS/feature/labels_train.npy",
)

x, y = dataset[0]

print("Dataset size:", len(dataset))
print("Input shape:", x.shape)
print("Input dtype:", x.dtype)
print("Label shape:", y.shape)
print("Label dtype:", y.dtype)
PY
```

Expected output:

```text
Dataset size: 24066
Input shape: torch.Size([8, 8, 5])
Input dtype: torch.float32
Label shape: torch.Size([57])
Label dtype: torch.float32
```

At this point, the development environment and MARS dataset are ready.

## Status

* [x] MARS dataset loader
* [x] High-level model structure
* [ ] eMamba block implementation
* [ ] Selective SSM implementation
* [ ] FP32 training
* [ ] Evaluation

## References

* J. Kim et al., *eMamba: Efficient Acceleration Framework for Mamba Models in Edge Computing*
* S. An and U. Y. Ogras, *MARS: mmWave-based Assistive Rehabilitation System for Smart Healthcare*
