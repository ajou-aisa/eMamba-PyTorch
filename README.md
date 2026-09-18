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

The original MARS repository is included as a Git submodule.

## Setup

```bash
git clone --recurse-submodules <repository-url>
cd eMamba

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

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
