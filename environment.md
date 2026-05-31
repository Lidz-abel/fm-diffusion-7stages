# Environment

## Basic

- Python: 3.10
- Main packages: PyTorch, NumPy, Matplotlib, tqdm, Jupyter

## Setup

```bash
conda create -n fm_diffusion python=3.10 -y
conda activate fm_diffusion
pip install -r requirements.txt
```

## Notes

Stage 1 experiments can run on CPU.
Later stages involving MNIST or FlowDCN may require GPU.
