# Stage 4: MNIST Conditional Diffusion with CFG

Train:

```bash
conda activate fm_diffusion
python labs/lab4/train_mnist_cfg.py --epochs 20 --batch_size 128 --device cuda
```

Sample one class:

```bash
python labs/lab4/sample_mnist_cfg.py --class_id 3 --cfg_scale 4 --device cuda
```

Compare guidance scales:

```bash
python labs/lab4/eval_cfg_scales.py --scales 0 1 2 4 7 --device cuda
```
