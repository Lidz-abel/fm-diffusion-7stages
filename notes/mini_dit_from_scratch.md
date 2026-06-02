# Mini-DiT from Scratch

## 1. Components

Implemented files:

- `src/dit/patch_embed.py`
- `src/dit/embeddings.py`
- `src/dit/dit_block.py`
- `src/dit/model.py`
- `src/dit/utils.py`

## 2. Patchify / Unpatchify

For MNIST with `image_size=28` and `patch_size=4`:

```text
num_patches = 7 * 7 = 49
patch_dim = 1 * 4 * 4 = 16
```

The round trip satisfies:

```python
x_rec = unpatchify(patchify(x))
assert torch.allclose(x, x_rec)
```

## 3. DiTBlock

The implemented block uses:

```text
LayerNorm -> self-attention -> residual
LayerNorm -> MLP -> residual
```

with AdaLN-Zero conditioning:

```text
shift, scale, gate = Linear(SiLU(c))
```

## 4. MiniDiT

The model follows:

```text
x_t -> patchify -> patch embedding + position embedding
    -> time/class conditioning
    -> DiT blocks
    -> final patch prediction
    -> unpatchify -> eps_pred
```

## 5. Acceptance

- Patchify/unpatchify round trip works.
- `MiniDiT(x, t, y).shape == x.shape`.
- MNIST DDPM training script runs.
- CFG sampling script runs.
- U-Net vs DiT comparison report is generated.

## 6. Completion Table

| Item | Status | Deliverable |
| --- | --- | --- |
| Understand DiT vs U-Net | Done | `notes/dit_vs_unet.md` |
| Explain DiT in diffusion | Done | `notes/dit_in_diffusion.md` |
| MNIST conditional diffusion | Done | `labs/lab4/train_mnist_unet_cfg.py` |
| U-Net + CFG | Done | `figures/dit/cfg_scale_comparison.png` |
| DiTBlock from scratch | Done | `src/dit/dit_block.py` |
| Patchify / unpatchify | Done | `src/dit/patch_embed.py` |
| Time/class conditioning | Done | `src/dit/embeddings.py` |
| MNIST DiT diffusion | Done | `checkpoints/mnist_dit.pt` |
| U-Net vs DiT comparison | Done | `reports/unet_vs_dit_comparison.md` |
