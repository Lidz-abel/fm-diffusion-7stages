# DiT in Diffusion

## 1. Pipeline Position

DiT is the denoising network inside a diffusion pipeline.

```text
noise image x_t
    -> patchify
patch tokens
    -> DiT Transformer
predicted noise patches
    -> unpatchify
predicted noise eps_theta(x_t, t, y)
    -> DDPM / DDIM sampler
x_{t-1}
```

## 2. Key Point

DiT is not a new diffusion objective.

DDPM, Flow Matching, and Rectified Flow define training targets or dynamics. U-Net, DiT, and FlowDCN are network architectures that implement the predictor.

## 3. Training Target

In this Mini-DiT lab, the objective remains DDPM noise prediction:

$$
\epsilon_\theta(x_t,t,y)\approx\epsilon.
$$

The model input/output shape is:

```text
x_t:      [B, 1, 28, 28]
t:        [B]
y:        [B]
eps_pred: [B, 1, 28, 28]
```

## 4. Conditioning

Mini-DiT uses:

```text
c = time_embedding(t) + class_embedding(y)
```

and injects `c` through AdaLN-Zero in each DiT block.
