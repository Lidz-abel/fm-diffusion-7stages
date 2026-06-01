# Stage 4: Guidance, Latent Diffusion, and Architectures

## 1. Goal

Stage 4 implements a minimal MNIST class-conditional DDPM with classifier-free guidance.

The model learns:

$$
\epsilon_\theta(x_t,t,y)\approx\epsilon
$$

where \(y\) is a digit label. During training, labels are randomly dropped to a `null_label`, so the same network learns both conditional and unconditional denoising.

## 2. Conditional Diffusion

Forward noising is unchanged from DDPM:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

The network receives:

- noisy image \(x_t\)
- timestep \(t\)
- class label \(y\)

and predicts the injected noise.

## 3. Label Conditioning

The U-Net uses:

$$
c(t,y)=\mathrm{TimeEmb}(t)+\mathrm{ClassEmb}(y).
$$

This conditioning vector is injected into residual blocks.

The unconditional class is represented by:

```text
null_label = 10
```

for MNIST labels `0` through `9`.

## 4. Classifier-Free Guidance

At sampling time:

$$
\epsilon_{\mathrm{cond}}=\epsilon_\theta(x_t,t,y)
$$

$$
\epsilon_{\mathrm{uncond}}=\epsilon_\theta(x_t,t,\varnothing)
$$

CFG combines them as:

$$
\epsilon_{\mathrm{cfg}}
=
\epsilon_{\mathrm{uncond}}
+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
$$

where \(s\) is the guidance scale.

Interpretation:

- `scale=0`: unconditional generation
- `scale=1`: normal conditional generation
- `scale=2/4`: stronger class conditioning
- `scale=7`: very strong conditioning, often less diverse

## 5. U-Net Architecture

The implemented MNIST U-Net follows:

```text
Input Conv
Down Block 1: 28x28 -> 14x14
Down Block 2: 14x14 -> 7x7
Middle Blocks
Up Block 1: 7x7 -> 14x14
Up Block 2: 14x14 -> 28x28
Output Conv
```

Each residual block receives the combined time/class embedding.

## 6. Latent Diffusion Pipeline

Latent diffusion changes the space where diffusion happens:

```text
image -> encoder -> latent -> diffusion model -> denoised latent -> decoder -> image
```

This reduces compute because the denoising network operates on lower-resolution latent tensors rather than pixels.

## 7. Stage 4 Outputs

- [ ] `src/time_embedding.py`
- [ ] `src/unet.py`
- [ ] `src/cfg_sampler.py`
- [ ] `src/mnist_dataset.py`
- [ ] `labs/lab4/train_mnist_cfg.py`
- [ ] `labs/lab4/sample_mnist_cfg.py`
- [ ] `labs/lab4/eval_cfg_scales.py`
- [ ] `figures/day4/cfg_scale_comparison.png`
- [ ] `figures/day4/class_cond_samples.png`
- [ ] `figures/day4/latent_diffusion_pipeline.png`

## 8. Check Questions

1. Why does CFG require unconditional training examples?
2. Why can class conditioning be added to time embedding?
3. Why does the U-Net output have the same shape as the input image?
4. What changes when CFG scale increases?
5. How is latent diffusion different from pixel-space diffusion?
