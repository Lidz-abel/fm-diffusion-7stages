# U-Net vs DiT Comparison

## Summary

On MNIST, U-Net generally has a stronger convolutional inductive bias and is easier to train with small models. Mini-DiT is more architecture-general, but typically benefits from more data, more compute, and larger model scale.

## Measured Smoke-Test Run

- U-Net parameters: `2,155,649`
- Mini-DiT parameters: `26,128`
- U-Net sampling time: `0.3219s`
- Mini-DiT sampling time: `0.0678s`
- CFG scale: `4.0`
- Timesteps: `10`

## Interpretation

U-Net processes image feature maps with convolutional locality and skip connections. DiT patchifies the image and denoises patch tokens using Transformer blocks with time/class conditioning. DiT's main advantage is scalability and a unified token modeling interface.
