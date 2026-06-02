# DiT vs U-Net

## 1. Why U-Net Works Well for Diffusion

U-Net is a natural denoising architecture because it processes image feature maps at multiple resolutions. Downsampling captures global context, upsampling recovers spatial detail, and skip connections preserve local structure.

## 2. Why DiT Can Replace U-Net

DiT treats image patches as tokens and uses Transformer blocks as the denoising network. The diffusion objective is unchanged; only the predictor architecture changes.

## 3. Core Comparison

| Dimension | U-Net Diffusion | DiT Diffusion |
| --- | --- | --- |
| Input form | image feature map | patch token |
| Core module | Conv / ResBlock / Attention | Transformer Block |
| Time conditioning | time embedding added to ResBlock | time embedding modulates Transformer |
| Class conditioning | class embedding / cross-attention | class embedding / AdaLN |
| Inductive bias | strong locality, translation equivariance | weaker image bias, more data-dependent |
| Best setting | small and medium image generation | large-scale generation and scalable modeling |

## 4. Small vs Large Data

On small datasets such as MNIST, U-Net often trains more easily because convolution provides useful image priors. DiT has a more general token interface, but usually needs larger data/model scale to show its advantage.

## 5. Why Modern Large Models Use Transformers

Transformers scale well with model size and data, support unified tokenization, and can share architectural ideas across image, video, text, and multimodal generation.
