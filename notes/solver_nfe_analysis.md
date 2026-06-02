# Solver and NFE Analysis for CIFAR-10 Rectified Flow

## 1. Why NFE Matters

NFE is the number of model calls used while solving the sampling ODE. Larger NFE uses smaller integration steps and usually produces more stable samples, but sampling becomes slower.

## 2. Euler vs Heun

Euler uses one velocity estimate per step. Heun uses a predictor and a corrector velocity, so it is often smoother at the same step count but roughly doubles model evaluations per step.

## 3. Low-NFE Sampling

Low NFE tests whether the learned velocity field is easy to integrate. The main comparison is NFE 5, 10, 20, 50, and 100.

## 4. Guidance Scale

Velocity guidance uses:

$$
v_{\mathrm{cfg}} = v_{\mathrm{uncond}} + w(v_{\mathrm{cond}} - v_{\mathrm{uncond}}).
$$

Scale 0 is close to unconditional sampling, scale 1 is normal conditional sampling, and larger scales strengthen the class direction.

## 5. Runtime and Memory

The benchmark script records total sampling time, seconds per image, peak CUDA memory, model type, solver, NFE, guidance scale, checkpoint, and parameter count.

## 6. Observations

The CSV and figures under `results/cifar_flow/` and `figures/cifar_flow/` should be used for the final analysis after a full training run.

## 7. Failure Cases

Expected failure modes include noisy low-NFE samples, over-strong guidance artifacts, and weaker animal class structure in early training.
