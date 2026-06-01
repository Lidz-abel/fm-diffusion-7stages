# Stage 2: Flow Matching

## 1. Goal

Stage 2 studies Flow Matching, which trains a neural vector field by regression.

The goal is to learn:

$$
v_\theta(x,t) \approx u_t^{\mathrm{target}}(x)
$$

so that the ODE

$$
dX_t = v_\theta(X_t,t)dt
$$

transports a simple prior distribution to the data distribution.

## 2. Straight Probability Path

We use the straight-line path:

$$
x_t = (1-t)x_0 + tz
$$

where

$$
x_0 \sim \mathcal N(0,I),
\qquad
z \sim p_{\mathrm{data}}.
$$

## 3. Target Velocity

The target velocity is:

$$
u_t = \frac{d}{dt}x_t = z - x_0.
$$

Therefore, the supervised learning target is:

$$
v_\theta(x_t,t) \approx z-x_0.
$$

## 4. Flow Matching Loss

The loss is:

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E_{t,x_0,z}
\left[
\|v_\theta(x_t,t)-(z-x_0)\|^2
\right].
$$

## 5. Sampling

After training, we sample by solving:

$$
dX_t = v_\theta(X_t,t)dt,
\qquad
X_0\sim \mathcal N(0,I).
$$

Using Euler method:

$$
X_{t+h}=X_t+h v_\theta(X_t,t).
$$

## 6. Simulation-Free Training

During training, we do not solve the ODE.

Instead, we directly sample:

$$
x_0\sim \mathcal N(0,I),\quad
z\sim p_{\mathrm{data}},\quad
t\sim U(0,1)
$$

and construct:

$$
x_t=(1-t)x_0+tz.
$$

This is why Flow Matching training is simulation-free.

## 7. Stage 2 Outputs

- [ ] `src/mlp.py`
- [ ] `src/flow_matching.py`
- [ ] `src/samplers.py`
- [ ] `labs/lab2/run_2d_flow_matching.py`
- [ ] `figures/stage2/fm_samples_nfe_compare.png`
- [ ] `figures/stage2/fm_vector_field_t_*.png`
- [ ] `figures/stage2/fm_trajectories.png`

## 8. Check Questions

1. What is a probability path?
2. What is a conditional probability path?
3. What is a marginal probability path?
4. What is a conditional vector field?
5. What is a marginal vector field?
6. Why can we train with conditional vector field targets?
7. What does simulation-free training mean?
8. What is the difference between training and sampling?
