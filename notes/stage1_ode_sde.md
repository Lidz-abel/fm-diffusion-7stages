# Stage 1: ODE, SDE, Flow Models and Diffusion Models

## 1. Generation as Sampling

The goal of generative modeling is to sample from the data distribution:

$$
z \sim p_{\mathrm{data}}.
$$

A dataset consists of finite samples:

$$
z_1,\dots,z_N \sim p_{\mathrm{data}}.
$$

## 2. Flow Model

A flow model is defined by an ODE:

$$
dX_t = u_t^\theta(X_t)dt.
$$

The initial distribution is simple:

$$
X_0 \sim p_{\mathrm{init}}.
$$

The goal is:

$$
X_1 \sim p_{\mathrm{data}}.
$$

Euler update:

$$
X_{t+h}=X_t+h u_t^\theta(X_t).
$$

## 3. Diffusion Model

A diffusion model is defined by an SDE:

$$
dX_t = u_t^\theta(X_t)dt+\sigma_t dW_t.
$$

Euler-Maruyama update:

$$
X_{t+h}=X_t+h u_t^\theta(X_t)+\sigma_t\sqrt h \epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

## 4. Brownian Motion

Brownian motion satisfies:

$$
W_{t+h}-W_t\sim \mathcal N(0,hI).
$$

Thus:

$$
W_{t+h}-W_t=\sqrt h \epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

## 5. ODE vs SDE

| Model | Equation | Randomness |
|---|---|---|
| Flow Model | \(dX_t=u_t^\theta(X_t)dt\) | initial \(X_0\) only |
| Diffusion Model | \(dX_t=u_t^\theta(X_t)dt+\sigma_t dW_t\) | initial \(X_0\) and Brownian noise |

## 6. Stage 1 Outputs

- [ ] `src/toy_data.py`
- [ ] `src/ode.py`
- [ ] `src/sde.py`
- [ ] `src/visualization.py`
- [ ] `labs/lab1/run_probability_path.py`
- [ ] `labs/lab1/run_ode_sde.py`
- [ ] `figures/stage1/probability_path_t_*.png`
- [ ] `figures/stage1/brownian_motion.png`
- [ ] `figures/stage1/ode_trajectories.png`
- [ ] `figures/stage1/sde_trajectories.png`

## 7. Reflection

Questions:

1. Why is generation modeled as sampling?
2. Why is a flow model deterministic after \(X_0\) is fixed?
3. Why does an SDE remain random after \(X_0\) is fixed?
4. Why does Euler-Maruyama contain the term \(\sqrt h\epsilon_t\)?
5. Why does \(\sigma_t=0\) reduce a diffusion model to a flow model?