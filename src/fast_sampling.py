import torch


@torch.no_grad()
def sample_with_euler(model, x_init: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Euler ODE sampler for Flow Matching NFE comparison.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    x = x_init.clone()
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full(
            (x.shape[0],),
            i / num_steps,
            device=x.device,
            dtype=x.dtype,
        )
        v = model(x, t)
        x = x + dt * v

    return x


@torch.no_grad()
def run_nfe_ablation(model, x_init: torch.Tensor, nfe_list: list[int]) -> dict[int, torch.Tensor]:
    """
    Run sampling with different numbers of function evaluations.
    """
    results = {}
    for nfe in nfe_list:
        results[nfe] = sample_with_euler(model=model, x_init=x_init, num_steps=nfe)
    return results
