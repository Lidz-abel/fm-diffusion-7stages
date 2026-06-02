from pathlib import Path

import matplotlib.pyplot as plt


def save_unified_framework(save_path: str | Path) -> None:
    """
    Save a compact unified framework diagram.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    nodes = {
        "Generative Modeling": (0.5, 0.92),
        "Continuous State Space": (0.28, 0.72),
        "Discrete State Space": (0.72, 0.72),
        "Flow Matching\nvelocity field": (0.12, 0.48),
        "DDPM\nnoise prediction": (0.28, 0.48),
        "Score-SDE\nscore function": (0.44, 0.48),
        "CFG\nconditional direction": (0.28, 0.28),
        "Consistency\nflow map idea": (0.28, 0.13),
        "Transition matrix": (0.62, 0.48),
        "Mask corruption": (0.78, 0.48),
        "Absorbing state": (0.62, 0.28),
        "Reverse denoising": (0.78, 0.28),
    }

    for text, (x, y) in nodes.items():
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5", edgecolor="#555555"),
            transform=ax.transAxes,
        )

    edges = [
        ("Generative Modeling", "Continuous State Space"),
        ("Generative Modeling", "Discrete State Space"),
        ("Continuous State Space", "Flow Matching\nvelocity field"),
        ("Continuous State Space", "DDPM\nnoise prediction"),
        ("Continuous State Space", "Score-SDE\nscore function"),
        ("DDPM\nnoise prediction", "CFG\nconditional direction"),
        ("Flow Matching\nvelocity field", "Consistency\nflow map idea"),
        ("Discrete State Space", "Transition matrix"),
        ("Discrete State Space", "Mask corruption"),
        ("Mask corruption", "Absorbing state"),
        ("Absorbing state", "Reverse denoising"),
    ]

    for src, dst in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        ax.annotate(
            "",
            xy=(x1, y1 + 0.05),
            xytext=(x0, y0 - 0.05),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_continuous_vs_discrete(save_path: str | Path) -> None:
    """
    Save a comparison diagram between continuous and discrete diffusion.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axes:
        ax.axis("off")

    axes[0].set_title("Continuous Diffusion")
    axes[0].text(
        0.5,
        0.62,
        r"$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$",
        ha="center",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef5ff", edgecolor="#446688"),
        transform=axes[0].transAxes,
    )
    axes[0].text(
        0.5,
        0.32,
        "state is real-valued\nnoise is Gaussian\nreverse model denoises vectors",
        ha="center",
        va="center",
        fontsize=11,
        transform=axes[0].transAxes,
    )

    axes[1].set_title("Discrete Diffusion")
    axes[1].text(
        0.5,
        0.62,
        "token -> token or [MASK]\nq(x_t | x_0) is a corruption transition",
        ha="center",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff4e8", edgecolor="#886644"),
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.5,
        0.32,
        "state is categorical\nnoise is a transition process\nreverse model predicts tokens",
        ha="center",
        va="center",
        fontsize=11,
        transform=axes[1].transAxes,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
