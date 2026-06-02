import torch


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dit_checkpoint_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return {
        "image_size": args.get("image_size", 28),
        "patch_size": args.get("patch_size", 4),
        "in_channels": args.get("in_channels", 1),
        "hidden_dim": args.get("hidden_dim", 192),
        "depth": args.get("depth", 4),
        "num_heads": args.get("num_heads", 4),
        "mlp_ratio": args.get("mlp_ratio", 4.0),
        "num_classes": args.get("num_classes", 10),
        "null_label": args.get("null_label", 10),
    }
