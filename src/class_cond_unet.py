from __future__ import annotations

from .image_unet import CIFAR10FlowUNet


class ClassConditionalUNet(CIFAR10FlowUNet):
    """
    Generic class-conditional image U-Net.

    This class intentionally inherits the existing CIFAR10FlowUNet implementation
    without wrapping it in a submodule, so old CIFAR / Visual checkpoints keep the
    same state_dict key names. The historical CIFAR10FlowUNet name is now treated
    as a 32/64/128-capable class-conditional U-Net implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        time_dim: int = 512,
        num_classes: int = 10,
        null_label: int | None = None,
        use_attention: bool = True,
        time_scale: float = 1.0,
        image_size: int = 32,
        attention_resolutions: list[int] | tuple[int, ...] | None = None,
    ):
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8 for the current U-Net.")
        if null_label is None:
            null_label = num_classes
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            emb_dim=time_dim,
            num_classes=num_classes,
            null_label=null_label,
            use_attention=use_attention,
            time_scale=time_scale,
            image_size=image_size,
            attention_resolutions=attention_resolutions,
        )


def build_class_cond_unet_from_args(args: dict) -> ClassConditionalUNet:
    """
    Build a ClassConditionalUNet from checkpoint or config args.
    """
    num_classes = int(args.get("num_classes", 10))
    return ClassConditionalUNet(
        in_channels=int(args.get("in_channels", 3)),
        base_channels=int(args.get("base_channels", 128)),
        time_dim=int(args.get("time_dim", args.get("emb_dim", 512))),
        num_classes=num_classes,
        null_label=int(args.get("null_label", num_classes)),
        use_attention=bool(args.get("use_attention", True)),
        time_scale=float(args.get("time_scale", 1.0)),
        image_size=int(args.get("image_size", 32)),
        attention_resolutions=args.get("attention_resolutions"),
    )
