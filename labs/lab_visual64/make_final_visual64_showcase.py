from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_path",
        type=str,
        default=str(ROOT / "figures" / "visual64" / "visual64_best_binary_pets_heun80_cfg2.png"),
    )
    parser.add_argument(
        "--fine_path",
        type=str,
        default=str(ROOT / "figures" / "visual64" / "visual64_best_37class_pets_heun80_cfg2.png"),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(ROOT / "figures" / "visual64" / "final_visual64_showcase_panel.png"),
    )
    return parser.parse_args()


def add_title(image: Image.Image, title: str, width: int) -> Image.Image:
    title_h = 48
    canvas = Image.new("RGB", (width, image.height + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, 12), title, fill=(20, 20, 20), font=font)
    canvas.paste(image, ((width - image.width) // 2, title_h))
    return canvas


def main():
    args = parse_args()
    binary = Image.open(args.binary_path).convert("RGB")
    fine = Image.open(args.fine_path).convert("RGB")

    width = max(binary.width, fine.width)
    binary_panel = add_title(binary, "Visual64 EDM Pets: binary cats vs dogs", width)
    fine_panel = add_title(fine, "Visual64 EDM Pets: 37-class fine-grained samples", width)

    gap = 24
    canvas = Image.new("RGB", (width, binary_panel.height + gap + fine_panel.height), "white")
    canvas.paste(binary_panel, (0, 0))
    canvas.paste(fine_panel, (0, binary_panel.height + gap))

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    print(f"Saved final Visual64 showcase panel to: {save_path}")


if __name__ == "__main__":
    main()
