from __future__ import annotations

from pathlib import Path
import argparse
import json

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a Figure-11-style side-by-side CFG comparison panel."
    )
    parser.add_argument("--left", type=str, required=True, help="Image grid for weak/no CFG.")
    parser.add_argument("--right", type=str, required=True, help="Image grid for strong CFG.")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--class_name", type=str, default="Oxford-IIIT Pets classes")
    parser.add_argument("--left_title", type=str, default="without classifier-free guidance, w=0")
    parser.add_argument("--right_title", type=str, default="classifier-free guidance, w=4")
    parser.add_argument("--title", type=str, default="Visual64 class-based classifier-free guidance")
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Same checkpoint, class ids, seed, solver, and number of sampling steps. "
            "Increasing guidance strengthens class adherence, but very large guidance can reduce diversity."
        ),
    )
    parser.add_argument("--metadata_out", type=str, default=None)
    parser.add_argument("--margin", type=int, default=34)
    parser.add_argument("--gap", type=int, default=28)
    parser.add_argument("--header_height", type=int, default=112)
    parser.add_argument("--footer_height", type=int, default=78)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, target_width: int) -> Image.Image:
    if image.width == target_width:
        return image
    scale = target_width / image.width
    target_height = max(1, round(image.height * scale))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((target_width, target_height), resample=resampling)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (25, 25, 25),
) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def main() -> None:
    args = parse_args()
    left_path = Path(args.left)
    right_path = Path(args.right)
    save_path = Path(args.save_path)
    if not left_path.exists():
        raise FileNotFoundError(left_path)
    if not right_path.exists():
        raise FileNotFoundError(right_path)

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    target_width = max(left.width, right.width)
    left = fit_image(left, target_width)
    right = fit_image(right, target_width)
    panel_height = max(left.height, right.height)

    margin = args.margin
    gap = args.gap
    header_height = args.header_height
    footer_height = args.footer_height
    label_height = 46

    canvas_width = margin * 2 + target_width * 2 + gap
    canvas_height = margin + header_height + label_height + panel_height + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28, bold=True)
    subtitle_font = load_font(17, bold=False)
    label_font = load_font(18, bold=True)
    caption_font = load_font(15, bold=False)

    draw_centered_text(
        draw,
        (margin, margin, canvas_width - margin, margin + 40),
        args.title,
        title_font,
        fill=(20, 38, 70),
    )
    draw_centered_text(
        draw,
        (margin, margin + 42, canvas_width - margin, margin + 74),
        f'class y = "{args.class_name}"',
        subtitle_font,
        fill=(65, 65, 65),
    )

    left_x = margin
    right_x = margin + target_width + gap
    label_top = margin + header_height
    image_top = label_top + label_height

    draw.rounded_rectangle(
        (left_x, label_top + 2, left_x + target_width, label_top + label_height - 5),
        radius=10,
        fill=(232, 238, 248),
        outline=(65, 88, 130),
        width=2,
    )
    draw.rounded_rectangle(
        (right_x, label_top + 2, right_x + target_width, label_top + label_height - 5),
        radius=10,
        fill=(234, 246, 238),
        outline=(54, 112, 75),
        width=2,
    )
    draw_centered_text(draw, (left_x, label_top, left_x + target_width, label_top + label_height), args.left_title, label_font)
    draw_centered_text(draw, (right_x, label_top, right_x + target_width, label_top + label_height), args.right_title, label_font)

    canvas.paste(left, (left_x, image_top))
    canvas.paste(right, (right_x, image_top))
    draw.rectangle((left_x, image_top, left_x + target_width, image_top + left.height), outline=(220, 220, 220), width=1)
    draw.rectangle((right_x, image_top, right_x + target_width, image_top + right.height), outline=(220, 220, 220), width=1)

    caption_top = image_top + panel_height + 20
    lines = wrap_text(draw, args.caption, caption_font, canvas_width - 2 * margin)
    for idx, line in enumerate(lines[:3]):
        draw_centered_text(
            draw,
            (margin, caption_top + idx * 20, canvas_width - margin, caption_top + (idx + 1) * 20),
            line,
            caption_font,
            fill=(70, 70, 70),
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)

    metadata = {
        "left": str(left_path),
        "right": str(right_path),
        "save_path": str(save_path),
        "class_name": args.class_name,
        "left_title": args.left_title,
        "right_title": args.right_title,
        "title": args.title,
        "caption": args.caption,
    }
    metadata_out = Path(args.metadata_out) if args.metadata_out else save_path.with_suffix(".json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved CFG comparison panel to: {save_path}")
    print(f"Saved metadata to: {metadata_out}")


if __name__ == "__main__":
    main()
