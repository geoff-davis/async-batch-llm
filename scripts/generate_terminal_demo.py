"""Generate the deterministic README terminal-demo GIF.

Maintainer command:

    uv run --with pillow python scripts/generate_terminal_demo.py

The rendered transcript mirrors the credential-free callable application
example. Pillow is a generation-time tool, not a package dependency.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "v0.20-quickstart.gif"

FRAMES = [
    [
        "$ uv run python examples/example_callable_application.py",
        "batch  0/5 items",
    ],
    [
        "$ uv run python examples/example_callable_application.py",
        "batch  1/5 items",
        "retry doc-1: billed response failed validation",
    ],
    [
        "$ uv run python examples/example_callable_application.py",
        "batch  3/5 items",
        "retry doc-1: recovered with item-local feedback",
        "retry doc-3: billed response failed validation",
    ],
    [
        "$ uv run python examples/example_callable_application.py",
        "batch  5/5 items",
        "",
        "Items:     5 total — 5 succeeded, 0 failed",
        "Stopped:   completed",
        "Retries:   2 extra attempt(s) across 2 item(s)",
        "Tokens:    in 26 (cached 0) · out 14",
    ],
    [
        "$ uv run python examples/example_callable_application.py",
        "batch  5/5 items",
        "",
        "Items:     5 total — 5 succeeded, 0 failed",
        "Stopped:   completed",
        "Retries:   2 extra attempt(s) across 2 item(s)",
        "Tokens:    in 26 (cached 0) · out 14",
        "completion order: ['doc-2', 'doc-4', 'doc-1', 'doc-3', 'doc-0']",
        "resume client calls: 0",
    ],
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _render(lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (1040, 520), "#10151c")
    draw = ImageDraw.Draw(image)
    title_font = _font(21)
    body_font = _font(19)
    draw.rounded_rectangle((10, 10, 1030, 510), radius=14, fill="#17202a", outline="#34495e")
    draw.ellipse((30, 29, 45, 44), fill="#ff5f57")
    draw.ellipse((53, 29, 68, 44), fill="#febc2e")
    draw.ellipse((76, 29, 91, 44), fill="#28c840")
    draw.text((112, 27), "async-batch-llm · local demo", font=title_font, fill="#c9d1d9")
    y = 76
    for index, line in enumerate(lines):
        color = "#7ee787" if index == 0 else "#d6deeb"
        if line.startswith("retry"):
            color = "#f2cc60"
        if line.startswith("resume"):
            color = "#79c0ff"
        draw.text((34, y), line, font=body_font, fill=color)
        y += 40
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    images = [_render(lines) for lines in FRAMES]
    images[0].save(
        OUTPUT,
        save_all=True,
        append_images=images[1:],
        duration=[700, 750, 850, 1_200, 2_500],
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
