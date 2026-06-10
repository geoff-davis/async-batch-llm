"""Generate the benchmark charts (and copy the cited JSON) for docs/benchmarks.md.

Reads the artifacts produced by ``example_batch_benchmark.py``:

    examples/data/benchmark_results/summary.json      (bake-off + wall-time race)
    examples/data/benchmark_results/throughput.json   (--throughput parity)

and writes, into ``docs/assets/``:

    benchmark-summary.json        (committed copy, cited by the docs)
    benchmark-throughput.json     (committed copy)
    benchmark-wall-time.png       (wall-time race: sequential vs gather vs abl)
    benchmark-cost.png            (per-provider cost bars, labelled with accuracy)
    benchmark-throughput.png      (throughput it/s: gather vs semaphore vs abl)

Accuracy is intentionally NOT charted — it's 95–97% across providers, too tight
to plot without being misleading; it lives in the bake-off table (and as a label
on the cost bars).

So the published charts/numbers are reproducible from a committed source of
truth. Requires matplotlib:

    uv sync --extra docs
    uv run python examples/generate_benchmark_charts.py
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "examples" / "data" / "benchmark_results"
ASSETS_DIR = ROOT / "docs" / "assets"

# Stable colour per orchestration / provider so the two charts read consistently.
_ORCH_COLORS = {"sequential": "#9e9e9e", "gather": "#90caf9", "async-batch-llm": "#1565c0"}
_DPI = 140


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        sys.exit(
            f"Missing {path}. Run the benchmark first:\n"
            "  uv run python examples/example_batch_benchmark.py\n"
            "  uv run python examples/example_batch_benchmark.py --throughput"
        )
    return json.loads(path.read_text())


def wall_time_chart(summary: dict, out: Path) -> None:
    """Grouped bars per provider: sequential vs naive gather vs async-batch-llm.

    Log y-axis because the sequential leg dwarfs the concurrent ones — the point
    is 'concurrency collapses wall time, and the framework matches a bare gather'.
    """
    rows = summary.get("wall_time_race", [])
    if not rows:
        print("No wall_time_race in summary.json; skipping wall-time chart.")
        return

    providers = [r["provider"] for r in rows]
    series = {
        "sequential": [r["sequential_s"] for r in rows],
        "gather": [r["gather_s"] for r in rows],
        "async-batch-llm": [r["async_batch_llm_s"] for r in rows],
    }

    x = range(len(providers))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, vals) in enumerate(series.items()):
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, vals, width, label=label, color=_ORCH_COLORS[label])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax.set_yscale("log")
    ax.set_ylabel("wall time (s, log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(providers)
    ax.set_title("Wall-time race — same workload, three orchestrations")
    ax.legend(title="orchestration", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"wrote {out}")


def cost_chart(summary: dict, out: Path) -> None:
    """Per-provider cost bars for the full test split, each labelled with its
    accuracy — so the chart shows the cost spread *and* that accuracy barely
    moves (no separate, misleadingly-tight accuracy chart)."""
    rows = sorted(summary.get("bakeoff", []), key=lambda r: r["cost_usd"])
    if not rows:
        print("No bakeoff in summary.json; skipping cost chart.")
        return

    labels = [r["provider"] for r in rows]
    costs = [r["cost_usd"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, costs, color="#1565c0", width=0.55)
    for bar, r in zip(bars, rows, strict=True):
        ax.annotate(
            f"${r['cost_usd']:.4f}\n{r['accuracy_pct']:.1f}% acc",
            (bar.get_x() + bar.get_width() / 2, r["cost_usd"]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("estimated cost for the full test split (USD)")
    ax.set_title("Cost for 1,319 problems — accuracy ~flat (95–97%), cost spans ~8×")
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"wrote {out}")


def throughput_chart(throughput: dict, out: Path) -> None:
    """Grouped bars per provider: chunked gather vs semaphore pool vs async-batch-llm
    items/sec, at the same worker count."""
    rows = throughput.get("rows", [])
    if not rows:
        print("No rows in throughput.json; skipping throughput chart.")
        return

    providers = [r["provider"] for r in rows]
    series = {
        "chunked gather": ("#9e9e9e", [r["chunked_gather"]["items_per_s"] for r in rows]),
        "semaphore pool": ("#90caf9", [r["semaphore_pool"]["items_per_s"] for r in rows]),
        "async-batch-llm": ("#1565c0", [r["async_batch_llm"]["items_per_s"] for r in rows]),
    }

    x = range(len(providers))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, (color, vals)) in enumerate(series.items()):
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, vals, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)

    ax.set_ylabel("throughput (items/sec)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(providers)
    ax.set_title("Throughput at scale — same workers; worker pool ≥ semaphore pool ≫ chunked")
    ax.legend(title="orchestration", frameon=False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load("summary.json")

    # Commit a copy of the source data the docs cite.
    shutil.copyfile(RESULTS_DIR / "summary.json", ASSETS_DIR / "benchmark-summary.json")

    wall_time_chart(summary, ASSETS_DIR / "benchmark-wall-time.png")
    cost_chart(summary, ASSETS_DIR / "benchmark-cost.png")

    throughput_path = RESULTS_DIR / "throughput.json"
    if throughput_path.exists():
        shutil.copyfile(throughput_path, ASSETS_DIR / "benchmark-throughput.json")
        throughput_chart(json.loads(throughput_path.read_text()), ASSETS_DIR / "benchmark-throughput.png")
    else:
        print("(no throughput.json yet — run with --throughput for the throughput chart)")


if __name__ == "__main__":
    main()
