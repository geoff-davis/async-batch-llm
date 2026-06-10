"""Generate the benchmark charts (and copy the cited JSON) for docs/benchmarks.md.

Reads the artifacts produced by ``example_batch_benchmark.py``:

    examples/data/benchmark_results/summary.json      (bake-off + wall-time race)
    examples/data/benchmark_results/throughput.json   (--throughput parity)

and writes, into ``docs/assets/``:

    benchmark-summary.json        (committed copy, cited by the docs)
    benchmark-throughput.json     (committed copy)
    benchmark-wall-time.png       (wall-time bars: sequential vs gather vs abl)
    benchmark-cost-accuracy.png   (cost-vs-accuracy scatter for the bake-off)

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


def cost_accuracy_chart(summary: dict, out: Path) -> None:
    """Scatter: estimated cost ($) vs accuracy (%), one labelled point per provider."""
    rows = summary.get("bakeoff", [])
    if not rows:
        print("No bakeoff in summary.json; skipping cost-accuracy chart.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        ax.scatter(r["cost_usd"], r["accuracy_pct"], s=90, zorder=3)
        ax.annotate(
            f"{r['provider']}\n{r['model_id']}",
            (r["cost_usd"], r["accuracy_pct"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )

    ax.set_xlabel("estimated cost for the full test split (USD)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Cost vs. accuracy — same framework, one strategy swap per provider")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load("summary.json")

    # Commit a copy of the source data the docs cite.
    shutil.copyfile(RESULTS_DIR / "summary.json", ASSETS_DIR / "benchmark-summary.json")
    throughput_path = RESULTS_DIR / "throughput.json"
    if throughput_path.exists():
        shutil.copyfile(throughput_path, ASSETS_DIR / "benchmark-throughput.json")
        print(f"copied throughput.json -> {ASSETS_DIR / 'benchmark-throughput.json'}")
    else:
        print("(no throughput.json yet — run with --throughput to include it)")

    wall_time_chart(summary, ASSETS_DIR / "benchmark-wall-time.png")
    cost_accuracy_chart(summary, ASSETS_DIR / "benchmark-cost-accuracy.png")


if __name__ == "__main__":
    main()
