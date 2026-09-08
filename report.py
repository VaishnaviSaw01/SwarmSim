"""
report.py -- renders a simulation run to disk: a PNG line chart of every
agent's opinion trajectory, and a JSON report bundling the run's
parameters, seed info, and evaluation metrics.

WHY matplotlib and plain JSON: both are inspectable, run fully offline,
and don't introduce any dependency the rest of the project doesn't already
need to explain. No templating engine, no report framework -- just a
chart and a dict.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this runs inside a FastAPI request, not a GUI
import matplotlib.pyplot as plt


def generate_chart(
    log: list[dict],
    out_path: str | Path,
    title: str = "Opinion score per agent per round",
) -> Path:
    """Plot every agent's opinion_score across rounds as one line each.

    WHY one line per agent rather than a smoothed aggregate: for a project
    about *inspecting* agent behavior, seeing individual trajectories (who
    moved, who held out, who oscillated) is more useful than a summary
    band -- and at the agent counts this project targets (tens of agents),
    the plot is still readable.

    Args:
        log: round-by-round log from simulate.run_simulation.
        out_path: where to save the PNG.
        title: chart title.

    Returns:
        The resolved Path the chart was written to.
    """
    by_agent: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in log:
        by_agent[row["agent_id"]].append((row["round"], row["opinion_score"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    num_agents = len(by_agent)

    for i, (agent_id, points) in enumerate(sorted(by_agent.items())):
        points.sort(key=lambda p: p[0])
        rounds, scores = zip(*points)
        ax.plot(rounds, scores, color=cmap(i / max(num_agents - 1, 1)), alpha=0.7, linewidth=1)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Round")
    ax.set_ylabel("Opinion score (-1 to 1)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_report(
    topic: str,
    params: dict,
    seed_info: dict,
    metrics: dict,
    chart_path: str | Path,
    out_json_path: str | Path,
) -> Path:
    """Write the full JSON report (params + seed info + eval metrics) to disk.

    Kept as one flat-ish dict rather than a Pydantic model: this file is a
    write-once artifact for humans and for main.py's HTTP response, not
    something that needs its own validated schema of its own.
    """
    report = {
        "topic": topic,
        "params": params,
        "seed": seed_info,
        "metrics": metrics,
        "chart_path": str(chart_path),
    }

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, indent=2))
    return out_json_path
