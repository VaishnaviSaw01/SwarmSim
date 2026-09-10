"""Tests for report.py -- chart rendering (in-memory and to-disk)."""

from pathlib import Path

from report import generate_chart, render_chart_png


def _make_log():
    return [
        {"round": r, "agent_id": a, "opinion_score": 0.1 * a - 0.05 * r}
        for r in range(3)
        for a in range(4)
    ]


def test_render_chart_png_returns_valid_png_bytes():
    png_bytes = render_chart_png(_make_log())
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")  # PNG magic number
    assert len(png_bytes) > 0


def test_generate_chart_writes_the_same_bytes_to_disk(tmp_path: Path):
    out_path = tmp_path / "nested" / "chart.png"
    result_path = generate_chart(_make_log(), out_path)

    assert result_path == out_path
    assert out_path.is_file()
    assert out_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
