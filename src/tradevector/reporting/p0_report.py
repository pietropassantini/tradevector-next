"""P0 Report generator."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_p0_report(
    hypothesis: dict,
    probe_results: dict,
    random_results: dict,
    verdict: str,
    symbol: str,
    timeframe: str,
    signal_column: str,
    output_dir: Path,
) -> Path:
    md_path = output_dir / f"{symbol}_{timeframe}_signal_probe.md"

    hyp_id = hypothesis.get("id", "unknown")
    hyp_name = hypothesis.get("name", "Unknown")

    lines = [
        f"# P0 Statistical Signal Probe — {hyp_name}",
        "",
        f"**Hypothesis:** {hyp_id}",
        f"**Symbol:** {symbol}",
        f"**Timeframe:** {timeframe}",
        f"**Signal column:** {signal_column}",
        "",
        f"## Hypothesis",
        "",
        f"- **What:** {hypothesis.get('event_predicted', 'N/A')}",
        f"- **Why:** {hypothesis.get('why_should_work', 'N/A')}",
        f"- **Assets:** {', '.join(hypothesis.get('assets', []))}",
        f"- **Horizons:** {', '.join(hypothesis.get('horizons', []))}",
        "",
        "## Signal Probe Results",
        "",
    ]

    full_metrics = probe_results.get("full_period", {})
    lines.append("| Horizon | N | Lead Corr | Mean Fwd Ret | Gross Exp | Median Gross | Hit Rate | Top Q Ret |")
    lines.append("|---------|---|-----------|-------------|-----------|-------------|----------|-----------|")
    for h_key in sorted(full_metrics.keys()):
        m = full_metrics[h_key]
        if isinstance(m, dict) and "error" not in m:
            lines.append(
                f"| {h_key} | {m['sample_size']} | {m['lead_correlation']:.4f} | "
                f"{m['mean_forward_return']:.6f} | {m['gross_expectancy']:.6f} | "
                f"{m['median_gross']:.6f} | {m['hit_rate_long']:.2%} | "
                f"{m['top_quantile_return']:.6f} |"
            )

    lines += [
        "",
        "## Random Baseline Comparison",
        "",
    ]

    rb = random_results.get("random_baseline", {})
    lines.append("| Horizon | Actual GE | Random Mean GE | Random Std GE | % Random Worse | Z-score |")
    lines.append("|---------|-----------|---------------|---------------|---------------|---------|")
    for h_key in sorted(rb.keys()):
        comp = rb[h_key]
        if "error" not in comp:
            lines.append(
                f"| {h_key} | {comp.get('actual_gross_expectancy', 0):.6f} | "
                f"{comp.get('random_mean_gross_expectancy', 0):.6f} | "
                f"{comp.get('random_std_gross_expectancy', 0):.6f} | "
                f"{comp.get('pct_random_worse', 0):.1%} | "
                f"{comp.get('z_score_vs_random', 0):.2f} |"
            )

    window_results = probe_results.get("windows", {})
    if window_results:
        lines += [
            "",
            "## Per-Window Analysis",
            "",
        ]
        for w_name, w_metrics in window_results.items():
            lines.append(f"### Window: {w_name}")
            lines.append("| Horizon | Gross Exp | Lead Corr | Hit Rate |")
            lines.append("|---------|-----------|-----------|----------|")
            for h_key in sorted(w_metrics.keys()):
                m = w_metrics[h_key]
                if isinstance(m, dict) and "error" not in m:
                    lines.append(
                        f"| {h_key} | {m['gross_expectancy']:.6f} | "
                        f"{m['lead_correlation']:.4f} | {m['hit_rate_long']:.2%} |"
                    )

    success_criteria = hypothesis.get("success_criteria", {})
    stop_criteria = hypothesis.get("stop_criteria", {})

    lines += [
        "",
        "## Success Criteria Check",
        "",
    ]

    for criterion, expected in success_criteria.items():
        lines.append(f"- **{criterion}:** expected {expected}")

    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "---",
        "",
        "*P0 signal probe completed. See JSON for full metrics.*",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"P0 report written to {md_path}")
    return md_path
