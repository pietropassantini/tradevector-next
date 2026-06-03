"""ML Report generator."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_ml_report(
    hypothesis: dict,
    ml_results: dict,
    decile_results: dict,
    verdict: str,
    symbol: str,
    timeframe: str,
    model_type: str,
    target_type: str,
    output_dir: Path,
) -> Path:
    hyp_id = hypothesis.get("id", "unknown")
    hyp_name = hypothesis.get("name", "Unknown")
    md_path = output_dir / f"{hyp_id}" / f"{symbol}_{timeframe}_{model_type}_{target_type}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# P0-ML Probe — {hyp_name}",
        "",
        f"**Hypothesis:** {hyp_id}",
        f"**Symbol:** {symbol}",
        f"**Timeframe:** {timeframe}",
        f"**Model:** {model_type}",
        f"**Target:** {target_type}",
        "",
        "## Decile Analysis",
        "",
    ]

    deciles = decile_results.get("deciles", {})
    if deciles:
        lines.append("| Decile | Count | Mean Fwd Ret | Median Fwd Ret | Hit Rate | Tail 5% | Tail 95% |")
        lines.append("|--------|-------|-------------|---------------|----------|---------|----------|")
        for d_label in sorted(deciles.keys()):
            d = deciles[d_label]
            lines.append(
                f"| {d_label} | {d['count']} | {d['mean_forward_return']:.6f} | "
                f"{d['median_forward_return']:.6f} | {d['hit_rate']:.2%} | "
                f"{d['tail_5pct']:.6f} | {d['tail_95pct']:.6f} |"
            )

    lines += [
        "",
        f"- **Top decile expectancy:** {decile_results.get('top_decile_expectancy', 'N/A')}",
        f"- **Bottom decile expectancy:** {decile_results.get('bottom_decile_expectancy', 'N/A')}",
        f"- **Score monotonicity:** {decile_results.get('score_monotonicity', 'N/A')}",
        "",
        f"## Verdict: {verdict}",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
