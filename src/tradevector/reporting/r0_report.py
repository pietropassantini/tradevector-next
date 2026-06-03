"""R0 Report generator."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_r0_report(
    source_name: str,
    source_config: dict,
    quality_report: dict,
    compatibility_report: dict,
    output_dir: Path,
) -> Path:
    md_path = output_dir / f"{source_name}_r0_report.md"

    lines = [
        f"# R0 Data Discovery — {source_name}",
        "",
        f"## Scheda R0",
        "",
        f"- **Nome fonte:** {source_name}",
        f"- **Provider:** {source_config.get('provider', 'N/A')}",
        f"- **Tipo dato:** {source_config.get('type', 'N/A')}",
        f"- **Frequenza del dato:** {source_config.get('granularity', 'N/A')}",
        f"- **Costo:** {'Free' if source_config.get('free') else 'Paid/Partial'}",
        f"- **Formato:** JSON via REST API",
        f"- **Backtestabilità:** {'Si' if source_config.get('historical') else 'No'}",
        "",
        "## Data Quality Summary",
        "",
    ]

    q = quality_report
    lines += [
        f"- **Rows:** {q.get('rows', 0)}",
        f"- **Period:** {q.get('period_start', 'N/A')} → {q.get('period_end', 'N/A')}",
        f"- **Timezone:** {q.get('timezone', 'N/A')}",
        f"- **Duplicate timestamps:** {q.get('duplicate_timestamps', 0)}",
        f"- **Missing intervals:** {q.get('missing_intervals', 0)}",
        f"- **Expected freq:** {q.get('expected_frequency', 'N/A')}",
        f"- **Actual freq:** {q.get('actual_frequency', 'N/A')}",
    ]

    anomalies = q.get("anomalies_detected", [])
    if anomalies:
        lines.append("")
        lines.append("### Anomalies")
        for a in anomalies:
            lines.append(f"- {a}")

    lines += [
        "",
        "## Compatibility with Candles",
        "",
    ]

    c = compatibility_report
    lines += [
        f"- **Overlap:** {c.get('overlap_start', 'N/A')} → {c.get('overlap_end', 'N/A')}",
        f"- **Candle freq:** {c.get('candle_frequency', 'N/A')}",
        f"- **External freq:** {c.get('external_frequency', 'N/A')}",
    ]

    issues = c.get("issues", [])
    if issues:
        lines.append("")
        lines.append("### Issues")
        for issue in issues:
            lines.append(f"- {issue}")

    has_issues = bool(anomalies) or bool(issues)
    verdict = "PASS" if not has_issues else "PASS with warnings" if len(anomalies) + len(issues) <= 2 else "FAIL"

    lines += [
        "",
        f"## Verdict: {verdict}",
        "",
        "---",
        "",
        "*R0 data discovery completed.*",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"R0 report written to {md_path}")
    return md_path
