"""Check v2: check raw/normalized data: coverage, timestamps, nulls, gaps, timezone."""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def check_data_quality(
    df: pd.DataFrame,
    expected_freq: Optional[str] = None,
    dataset_name: str = "unknown",
) -> dict:
    report = {
        "dataset": dataset_name,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "period_start": str(df.index.min()) if len(df) > 0 else None,
        "period_end": str(df.index.max()) if len(df) > 0 else None,
        "duplicate_timestamps": 0,
        "missing_intervals": 0,
        "expected_frequency": expected_freq,
        "actual_frequency": None,
        "gaps": [],
        "timezone": str(df.index.tz) if df.index.tz else "none",
        "null_counts": {},
        "anomalies_detected": [],
    }

    if len(df) == 0:
        report["anomalies_detected"].append("Empty dataframe")
        return report

    dupes = df.index.duplicated().sum()
    report["duplicate_timestamps"] = int(dupes)
    if dupes > 0:
        report["anomalies_detected"].append(f"{dupes} duplicate timestamps")

    nulls = df.isnull().sum()
    report["null_counts"] = {k: int(v) for k, v in nulls.items() if v > 0}
    if nulls.any():
        report["anomalies_detected"].append(f"Null values found in columns: {list(nulls[nulls > 0].index)}")

    freq_seconds = df.index.to_series().diff().median().total_seconds() if len(df) > 1 else None
    if freq_seconds:
        report["actual_frequency"] = f"{freq_seconds:.0f}s"
        expected_seconds = _parse_freq_to_seconds(expected_freq)
        if expected_seconds and abs(freq_seconds - expected_seconds) > (expected_seconds * 0.1):
            report["anomalies_detected"].append(
                f"Frequency mismatch: expected {expected_freq} ({expected_seconds}s), got {freq_seconds:.0f}s"
            )

    if len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median()
        gap_threshold = median_diff * 3
        gaps = diffs[diffs > gap_threshold]
        report["missing_intervals"] = len(gaps)
        report["gaps"] = [
            {"start": str(gaps.index[i] - diffs.iloc[diffs.index.get_loc(gaps.index[i]) - 1]),
             "end": str(gaps.index[i]),
             "duration": str(gaps.iloc[i])}
            for i in range(min(len(gaps), 10))
        ]

    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        outliers = (z_scores > 5).sum()
        if outliers > 0:
            report["anomalies_detected"].append(
                f"{outliers} extreme values (>5 sigma) in {col}"
            )

    return report


def _parse_freq_to_seconds(freq: Optional[str]) -> Optional[int]:
    if freq is None:
        return None
    mapping = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "8h": 28800,
        "1d": 86400,
    }
    return mapping.get(freq)


def check_compatibility_with_candles(
    external_df: pd.DataFrame,
    candles_df: pd.DataFrame,
) -> dict:
    report = {
        "candle_count": len(candles_df),
        "external_count": len(external_df),
        "overlap_start": None,
        "overlap_end": None,
        "candle_frequency": None,
        "external_frequency": None,
        "shared_timestamps": 0,
        "issues": [],
    }

    if len(external_df) == 0 or len(candles_df) == 0:
        report["issues"].append("One or both datasets are empty")
        return report

    overlap_start = max(external_df.index.min(), candles_df.index.min())
    overlap_end = min(external_df.index.max(), candles_df.index.max())
    report["overlap_start"] = str(overlap_start)
    report["overlap_end"] = str(overlap_end)

    if overlap_start > overlap_end:
        report["issues"].append("No temporal overlap between datasets")
        return report

    if len(candles_df) > 1:
        report["candle_frequency"] = str(candles_df.index.to_series().diff().median())
    if len(external_df) > 1:
        report["external_frequency"] = str(external_df.index.to_series().diff().median())

    return report


def generate_r0_report(
    quality_report: dict,
    compatibility_report: dict,
    output_dir: Path,
    dataset_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{dataset_name}_quality_report.md"

    lines = [
        f"# R0 Quality Report — {dataset_name}",
        "",
        f"**Generated:** {quality_report['checked_at']}",
        "",
        "## Data Quality",
        "",
        f"- Rows: {quality_report['rows']}",
        f"- Period: {quality_report['period_start']} → {quality_report['period_end']}",
        f"- Timezone: {quality_report['timezone']}",
        f"- Columns: {', '.join(quality_report['columns'])}",
        f"- Duplicate timestamps: {quality_report['duplicate_timestamps']}",
        f"- Missing intervals: {quality_report['missing_intervals']}",
        f"- Expected frequency: {quality_report['expected_frequency']}",
        f"- Actual frequency: {quality_report['actual_frequency']}",
        "",
        "## Null Values",
        "",
    ]

    if quality_report["null_counts"]:
        for col, count in quality_report["null_counts"].items():
            lines.append(f"- {col}: {count}")
    else:
        lines.append("- None")

    lines += [
        "",
        "## Anomalies",
        "",
    ]

    if quality_report["anomalies_detected"]:
        for a in quality_report["anomalies_detected"]:
            lines.append(f"- {a}")
    else:
        lines.append("- None detected")

    lines += [
        "",
        "## Compatibility with Candles",
        "",
        f"- Candle count: {compatibility_report.get('candle_count', 'N/A')}",
        f"- External count: {compatibility_report.get('external_count', 'N/A')}",
        f"- Overlap: {compatibility_report.get('overlap_start', 'N/A')} → {compatibility_report.get('overlap_end', 'N/A')}",
        f"- Candle frequency: {compatibility_report.get('candle_frequency', 'N/A')}",
        f"- External frequency: {compatibility_report.get('external_frequency', 'N/A')}",
    ]

    if compatibility_report.get("issues"):
        lines.append("")
        lines.append("### Issues")
        for issue in compatibility_report["issues"]:
            lines.append(f"- {issue}")

    lines += [
        "",
        "## Verdict",
        "",
    ]

    issues = len(quality_report["anomalies_detected"]) + len(compatibility_report.get("issues", []))
    if issues == 0:
        lines.append("PASS — No quality issues detected.")
    elif issues <= 2:
        lines.append("PASS with warnings — Minor issues detected, review before P0.")
    else:
        lines.append("FAIL — Critical data quality issues.")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    import json
    json_path = output_dir / f"{dataset_name}_quality_report.json"
    with open(json_path, "w") as f:
        json.dump({"quality": quality_report, "compatibility": compatibility_report}, f, indent=2, default=str)

    return md_path


def main():
    parser = argparse.ArgumentParser(description="Check data quality of raw/normalized data")
    parser.add_argument("--data-path", required=True, help="Path to parquet/csv file")
    parser.add_argument("--candles-path", default=None, help="Path to candles parquet for compatibility check")
    parser.add_argument("--expected-freq", default=None, help="Expected frequency (e.g. 5m, 1h)")
    parser.add_argument("--dataset-name", default="dataset", help="Name for report files")
    parser.add_argument("--output-dir", default=None, help="Output directory for report")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "reports" / "r0"

    df = load_data(Path(args.data_path))
    quality = check_data_quality(df, args.expected_freq, args.dataset_name)

    compatibility = {}
    if args.candles_path:
        candles_df = load_data(Path(args.candles_path))
        compatibility = check_compatibility_with_candles(df, candles_df)
    else:
        compatibility = {"issues": ["No candles provided for compatibility check"]}

    report_path = generate_r0_report(quality, compatibility, output_dir, args.dataset_name)
    logger.info(f"Report generated: {report_path}")

    for key, value in quality.items():
        if key not in ("gaps", "null_counts", "columns"):
            logger.info(f"  {key}: {value}")

    if quality["anomalies_detected"]:
        logger.warning(f"  anomalies: {quality['anomalies_detected']}")


if __name__ == "__main__":
    main()
