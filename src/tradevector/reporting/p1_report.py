"""P1 Report generator."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_p1_report(
    strategy_id: str,
    symbol: str,
    timeframe: str,
    backtest_results: dict,
    output_dir: Path,
) -> Path:
    md_path = output_dir / f"{strategy_id}" / f"{symbol}_{timeframe}_backtest.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# P1 Backtest — {strategy_id}",
        "",
        f"**Symbol:** {symbol}",
        f"**Timeframe:** {timeframe}",
        "",
        "## Performance",
        "",
    ]

    for key, value in backtest_results.items():
        if isinstance(value, (int, float)):
            lines.append(f"- **{key}:** {value:.4f}")
        else:
            lines.append(f"- **{key}:** {value}")

    lines += [
        "",
        "---",
        "*P1 backtest completed.*",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
