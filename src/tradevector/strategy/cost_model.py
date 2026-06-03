"""Cost model for strategy backtesting.

Realistic cost estimates for futures trading.
"""

import logging

logger = logging.getLogger(__name__)


def f0_realistic(
    position_size: float = 1.0,
    taker_fee_bps: float = 4.0,
    maker_fee_bps: float = 2.0,
    slippage_bps: float = 2.0,
    funding_daily_bps: float = 1.0,
    holding_bars: int = 4,
) -> dict:
    taker_fee = taker_fee_bps / 10000
    maker_fee = maker_fee_bps / 10000
    slippage = slippage_bps / 10000
    funding = funding_daily_bps / 10000 / 24 * holding_bars

    per_trade_cost = (taker_fee + maker_fee + slippage) * position_size

    return {
        "per_trade_cost_bps": float((taker_fee + maker_fee + slippage) * 10000),
        "per_trade_cost_absolute": per_trade_cost,
        "funding_estimate_bps": float(funding * 10000),
        "total_roundtrip_cost_bps": float((taker_fee + maker_fee + slippage + funding) * 10000),
    }


def compute_costs(
    n_trades: int,
    position_size: float = 1.0,
    holding_bars: int = 4,
) -> dict:
    costs = f0_realistic(position_size=position_size, holding_bars=holding_bars)
    costs["total_cost"] = costs["per_trade_cost_absolute"] * n_trades
    return costs
