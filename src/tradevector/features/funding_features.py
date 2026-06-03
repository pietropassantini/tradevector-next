"""Feature builder per Funding Rate Positioning Reversal hypothesis.

Funding 8h allineato a candle 1h via ffill. Features semplici, no tuning.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STORICO_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "storico"


BINANCE_KLINES_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "binance" / "klines"


def load_storico_klines(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    """Carica klines dallo storico, fallback a Binance raw se non disponibile."""
    pattern = f"{symbol}_{timeframe}_*.parquet"
    matches = list(STORICO_DIR.glob(pattern))
    if matches:
        path = sorted(matches)[-1]
        df = pd.read_parquet(path)
        for col in ["asset", "timeframe"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(f"Klines storico: {len(df)} righe, {df.index.min()} → {df.index.max()}")
        return df

    # Fallback: klines Binance raw
    binance_path = BINANCE_KLINES_DIR / symbol / f"{symbol}_{timeframe}.parquet"
    if binance_path.exists():
        df = pd.read_parquet(binance_path)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(f"Klines Binance (fallback): {len(df)} righe, {df.index.min()} → {df.index.max()}")
        return df

    raise FileNotFoundError(
        f"Klines non trovate per {symbol} {timeframe}. "
        f"Storico: {STORICO_DIR / pattern} | Binance: {binance_path}"
    )


def load_storico_funding(symbol: str) -> pd.Series:
    """Carica funding rate dallo storico. Pulisce timestamp con sub-ms."""
    pattern = f"{symbol}_funding_*.parquet"
    matches = list(STORICO_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Funding storico non trovato: {STORICO_DIR / pattern}")
    path = sorted(matches)[-1]
    df = pd.read_parquet(path)
    funding = df["funding_rate"].copy()
    # Binance a volte ha offset .001000 — arrotonda al secondo
    funding.index = funding.index.floor("s")
    funding = funding[~funding.index.duplicated(keep="first")]
    funding = pd.to_numeric(funding, errors="coerce").dropna()
    logger.info(f"Funding storico: {len(funding)} record, {funding.index.min()} → {funding.index.max()}")
    return funding


def build_funding_features(
    candles: pd.DataFrame,
    funding: pd.Series,
    zscore_periods: int = 90,
    lookback_periods: int = 20,
    cumsum_periods: int = 9,
) -> pd.DataFrame:
    """Costruisce feature dal funding rate allineato alle candle 1h.

    zscore_periods=90: finestra di 90 record funding (8h ciascuno = 30gg).
    cumsum_periods=9: somma ultimi 9 funding (= 3 giorni).
    """
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(candles.columns):
        raise ValueError(f"Candles mancano: {required - set(candles.columns)}")

    features = pd.DataFrame(index=candles.index)
    features["close"] = candles["close"]
    features["volume"] = candles["volume"]

    # Klines features
    features["returns"] = candles["close"].pct_change(fill_method=None)
    features["log_returns"] = np.log(candles["close"] / candles["close"].shift(1))
    atr = _compute_atr(candles, lookback_periods)
    features["atr"] = atr
    features["atr_norm"] = atr / candles["close"]
    features["volume_zscore"] = (
        (candles["volume"] - candles["volume"].rolling(zscore_periods * 8).mean())
        / candles["volume"].rolling(zscore_periods * 8).std()
    )

    # Allinea funding (8h) alle candle (1h) — ffill, no lookahead
    funding_aligned = funding.reindex(candles.index, method="ffill")
    features["funding_rate"] = funding_aligned

    # Zscore su finestra mobile (calcolato sul funding 8h, poi allineato)
    funding_on_8h = funding.copy()
    funding_zscore_8h = (
        (funding_on_8h - funding_on_8h.rolling(zscore_periods).mean())
        / funding_on_8h.rolling(zscore_periods).std()
    )
    features["funding_zscore"] = funding_zscore_8h.reindex(candles.index, method="ffill")

    # Cumsum ultimi N record funding (proxy accumulo leva)
    funding_cumsum_8h = funding_on_8h.rolling(cumsum_periods).sum()
    features["funding_cumsum_3d"] = funding_cumsum_8h.reindex(candles.index, method="ffill")

    # Persistenza del regime (quante volte consecutive stesso segno)
    funding_sign = np.sign(funding_on_8h)
    persistence = _compute_sign_persistence(funding_sign)
    features["funding_regime_persistence"] = persistence.reindex(candles.index, method="ffill")

    # Flag estremo: |zscore| > 2
    features["funding_extreme_flag"] = (features["funding_zscore"].abs() > 2).astype(float)

    features = features.replace([np.inf, -np.inf], np.nan)

    null_frac = features.isnull().mean().mean()
    complete = features.dropna()
    logger.info(f"Funding features: {len(features)} righe x {len(features.columns)} col")
    logger.info(f"Null fraction: {null_frac:.2%} | Righe complete: {len(complete)}")
    return features


def _compute_atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = candles["high"], candles["low"], candles["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _compute_sign_persistence(sign_series: pd.Series) -> pd.Series:
    """Conta quante barre consecutive con stesso segno (positivo = long crowded)."""
    result = pd.Series(0, index=sign_series.index, dtype=float)
    count = 0
    prev = 0
    for i, (idx, val) in enumerate(sign_series.items()):
        if val == prev and val != 0:
            count += 1
        else:
            count = 1 if val != 0 else 0
        result.iloc[i] = count * val
        prev = val
    return result


def save_funding_features(
    features: pd.DataFrame,
    dataset_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data" / "features"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dataset_name}.parquet"
    features.to_parquet(path)
    logger.info(f"Salvato: {path}")
    return path
