"""Sonda P0: causalità delle soglie top/bottom e baseline per permutazione."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.validation.random_baseline import (
    generate_permuted_signals,
    run_random_baseline,
)
from tradevector.validation.signal_probe import (
    compute_quantile_masks,
    compute_signal_metrics,
    compute_forward_returns,
)

WINDOW = 100


@pytest.fixture
def serie():
    """Regime che si sposta: la coda ha valori sistematicamente più alti."""
    rng = np.random.default_rng(7)
    n = 500
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    score = np.concatenate([rng.normal(0, 1, 350), rng.normal(5, 1, 150)])
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"close": close, "score": score}, index=idx)


class TestMaschereCausali:
    def test_soglie_mobili_non_guardano_avanti(self, serie):
        """Troncando la serie, la classificazione del prefisso non cambia."""
        s = serie["score"]
        limite = 350
        top_full, _ = compute_quantile_masks(s, 0.8, 0.2, WINDOW)
        top_tronc, _ = compute_quantile_masks(s.iloc[:limite], 0.8, 0.2, WINDOW)
        assert top_full.iloc[:limite].tolist() == top_tronc.tolist()

    def test_soglie_in_sample_invece_cambiano(self, serie):
        s = serie["score"]
        limite = 350
        top_full, _ = compute_quantile_masks(s, 0.8, 0.2, None)
        top_tronc, _ = compute_quantile_masks(s.iloc[:limite], 0.8, 0.2, None)
        assert top_full.iloc[:limite].tolist() != top_tronc.tolist()

    def test_il_regime_alto_non_monopolizza_il_top(self, serie):
        """In-sample il top finisce quasi tutto nella coda; con finestra mobile no."""
        s = serie["score"]
        top_in_sample, _ = compute_quantile_masks(s, 0.8, 0.2, None)
        top_rolling, _ = compute_quantile_masks(s, 0.8, 0.2, WINDOW)
        quota_coda_in_sample = top_in_sample.iloc[350:].sum() / max(top_in_sample.sum(), 1)
        quota_coda_rolling = top_rolling.iloc[350:].sum() / max(top_rolling.sum(), 1)
        assert quota_coda_in_sample > 0.9
        assert quota_coda_rolling < 0.6

    def test_warmup_escluso(self, serie):
        top, bottom = compute_quantile_masks(serie["score"], 0.8, 0.2, WINDOW)
        min_periods = max(30, WINDOW // 4)
        assert not top.iloc[: min_periods - 1].any()
        assert not bottom.iloc[: min_periods - 1].any()

    def test_segnale_costante_non_produce_maschere(self, serie):
        piatto = pd.Series(1.0, index=serie.index)
        top, bottom = compute_quantile_masks(piatto, 0.8, 0.2, WINDOW)
        assert top.sum() == 0
        assert bottom.sum() == 0


class TestMetriche:
    def test_etichetta_la_modalita(self, serie):
        fwd = compute_forward_returns(serie["close"], [4])
        m = compute_signal_metrics(serie["score"], fwd, [4], quantile_window=WINDOW)
        assert m["h4"]["thresholds"] == f"rolling_{WINDOW}"
        m = compute_signal_metrics(serie["score"], fwd, [4], quantile_window=None)
        assert m["h4"]["thresholds"] == "in_sample"

    def test_segnale_costante_segnala_errore(self, serie):
        fwd = compute_forward_returns(serie["close"], [4])
        piatto = pd.Series(1.0, index=serie.index)
        m = compute_signal_metrics(piatto, fwd, [4], quantile_window=WINDOW)
        assert "error" in m["h4"]


class TestBaselinePermutato:
    def test_conserva_la_distribuzione(self, serie):
        perm = generate_permuted_signals(serie["score"], n_trials=3, seed=1)
        atteso = np.sort(serie["score"].to_numpy())
        for col in perm.columns:
            assert np.allclose(np.sort(perm[col].to_numpy()), atteso)

    def test_rompe_l_ordine_temporale(self, serie):
        perm = generate_permuted_signals(serie["score"], n_trials=1, seed=1)
        assert not np.allclose(perm["random_0"].to_numpy(), serie["score"].to_numpy())

    def test_baseline_produce_metriche_valide(self, serie):
        """Il baseline precedente degenerava e non era confrontabile."""
        out = run_random_baseline(
            features=serie, signal_column="score", horizons=[4],
            n_trials=10, quantile_window=WINDOW,
        )
        comp = out["random_baseline"]["h4"]
        assert "error" not in comp
        assert comp["n_random_trials"] == 10
        assert 0.0 <= comp["pct_random_worse"] <= 1.0
