"""Harness P1: causalità delle soglie e posizioni contemporanee.

Il backtest deve poter essere confrontato con l'esecuzione live. Due modi di
rompere il confronto: scegliere le soglie guardando tutto il campione, e
sommare trade sovrapposti che nessun capitale singolo avrebbe potuto tenere.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.strategy.minimal_strategy import run_minimal_strategy

WINDOW = 100
EXIT_BARS = 8


@pytest.fixture
def serie():
    """Segnale con cambio di regime nella coda: le soglie in-sample ne risentono."""
    rng = np.random.default_rng(42)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    score = np.concatenate([rng.normal(0, 1, 300), rng.normal(4, 1, 100)])
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"close": close, "score": score}, index=idx)


def _esegui(df, **kw):
    base = dict(
        score_column="score", entry_threshold_long=0.8, entry_threshold_short=0.2,
        exit_bars=EXIT_BARS, cost_bps=6.0, quantile_window=WINDOW, max_concurrent=1,
    )
    base.update(kw)
    return run_minimal_strategy(features=df, **base)


class TestCausalita:
    def test_soglie_causali_ignorano_le_barre_future(self, serie):
        """Troncando la serie, i trade nel prefisso comune devono restare identici."""
        completo = _esegui(serie)
        troncato = _esegui(serie.iloc[:300])

        limite = 300 - EXIT_BARS
        a = [t["entry_idx"] for t in completo["trades"] if t["exit_idx"] < limite]
        b = [t["entry_idx"] for t in troncato["trades"] if t["exit_idx"] < limite]
        assert a == b

    def test_soglie_in_sample_invece_cambiano(self, serie):
        """Controprova: con le soglie sull'intero campione il prefisso si muove."""
        completo = _esegui(serie, quantile_window=None)
        troncato = _esegui(serie.iloc[:300], quantile_window=None)

        limite = 300 - EXIT_BARS
        a = [t["entry_idx"] for t in completo["trades"] if t["exit_idx"] < limite]
        b = [t["entry_idx"] for t in troncato["trades"] if t["exit_idx"] < limite]
        assert a != b

    def test_etichetta_la_modalita(self, serie):
        assert _esegui(serie)["thresholds"] == f"rolling_{WINDOW}"
        assert _esegui(serie, quantile_window=None)["thresholds"] == "in_sample"


class TestPosizioniContemporanee:
    def test_una_posizione_alla_volta_non_si_sovrappone(self, serie):
        trades = _esegui(serie)["trades"]
        assert len(trades) > 1
        for precedente, successivo in zip(trades, trades[1:]):
            assert successivo["entry_idx"] >= precedente["exit_idx"]

    def test_sovrapposizione_illimitata_produce_piu_trade(self, serie):
        assert (
            _esegui(serie, max_concurrent=None)["n_trades"]
            > _esegui(serie, max_concurrent=1)["n_trades"]
        )

    def test_tetto_a_tre_rispettato(self, serie):
        trades = _esegui(serie, max_concurrent=3)["trades"]
        for t in trades:
            aperte = sum(
                1 for altro in trades
                if altro["entry_idx"] < t["entry_idx"] < altro["exit_idx"]
            )
            assert aperte < 3

    def test_composto_solo_se_sequenziale(self, serie):
        assert np.isfinite(_esegui(serie)["net_return_compounded"])
        assert np.isnan(_esegui(serie, max_concurrent=None)["net_return_compounded"])


class TestGuardie:
    def test_segnale_costante_non_apre_nulla(self, serie):
        piatto = serie.copy()
        piatto["score"] = 1.0
        assert _esegui(piatto)["n_trades"] == 0

    def test_costi_riducono_il_netto(self, serie):
        senza = _esegui(serie, cost_bps=0.0)
        con = _esegui(serie, cost_bps=6.0)
        assert con["net_expectancy"] < senza["net_expectancy"]
        assert senza["net_expectancy"] == pytest.approx(senza["gross_expectancy"])

    def test_warmup_insufficiente_non_apre(self, serie):
        """Con finestra piu' lunga della serie non ci sono soglie affidabili."""
        assert _esegui(serie.iloc[:20], quantile_window=WINDOW)["n_trades"] == 0
