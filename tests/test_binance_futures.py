"""Collector futures/data: accumulo e ancoraggio, senza toccare la rete."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.data.binance_futures import (
    FUTURES_DATA_ENDPOINTS,
    MAX_LOOKBACK_DAYS,
    accumulate,
    coerce_numeric,
    metric_path,
    resolve_start_time,
    to_dataframe,
)


def _record(ts_ms, oi="100.5"):
    return {"symbol": "BTCUSDT", "sumOpenInterest": oi,
            "CMCCirculatingSupply": "20049956.0", "timestamp": ts_ms}


@pytest.fixture
def archivio(tmp_path):
    return tmp_path / "BTCUSDT_1h.parquet"


class TestResolveStartTime:
    def test_archivio_assente_usa_il_floor(self, archivio):
        atteso = (datetime.now(timezone.utc) - timedelta(days=MAX_LOOKBACK_DAYS)).timestamp() * 1000
        assert abs(resolve_start_time(archivio) - atteso) < 5000

    def test_riparte_dopo_ultimo_timestamp(self, archivio):
        ultimo = datetime.now(timezone.utc) - timedelta(hours=3)
        to_dataframe([_record(int(ultimo.timestamp() * 1000))]).to_parquet(archivio)
        assert resolve_start_time(archivio) == int(ultimo.timestamp() * 1000) + 1

    def test_archivio_vecchio_non_supera_la_retention(self, archivio):
        vecchio = datetime.now(timezone.utc) - timedelta(days=200)
        to_dataframe([_record(int(vecchio.timestamp() * 1000))]).to_parquet(archivio)
        floor_ms = (
            datetime.now(timezone.utc) - timedelta(days=MAX_LOOKBACK_DAYS)
        ).timestamp() * 1000
        assert abs(resolve_start_time(archivio) - floor_ms) < 5000


class TestAccumulo:
    def test_non_sovrascrive_lo_storico(self, archivio):
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        vecchi = [_record(int((base + timedelta(hours=i)).timestamp() * 1000)) for i in range(5)]
        accumulate(to_dataframe(vecchi), archivio)

        nuovi = [_record(int((base + timedelta(hours=i)).timestamp() * 1000)) for i in range(5, 8)]
        out = accumulate(to_dataframe(nuovi), archivio)

        assert len(out) == 8
        assert len(pd.read_parquet(archivio)) == 8

    def test_sovrapposizioni_tengono_il_dato_nuovo(self, archivio):
        ts = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
        accumulate(to_dataframe([_record(ts, oi="1.0")]), archivio)
        out = accumulate(to_dataframe([_record(ts, oi="2.0")]), archivio)
        assert len(out) == 1
        assert out["sumOpenInterest"].iloc[0] == 2.0

    def test_deduplica_anche_alla_prima_scrittura(self, archivio):
        """La paginazione puo' restituire record sovrapposti: un indice
        duplicato non deve finire su disco nemmeno al primo salvataggio."""
        ts = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
        out = accumulate(to_dataframe([_record(ts, oi="1.0"), _record(ts, oi="2.0")]), archivio)
        assert len(out) == 1
        assert out["sumOpenInterest"].iloc[0] == 2.0
        assert not pd.read_parquet(archivio).index.duplicated().any()

    def test_archivio_con_colonne_stringa_si_fonde(self, archivio):
        """Gli storici vecchi tenevano alcune colonne come stringa: il concat
        con le nuove float non deve rompere la serializzazione parquet."""
        ts = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
        legacy = pd.DataFrame(
            [{"symbol": "BTCUSDT", "sumOpenInterest": 1.0,
              "CMCCirculatingSupply": "20049956.00000000",
              "timestamp": pd.to_datetime(ts, unit="ms", utc=True)}]
        ).set_index("timestamp")
        legacy.to_parquet(archivio)

        out = accumulate(to_dataframe([_record(ts + 3600000)]), archivio)
        assert len(out) == 2
        assert pd.api.types.is_numeric_dtype(out["CMCCirculatingSupply"])


class TestVarie:
    def test_coerce_lascia_symbol_stringa(self):
        df = coerce_numeric(pd.DataFrame([{"symbol": "BTCUSDT", "v": "1.5"}]))
        assert df["symbol"].iloc[0] == "BTCUSDT"
        assert df["v"].iloc[0] == 1.5

    def test_open_interest_mantiene_il_percorso_storico(self, tmp_path):
        p = metric_path("open_interest", "BTCUSDT", "1h", raw_dir=tmp_path)
        assert p == tmp_path / "open_interest" / "BTCUSDT" / "BTCUSDT_1h.parquet"

    def test_metrica_sconosciuta(self, tmp_path):
        with pytest.raises(ValueError):
            metric_path("inventata", "BTCUSDT", "1h", raw_dir=tmp_path)

    def test_tutte_le_serie_hanno_una_cartella_distinta(self):
        cartelle = [folder for _, folder in FUTURES_DATA_ENDPOINTS.values()]
        assert len(cartelle) == len(set(cartelle))
