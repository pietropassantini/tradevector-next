# TradeVector Next

Pipeline di ricerca quantitativa per individuare, validare e trasformare fonti informative di mercato in segnali predittivi misurabili.

## Principio centrale

> Non si costruisce una strategia prima di aver dimostrato che esiste un segnale.

## Pipeline

```
R0 Data Discovery → R0 Data Acquisition → R0 Data Quality & Alignment
    ↓
P0 Statistical Signal Probe → P0 ML Signal Probe
    ↓
P1 Minimal Strategy → P2 Paper Trading → Candidate Live Review
```

## Installazione

```bash
pip install -e .
# oppure con dipendenze ML
pip install -e ".[ml]"
# sviluppo
pip install -e ".[dev,ml]"
```

## Primo obiettivo

Rispondere alla domanda:

> L'open interest aggiunge informazione predittiva misurabile rispetto alle sole candle?

Ipotesi: **OI Compression Breakout** su BTC/ETH 1h/4h.

## Struttura

```
tradevector-next/
├── config/           # Configurazioni (data sources, strategie)
├── data/             # Dati raw, normalized, aligned, features
├── research/         # Ipotesi di ricerca
├── reports/          # Report per fase (r0, p0, p0_ml, p1, p2)
├── scripts/          # Entry point eseguibili
├── src/tradevector/  # Libreria core
└── tests/            # Test
```

Vedi [PRD.md](PRD.md) per la specifica completa.
