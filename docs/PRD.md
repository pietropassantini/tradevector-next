# PRD — TradeVector Next

## Progetto di ricerca quantitativa con nuove fonti informative e ML signal validation

## 1. Nome prodotto

**TradeVector Next**

---

## 2. Obiettivo del progetto

TradeVector Next è un progetto di ricerca quantitativa orientato a individuare, validare e trasformare nuove fonti informative di mercato in segnali predittivi misurabili.

Il progetto non nasce per ottimizzare strategie esistenti, ma per costruire una pipeline disciplinata che permetta di:

1. valutare nuove fonti dati;
2. trasformarle in feature;
3. verificare se contengono edge lordo;
4. validare il segnale con test statistici e ML;
5. costruire solo dopo una strategia minimale;
6. promuovere a paper trading solo segnali robusti.

Il principio centrale è:

> Non si costruisce una strategia prima di aver dimostrato che esiste un segnale.

---

## 3. Problema da risolvere

Nel trading quantitativo il rischio principale non è scrivere codice sbagliato, ma costruire codice corretto sopra un’ipotesi senza edge.

TradeVector Next deve evitare:

* tuning prematuro;
* overfitting;
* ottimizzazione su finestre fortunate;
* strategie costruite su correlazioni contemporanee;
* modelli ML usati come oracoli;
* backtest positivi ma non riproducibili;
* promozione di segnali con gross expectancy circa zero.

Il progetto deve quindi introdurre un processo in cui ogni idea viene falsificata rapidamente prima di assorbire settimane di sviluppo.

---

## 4. Scope del progetto

### 4.1 In scope

Sono inclusi:

* raccolta e normalizzazione dati da fonti esterne;
* open interest;
* liquidations;
* funding multi-exchange;
* basis spot/futures;
* cross-exchange dislocation;
* dati on-chain, se accessibili;
* dati options, se accessibili;
* feature engineering minimale;
* P0 statistico;
* P0 ML;
* validazione walk-forward;
* analisi per decili di score;
* strategia minimale P1;
* paper trading P2;
* reportistica tecnica;
* gestione esperimenti;
* versionamento risultati.

### 4.2 Out of scope

Sono esclusi dalla prima versione:

* HFT;
* ultra-low latency;
* colocazione;
* market making professionale;
* order book L2 full replay;
* simulazione queue position;
* live trading con capitale reale;
* ottimizzazione massiva dei parametri;
* deep learning complesso nella fase iniziale;
* modelli LSTM/Transformer come primo approccio;
* strategie basate solo su nuovi indicatori OHLCV.

---

## 5. Utenti del sistema

### 5.1 Researcher

Usa il sistema per:

* definire ipotesi;
* lanciare R0 dati;
* eseguire P0 statistici;
* eseguire P0 ML;
* leggere report;
* decidere se una tesi passa o fallisce.

### 5.2 Quant developer

Usa il sistema per:

* implementare downloader;
* creare feature;
* costruire script throwaway;
* trasformare segnali validati in strategie minimali;
* mantenere pipeline riproducibili.

### 5.3 Reviewer

Usa il sistema per:

* controllare metodologia;
* verificare assenza di leakage;
* validare risultati;
* approvare o bloccare il passaggio di fase.

---

## 6. Architettura logica

Il sistema sarà composto dai seguenti moduli:

1. **Data Source Registry**
2. **Data Downloader**
3. **Raw Data Storage**
4. **Data Quality Checker**
5. **Feature Builder**
6. **P0 Statistical Probe**
7. **P0 ML Probe**
8. **Experiment Tracker**
9. **Minimal Strategy Engine**
10. **Backtest Engine**
11. **Paper Trading Monitor**
12. **Report Generator**

---

## 7. Pipeline generale

```text
R0 Data Discovery
    ↓
R0 Data Acquisition
    ↓
R0 Data Quality & Alignment
    ↓
P0 Statistical Signal Probe
    ↓
P0 ML Signal Probe
    ↓
P1 Minimal Strategy
    ↓
P2 Paper Trading
    ↓
Candidate Live Review
```

Ogni fase produce un verdetto:

```text
PASS
FAIL
INCONCLUSIVE
NEEDS_DATA
OUT_OF_SCOPE
```

---

# 8. Fase R0 — Data Discovery

## 8.1 Obiettivo

Verificare se una nuova fonte dati è disponibile, accessibile, affidabile e utilizzabile per test quantitativi.

## 8.2 Input

Una fonte dati candidata.

Esempi:

* Binance Open Interest API;
* Bybit Open Interest API;
* Coinglass API;
* Glassnode API;
* CryptoQuant;
* Deribit API;
* dati cross-exchange;
* dati on-chain.

## 8.3 Output

Una scheda R0 compilata.

## 8.4 Scheda R0

```text
Nome fonte:
Provider:
Tipo dato:
Asset supportati:
Exchange supportati:
Frequenza del dato:
Storico disponibile:
Costo:
Formato:
Timestamp:
Allineamento temporale:
Granularità minima utilizzabile:
Possibilità di sincronizzazione con candle:
Gap noti:
Limiti API:
Rate limit:
Qualità documentazione:
Backtestabilità:
Necessità di raccolta live:
Note:
Verdetto:
```

## 8.5 Criteri di successo R0

La fonte passa R0 se:

* è accessibile programmaticamente;
* ha timestamp chiari;
* ha granularità compatibile con l’ipotesi;
* può essere sincronizzata con le candle;
* ha storico sufficiente oppure può essere raccolta live;
* introduce informazione nuova rispetto a OHLCV/funding già testati.

## 8.6 Criteri di stop R0

La fonte fallisce R0 se:

* non ha storico utilizzabile;
* il timestamp è ambiguo;
* la granularità è troppo bassa per l’ipotesi;
* il dato non è scaricabile in modo riproducibile;
* il costo è incompatibile con lo scope;
* non aggiunge informazione nuova.

---

# 9. Step implementativi R0

## Step R0.1 — Creare Data Source Registry

### Descrizione

Creare un file di configurazione che descrive tutte le fonti dati candidate.

### File suggerito

```text
config/data_sources.yml
```

### Contenuto esempio

```yaml
sources:
  binance_open_interest:
    provider: binance
    type: open_interest
    free: true
    requires_api_key: false
    granularity: 5m
    historical: true
    status: candidate

  coinglass_liquidations:
    provider: coinglass
    type: liquidations
    free: partial
    requires_api_key: true
    granularity: unknown
    historical: unknown
    status: candidate
```

### Deliverable

* file `data_sources.yml`;
* lista fonti candidate;
* priorità fonti.

---

## Step R0.2 — Implementare downloader prototipo

### Descrizione

Per ogni fonte prioritaria creare un downloader minimo.

### File suggeriti

```text
scripts/download_open_interest_binance.py
scripts/download_liquidations_coinglass.py
scripts/download_funding_bybit.py
```

### Requisiti

Ogni downloader deve:

* scaricare dati grezzi;
* salvare raw response;
* normalizzare in parquet;
* preservare timestamp originale;
* loggare eventuali gap;
* non applicare feature engineering.

### Output

```text
data/raw/{provider}/{dataset}/{symbol}/{timeframe}.parquet
```

---

## Step R0.3 — Implementare Data Quality Checker

### Descrizione

Creare uno script per controllare qualità, copertura e allineamento.

### File suggerito

```text
scripts/check_data_quality.py
```

### Controlli minimi

* numero righe;
* periodo coperto;
* timestamp duplicati;
* timestamp mancanti;
* frequenza reale;
* gap;
* timezone;
* valori nulli;
* valori anomali;
* compatibilità con candle.

### Output

```text
reports/r0/{dataset}_quality_report.md
```

---

## Step R0.4 — Implementare allineamento temporale

### Descrizione

Creare una funzione comune per allineare dati esterni alle candle.

### File suggerito

```text
src/tradevector/data/alignment.py
```

### Funzioni richieste

```python
align_to_candles(external_df, candles_df, method="last_known")
validate_alignment(aligned_df)
infer_data_granularity(df)
```

### Regole

* vietato usare dati futuri;
* usare solo informazioni disponibili al timestamp della candle;
* se il dato è daily, non può essere usato in test 5m come se fosse intraday;
* se il dato è event-based, deve essere aggregato in modo esplicito.

---

# 10. Fase P0 — Statistical Signal Probe

## 10.1 Obiettivo

Verificare se una feature grezza contiene informazione predittiva prima di costruire una strategia.

## 10.2 Input

* candle OHLCV;
* dati esterni allineati;
* feature grezza;
* target forward return.

## 10.3 Output

Report P0 statistico.

## 10.4 Metriche P0

```text
lead_correlation
mean_forward_return
gross_expectancy
median_gross
hit_rate
payoff_ratio
tail_loss
sample_size
window_stability
asset_stability
baseline_random_comparison
```

## 10.5 Regola decisiva

> La mean gross è decisiva. La median gross è diagnostica.

Una median positiva non basta se la mean gross è circa zero.

## 10.6 Lead multi-orizzonte

Ogni P0 deve testare più orizzonti.

Esempio:

| Timeframe dato | Lead da testare      |
| -------------- | -------------------- |
| 1m             | 1, 3, 5, 15 barre    |
| 5m             | 1, 2, 3, 6, 12 barre |
| 1h             | 1, 2, 4, 8 barre     |
| 4h             | 1, 2, 3, 6 barre     |
| Daily          | 1, 2, 3, 5 barre     |

## 10.7 Criteri di successo P0

Una feature passa P0 se:

* gross expectancy > 0;
* lead correlation non circa zero;
* effetto non solo contemporaneo;
* risultato stabile su più finestre;
* risultato non dominato da outlier;
* effetto coerente con l’ipotesi;
* ampiezza potenzialmente sufficiente per superare i costi.

## 10.8 Criteri di stop P0

Una feature fallisce P0 se:

* gross expectancy circa zero;
* lead correlation circa zero;
* effetto solo contemporaneo;
* median positiva ma mean circa zero;
* risultato positivo solo su un lead isolato;
* risultato non riproducibile su finestre OOS.

---

# 11. Step implementativi P0 statistico

## Step P0.1 — Definire schema ipotesi

### File suggerito

```text
research/hypotheses/{hypothesis_id}.yml
```

### Template

```yaml
id: oi_compression_breakout
name: OI Compression Breakout
data_sources:
  - binance_open_interest
  - binance_klines
information_new: "Derivatives leverage and positioning"
why_should_work: "OI growth during price compression may indicate leverage buildup"
event_predicted: "volatility expansion or directional breakout"
horizons:
  - 1h
  - 4h
assets:
  - BTCUSDT
  - ETHUSDT
features:
  - oi_change_pct
  - price_range_norm
  - atr_norm
  - funding_context
success_criteria:
  gross_expectancy: "> 0"
  lead_correlation: "not near zero"
  stability: "multi-window"
stop_criteria:
  gross_expectancy: "~ 0"
  lead_correlation: "~ 0"
```

---

## Step P0.2 — Creare Feature Builder minimale

### File suggerito

```text
src/tradevector/features/oi_features.py
```

### Feature iniziali

```text
oi_change_pct
oi_zscore
price_range_norm
atr_norm
oi_price_divergence
funding_zscore
liquidation_imbalance
basis_spot_future
```

### Regole

* feature semplici;
* niente tuning;
* niente feature selezionate dopo aver visto il risultato;
* niente leakage temporale.

---

## Step P0.3 — Creare script di signal probe

### File suggerito

```text
scripts/p0_signal_probe.py
```

### Parametri CLI

```bash
python scripts/p0_signal_probe.py \
  --hypothesis oi_compression_breakout \
  --symbol BTCUSDT \
  --timeframe 1h \
  --leads 1,2,4,8 \
  --windows W1,W2,W3
```

### Output

```text
reports/p0/{hypothesis_id}/{symbol}_{timeframe}_signal_probe.md
reports/p0/{hypothesis_id}/{symbol}_{timeframe}_signal_probe.json
```

---

## Step P0.4 — Baseline random

### Descrizione

Ogni P0 deve confrontare il segnale con una baseline casuale.

### Implementazione

Creare campioni random con stessa frequenza del segnale e confrontare:

* mean forward return;
* gross expectancy;
* hit rate;
* tail move frequency.

### File suggerito

```text
src/tradevector/validation/random_baseline.py
```

---

# 12. Fase P0-ML — Machine Learning Signal Probe

## 12.1 Obiettivo

Verificare se combinazioni non lineari di feature contengono informazione predittiva.

Il ML non deve creare la strategia.

Il ML deve produrre uno score predittivo da validare OOS.

## 12.2 Modelli ammessi nella prima versione

* Logistic Regression;
* Ridge/Lasso;
* Random Forest;
* XGBoost;
* LightGBM.

## 12.3 Modelli non ammessi nella prima versione

* LSTM;
* Transformer;
* reti neurali profonde;
* reinforcement learning;
* genetic programming.

## 12.4 Target ammessi

### Target direzionale

```text
y = 1 se forward_return_h > costo_minimo
y = -1 se forward_return_h < -costo_minimo
y = 0 altrimenti
```

### Target volatility expansion

```text
y = 1 se abs(forward_return_h) > k * ATR
```

### Target breakout

```text
y = 1 se max_move_h > threshold
```

### Target squeeze

```text
y = 1 se movimento futuro forte avviene contro il lato affollato
```

## 12.5 Validazione

La validazione deve essere temporale.

Sono vietati:

* random split;
* shuffle;
* cross-validation standard non temporale;
* training su dati futuri;
* feature calcolate usando informazioni future.

Validazioni ammesse:

* walk-forward validation;
* expanding window;
* rolling window;
* embargo pari almeno all’orizzonte del target;
* purging di campioni sovrapposti.

## 12.6 Metriche ML

Metriche diagnostiche:

```text
AUC
precision
recall
F1
calibration
feature_importance
SHAP
```

Metriche decisive:

```text
gross_expectancy_by_score_decile
forward_return_by_score_decile
score_monotonicity
top_decile_expectancy
bottom_decile_expectancy
coverage
stability_by_window
stability_by_asset
```

## 12.7 Criterio di successo P0-ML

Il modello passa se:

* top decile ha gross expectancy positiva;
* score più alto corrisponde a rendimento atteso migliore;
* relazione stabile OOS;
* modello batte baseline random;
* risultato non dipende da una singola finestra;
* feature importance non è completamente instabile;
* performance resta sensata con costi realistici stimati.

## 12.8 Criterio di stop P0-ML

Il modello fallisce se:

* top decile gross expectancy circa zero;
* score non monotono;
* performance solo in train;
* performance sparisce OOS;
* una sola finestra produce tutto il risultato;
* il modello complesso batte il semplice solo in training;
* il segnale non è traducibile in decisione operativa.

---

# 13. Step implementativi P0-ML

## Step ML.1 — Creare dataset supervisionato

### File suggerito

```text
src/tradevector/ml/dataset_builder.py
```

### Funzioni

```python
build_supervised_dataset(features_df, target_horizon, target_type)
add_forward_returns(df, horizons)
add_labels(df, threshold_config)
apply_embargo(df, horizon)
```

---

## Step ML.2 — Implementare temporal split

### File suggerito

```text
src/tradevector/ml/time_split.py
```

### Funzioni

```python
walk_forward_split(df, train_size, test_size, embargo)
expanding_window_split(df, min_train_size, test_size, embargo)
purge_overlapping_samples(df, horizon)
```

---

## Step ML.3 — Implementare trainer

### File suggerito

```text
scripts/p0_ml_probe.py
```

### Parametri CLI

```bash
python scripts/p0_ml_probe.py \
  --hypothesis oi_compression_breakout \
  --symbol BTCUSDT \
  --timeframe 1h \
  --target volatility_expansion \
  --horizon 4 \
  --model lightgbm \
  --validation walk_forward
```

---

## Step ML.4 — Analisi per decili

### File suggerito

```text
src/tradevector/ml/decile_analysis.py
```

### Output

Per ogni decile score:

* numero campioni;
* forward return medio;
* forward return mediano;
* gross expectancy;
* hit rate;
* tail move frequency;
* confronto baseline.

---

## Step ML.5 — Report ML

### File suggerito

```text
src/tradevector/reporting/ml_report.py
```

### Output

```text
reports/p0_ml/{hypothesis_id}/{symbol}_{timeframe}_{model}_{target}.md
reports/p0_ml/{hypothesis_id}/{symbol}_{timeframe}_{model}_{target}.json
```

---

# 14. Fase P1 — Minimal Strategy

## 14.1 Obiettivo

Trasformare un segnale validato in una strategia minimale.

La strategia non deve ottimizzare tutto.

Deve solo verificare se il segnale può diventare decisione operativa.

## 14.2 Input

* segnale P0/P0-ML validato;
* score o feature;
* cost model;
* regole entry/exit semplici.

## 14.3 Output

Backtest P1.

## 14.4 Regole P1

Ammesso:

* soglia semplice sullo score;
* entry long/short/no trade;
* exit time-based;
* stop loss semplice;
* take profit semplice;
* cost model realistico;
* test multi-window;
* test multi-asset.

Vietato:

* grid search esteso;
* param sweep massivo;
* tuning post-hoc;
* aggiunta progressiva di filtri;
* ottimizzazione su una sola finestra;
* promozione su pochi trade.

## 14.5 Criteri di successo P1

La strategia passa se:

* net expectancy > 0;
* gross expectancy > 0;
* gross/cost ratio >= 1.2;
* sample size sufficiente;
* drawdown accettabile;
* risultato stabile OOS;
* risultato non dominato da pochi outlier;
* performance coerente con P0.

---

# 15. Step implementativi P1

## Step P1.1 — Creare Strategy Config

### File suggerito

```text
config/strategies/{strategy_id}.yml
```

### Esempio

```yaml
id: oi_compression_breakout_v1
hypothesis: oi_compression_breakout
signal_source: p0_ml_score
entry:
  long_if_score_above: 0.90
  short_if_score_below: 0.10
exit:
  type: time_based
  bars: 4
risk:
  max_position: 1
cost_model:
  type: f0_realistic
```

---

## Step P1.2 — Implementare minimal strategy runner

### File suggerito

```text
scripts/p1_strategy_backtest.py
```

### Parametri CLI

```bash
python scripts/p1_strategy_backtest.py \
  --strategy oi_compression_breakout_v1 \
  --symbol BTCUSDT \
  --timeframe 1h \
  --windows W1,W2,W3
```

---

## Step P1.3 — Report P1

### Output

```text
reports/p1/{strategy_id}/{symbol}_{timeframe}_backtest.md
reports/p1/{strategy_id}/{symbol}_{timeframe}_backtest.json
```

### Metriche

* total return;
* gross expectancy;
* net expectancy;
* number of trades;
* win rate;
* payoff ratio;
* max drawdown;
* profit factor;
* Sharpe diagnostico;
* cost impact;
* per-window performance;
* per-asset performance.

---

# 16. Fase P2 — Paper Trading

## 16.1 Obiettivo

Verificare se la strategia mantiene comportamento coerente in ambiente live/pseudo-live.

## 16.2 Scope P2

P2 non usa capitale reale.

P2 registra:

* segnali;
* prezzo teorico;
* prezzo eseguibile;
* slippage stimato;
* outcome;
* differenza rispetto al backtest.

## 16.3 Step implementativi P2

### Step P2.1 — Signal Scheduler

File suggerito:

```text
scripts/p2_signal_scheduler.py
```

Funzioni:

* carica strategia;
* aggiorna dati;
* calcola feature;
* calcola score;
* genera segnale;
* salva segnale.

### Step P2.2 — Paper Ledger

File suggerito:

```text
src/tradevector/paper/ledger.py
```

Campi ledger:

```text
signal_id
timestamp
symbol
timeframe
strategy_id
side
entry_price_theoretical
entry_price_estimated
exit_time
exit_price_theoretical
exit_price_estimated
gross_pnl
net_pnl_estimated
slippage_estimated
status
```

### Step P2.3 — Paper Report

Output:

```text
reports/p2/{strategy_id}/weekly_report.md
reports/p2/{strategy_id}/paper_ledger.parquet
```

---

# 17. Prima ipotesi implementativa

## 17.1 Nome

**OI Compression Breakout**

## 17.2 Ipotesi

Quando l’open interest cresce rapidamente mentre il prezzo resta compresso in un range stretto, il mercato accumula leva. Questa leva può anticipare una rottura direzionale, uno squeeze o un aumento significativo della volatilità futura.

## 17.3 Informazione nuova

Posizionamento e leva del mercato derivatives.

## 17.4 Fonti dati candidate

1. Binance API;
2. Bybit API;
3. Coinglass API;
4. Glassnode;
5. CryptoQuant.

## 17.5 Feature iniziali

```text
oi_change_pct
oi_zscore
price_range_norm
atr_norm
oi_price_divergence
funding_zscore
volume_zscore
liquidation_imbalance
basis_spot_future
```

## 17.6 Target iniziale

Target primario:

```text
volatility_expansion_4h
```

Target secondario:

```text
directional_breakout_4h
```

## 17.7 Timeframe iniziale

```text
1h
4h
```

## 17.8 Asset iniziali

```text
BTCUSDT
ETHUSDT
```

## 17.9 Primo test P0

Misurare se crescita anomala di OI combinata con compressione del prezzo anticipa:

* aumento di volatilità futura;
* breakout direzionale;
* incremento della probabilità di movimento sopra soglia ATR.

---

# 18. Struttura repository suggerita

```text
tradevector-next/
│
├── config/
│   ├── data_sources.yml
│   ├── experiments.yml
│   └── strategies/
│
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── aligned/
│   └── features/
│
├── research/
│   └── hypotheses/
│
├── reports/
│   ├── r0/
│   ├── p0/
│   ├── p0_ml/
│   ├── p1/
│   └── p2/
│
├── scripts/
│   ├── download_open_interest_binance.py
│   ├── check_data_quality.py
│   ├── p0_signal_probe.py
│   ├── p0_ml_probe.py
│   ├── p1_strategy_backtest.py
│   └── p2_signal_scheduler.py
│
├── src/
│   └── tradevector/
│       ├── data/
│       │   ├── alignment.py
│       │   ├── quality.py
│       │   └── loaders.py
│       │
│       ├── features/
│       │   ├── oi_features.py
│       │   ├── liquidation_features.py
│       │   └── market_features.py
│       │
│       ├── validation/
│       │   ├── signal_probe.py
│       │   ├── random_baseline.py
│       │   └── metrics.py
│       │
│       ├── ml/
│       │   ├── dataset_builder.py
│       │   ├── time_split.py
│       │   ├── trainer.py
│       │   ├── decile_analysis.py
│       │   └── evaluation.py
│       │
│       ├── strategy/
│       │   ├── minimal_strategy.py
│       │   ├── cost_model.py
│       │   └── backtest.py
│       │
│       ├── paper/
│       │   ├── ledger.py
│       │   └── monitor.py
│       │
│       └── reporting/
│           ├── r0_report.py
│           ├── p0_report.py
│           ├── ml_report.py
│           └── p1_report.py
│
├── tests/
│   ├── test_alignment.py
│   ├── test_features.py
│   ├── test_time_split.py
│   └── test_no_leakage.py
│
├── pyproject.toml
├── README.md
└── PRD.md
```

---

# 19. Roadmap implementativa

## Milestone 0 — Bootstrap progetto

### Obiettivo

Creare struttura base repository.

### Task

* creare repository;
* creare struttura directory;
* configurare ambiente Python;
* configurare `pyproject.toml`;
* aggiungere README;
* aggiungere PRD;
* aggiungere cartelle `data`, `reports`, `scripts`, `src`, `tests`.

### Deliverable

* repository iniziale;
* commit `bootstrap tradevector-next`.

---

## Milestone 1 — R0 Data Layer

### Obiettivo

Scaricare e validare prime fonti dati.

### Task

* creare `data_sources.yml`;
* implementare downloader Binance Open Interest;
* implementare downloader Binance funding, se utile;
* implementare quality checker;
* implementare alignment module;
* produrre primo report R0.

### Deliverable

* dati raw;
* dati normalized;
* report qualità;
* verdetto R0.

---

## Milestone 2 — Feature Layer

### Obiettivo

Costruire feature grezze per OI Compression Breakout.

### Task

* implementare `oi_features.py`;
* calcolare `oi_change_pct`;
* calcolare `oi_zscore`;
* calcolare `price_range_norm`;
* calcolare `atr_norm`;
* calcolare `oi_price_divergence`;
* salvare feature parquet;
* test unitari su feature.

### Deliverable

* dataset feature;
* test verdi;
* documentazione feature.

---

## Milestone 3 — P0 Statistical Probe

### Obiettivo

Misurare edge lordo statistico.

### Task

* implementare `p0_signal_probe.py`;
* implementare lead multi-orizzonte;
* implementare baseline random;
* implementare report P0;
* eseguire test BTC/ETH su 1h/4h.

### Deliverable

* report P0;
* verdetto PASS/FAIL/INCONCLUSIVE.

---

## Milestone 4 — P0 ML Probe

### Obiettivo

Verificare se combinazioni non lineari di feature contengono segnale.

### Task

* implementare dataset builder;
* implementare target `volatility_expansion`;
* implementare target `directional_breakout`;
* implementare walk-forward split;
* implementare embargo;
* implementare trainer LightGBM/XGBoost;
* implementare decile analysis;
* implementare report ML.

### Deliverable

* report P0-ML;
* decile analysis;
* feature importance;
* verdetto ML.

---

## Milestone 5 — P1 Minimal Strategy

### Obiettivo

Trasformare solo segnali validati in una strategia minima.

### Task

* creare config strategia;
* implementare minimal strategy runner;
* implementare cost model;
* implementare report P1;
* test multi-window e multi-asset.

### Deliverable

* backtest P1;
* report strategia;
* verdetto paper/no paper.

---

## Milestone 6 — P2 Paper Trading

### Obiettivo

Monitorare strategia in pseudo-live.

### Task

* implementare signal scheduler;
* implementare paper ledger;
* implementare report settimanale;
* confrontare segnali live con aspettativa backtest.

### Deliverable

* paper ledger;
* report settimanali;
* decisione finale.

---

# 20. Test richiesti

## 20.1 Test tecnici

* test allineamento temporale;
* test assenza leakage;
* test feature;
* test split temporale;
* test embargo;
* test baseline random;
* test calcolo expectancy;
* test report generation.

## 20.2 Test metodologici

Ogni esperimento deve verificare:

* nessun dato futuro nelle feature;
* target calcolato solo dopo timestamp segnale;
* training sempre precedente al test;
* embargo applicato;
* nessuno shuffle temporale;
* costo applicato solo in P1;
* P0 sempre lordo prima del netto.

---

# 21. Definition of Done

## 21.1 DoD R0

R0 è completato quando:

* fonte dati documentata;
* downloader funzionante o motivo di stop documentato;
* qualità dati verificata;
* allineamento temporale chiarito;
* report generato;
* verdetto assegnato.

## 21.2 DoD P0

P0 è completato quando:

* feature calcolata;
* lead test multi-orizzonte eseguito;
* gross expectancy misurata;
* baseline random confrontata;
* finestre OOS valutate;
* report generato;
* verdetto assegnato.

## 21.3 DoD P0-ML

P0-ML è completato quando:

* dataset supervisionato creato;
* target definito;
* walk-forward applicato;
* embargo applicato;
* modello baseline confrontato;
* decile analysis prodotta;
* report generato;
* verdetto assegnato.

## 21.4 DoD P1

P1 è completato quando:
P1 è completato quando:

* strategia minimale implementata;
* cost model applicato;
* backtest OOS completato;
* report generato;
* decisione paper/no paper presa.

## 21.5 DoD P2

P2 è completato quando:

* segnali raccolti per periodo minimo;
* ledger completo;
* slippage stimato;
* confronto con backtest prodotto;
* decisione finale documentata.

---

# 22. Principi non negoziabili

1. Una nuova ipotesi deve introdurre nuova informazione che il mercato non prezza già.
2. Nessuna strategia prima del segnale.
3. P0 misura prima il lordo.
4. Mean gross > median gross.
5. Correlazione contemporanea non è edge.
6. Split temporale obbligatorio.
7. No random split su serie temporali.
8. No tuning per salvare ipotesi deboli.
9. No ML senza baseline semplice.
10. No paper trading senza P1 positivo.
11. Ogni fase deve poter dire FAIL.
12. Un FAIL rapido è un risultato utile.

---

# 23. Primo backlog operativo

## Backlog iniziale

| ID      | Task                                          | Priorità |
| ------- |-----------------------------------------------| -------- |
| TVN-001 | Creare repository ``                          | Alta     |
| TVN-002 | Creare struttura directory                    | Alta     |
| TVN-003 | Aggiungere `PRD.md`                           | Alta     |
| TVN-004 | Creare `data_sources.yml`                     | Alta     |
| TVN-005 | Implementare downloader Binance Open Interest | Alta     |
| TVN-006 | Implementare data quality checker             | Alta     |
| TVN-007 | Implementare alignment module                 | Alta     |
| TVN-008 | Creare ipotesi `oi_compression_breakout.yml`  | Alta     |
| TVN-009 | Implementare feature OI base                  | Alta     |
| TVN-010 | Implementare P0 signal probe                  | Alta     |
| TVN-011 | Implementare baseline random                  | Media    |
| TVN-012 | Generare primo report P0 BTC 1h               | Alta     |
| TVN-013 | Estendere P0 a ETH 1h/4h                      | Media    |
| TVN-014 | Implementare dataset builder ML               | Media    |
| TVN-015 | Implementare walk-forward split               | Media    |
| TVN-016 | Implementare decile analysis                  | Media    |
| TVN-017 | Implementare P0 ML probe                      | Media    |
| TVN-018 | Scrivere test anti-leakage                    | Alta     |
| TVN-019 | Implementare P1 minimal strategy runner       | Bassa    |
| TVN-020 | Implementare P2 paper ledger                  | Bassa    |

---

# 24. Prima sequenza consigliata di implementazione

La prima sequenza da implementare è:

```text
TVN-001 → TVN-002 → TVN-003 → TVN-004
→ TVN-005 → TVN-006 → TVN-007
→ TVN-008 → TVN-009 → TVN-010
→ TVN-012
```

Questo produce il primo risultato utile:

> Verdetto P0 su OI Compression Breakout BTC 1h.

Solo dopo quel risultato ha senso estendere a ETH, ML o strategia.

---

# 25. Conclusione

TradeVector Next deve essere costruito come laboratorio di ricerca, non come engine di trading prematuro.

La prima implementazione deve concentrarsi su:

1. dati;
2. allineamento temporale;
3. feature grezze;
4. segnale lordo;
5. validazione statistica;
6. solo dopo ML;
7. solo dopo strategia.

La direzione iniziale consigliata è:

> OI Compression Breakout su BTC/ETH 1h/4h.

Il primo obiettivo concreto non è fare profitto.

Il primo obiettivo concreto è rispondere a questa domanda:

> L’open interest aggiunge informazione predittiva misurabile rispetto alle sole candle?

Se la risposta è sì, il progetto passa a P0-ML e P1.

Se la risposta è no, la pipeline avrà comunque prodotto valore: avrà falsificato rapidamente una tesi senza costruire un altro sistema inutile.
