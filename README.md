# Intraday Mean Reversion: Backtest to Paper Trading

A single-name intraday mean-reversion strategy taken from research notebook through
parameter tuning to a live paper-trading loop against the Alpaca API.

The point of the project is the gap between the two: a z-score reversion rule that looks
profitable on a tuned backtest has to survive slippage, live bar latency, and position
reconciliation before it means anything.

## Strategy

Standardize the close against a rolling window to get a z-score, enter when price is far
from the mean, and exit when it reverts back inside a tighter band.

Entries are gated by a momentum filter with the sign deliberately chosen: a long entry
requires the z-score to be deeply negative *and* momentum already turning up, so the
strategy is not stepping in front of a divergence that is still widening.

Parameters (`src/paper_trader.py`, tuned in `src/exp.ipynb`):

| Parameter | Value | Meaning |
| --- | --- | --- |
| `Z_WIN` | 70 | Rolling window for mean and standard deviation |
| `ENTRY_Z` | 2.5 | Standard deviations from the mean to open |
| `EXIT_Z` | 0.5 | Band to close back inside |
| `MOM_THR` | 0.0001 | Momentum gate on entries |

`EXIT_Z` matters more than it looks. It started equal to `ENTRY_Z`, which meant a
position was held until a full opposing spike rather than until the reversion the
strategy was actually predicting — the exit was not testing the hypothesis the entry
made.

## Research notebook

`src/exp.ipynb` contains the backtest: signal construction, a cost-aware backtest with
slippage in basis points, and a grid search over entry/exit thresholds, window length,
and the momentum gate. Evaluation is on a held-out tail of the sample rather than the
window used for tuning.

## Paper trader

`src/paper_trader.py` runs the tuned rule live against Alpaca paper trading on 1-minute
bars. It reconciles against any existing broker position on startup rather than assuming
it begins flat, writes a heartbeat file so a stalled loop is detectable, and logs every
action to CSV with the z-score and momentum that triggered it, so live fills can be
compared against what the backtest expected.

## Run

Credentials are read from the environment and never stored in the repository:

```bash
export APCA_API_KEY_ID=<your key>
export APCA_API_SECRET_KEY=<your secret>

pip install -r requirements.txt
python src/paper_trader.py
```

Paper trading endpoint only (`paper-api.alpaca.markets`).

## Note

Research code, not investment advice. Results are from a tuned backtest and a paper
account; nothing here has traded real capital.
