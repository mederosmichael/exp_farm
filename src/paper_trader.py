"""
Paper trader for mean-reversion z-score strategy.
Runs against Alpaca paper trading API using 1-min AAPL bars.

Usage:
    export APCA_API_KEY_ID=<your key>
    export APCA_API_SECRET_KEY=<your secret>
    python src/paper_trader.py
"""

import os
import sys
import time
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi

# ── Strategy parameters (from notebook tuning) ──────────────────────────
SYMBOL = "AAPL"
ENTRY_Z = 2.5
EXIT_Z  = 0.5   # exit when z reverts near mean (was 2.5 — held until full opposing spike, not mean-reversion)
Z_WIN = 70
K = 1
MOM_THR = 0.0001
QTY = 1  # shares per trade

# ── Alpaca config ────────────────────────────────────────────────────────
BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = os.environ.get("APCA_API_KEY_ID", "")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY", "")

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "paper_trades.csv"
HEARTBEAT_FILE = DATA_DIR / ".alpaca_heartbeat"

POLL_INTERVAL = 60  # seconds


# ── Signal engine (mirrors notebook signals()) ──────────────────────────
def compute_signals(closes: pd.Series):
    """Return z-score, momentum, and entry/exit flags for the latest bar."""
    mu = closes.rolling(Z_WIN).mean()
    sig = closes.rolling(Z_WIN).std()
    z = (closes - mu) / sig

    mom = closes / closes.shift(K) - 1.0

    long_entry  = (z < -ENTRY_Z) & (mom > MOM_THR)
    short_entry = (z > ENTRY_Z)  & (mom < -MOM_THR)
    long_exit   = z > EXIT_Z    # exit long when z reverts above +0.5
    short_exit  = z < -EXIT_Z   # exit short when z reverts below -0.5

    return z, mom, long_entry, short_entry, long_exit, short_exit


# ── CSV logger ───────────────────────────────────────────────────────────
def init_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "action", "side", "price", "qty",
                "z_score", "momentum", "position_after", "trade_count",
            ])


def log_trade(action, side, price, z_score, momentum, position_after, trade_count):
    ts = datetime.utcnow().isoformat()
    row = [ts, action, side, f"{price:.4f}", QTY,
           f"{z_score:.4f}", f"{momentum:.6f}", position_after, trade_count]
    print(f"  >> TRADE  {action:5s} {side:5s}  px={price:.2f}  "
          f"z={z_score:.4f}  mom={momentum:.6f}")
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ── Alpaca helpers ───────────────────────────────────────────────────────
def submit_order(api, side: str):
    """Submit a market order and return the filled order."""
    print(f"  -> Submitting {side} market order for {QTY} {SYMBOL}")
    order = api.submit_order(
        symbol=SYMBOL,
        qty=QTY,
        side=side,
        type="market",
        time_in_force="gtc",
    )
    # wait for fill (up to 30s)
    for _ in range(30):
        order = api.get_order(order.id)
        if order.status == "filled":
            print(f"  <- Filled @ {order.filled_avg_price}")
            return order
        time.sleep(1)
    print(f"  !! Order not filled within 30s (status={order.status})")
    return order


def get_current_position(api):
    """Return current qty (+long, -short, 0 flat) from Alpaca."""
    try:
        pos = api.get_position(SYMBOL)
        return int(pos.qty) if pos.side == "long" else -int(pos.qty)
    except tradeapi.rest.APIError:
        return 0


def close_position(api):
    """Close any existing Alpaca position in the symbol."""
    try:
        api.close_position(SYMBOL)
        print(f"  <- Closed existing {SYMBOL} position on Alpaca")
        time.sleep(2)
    except tradeapi.rest.APIError:
        pass  # no position to close


# ── Main loop ────────────────────────────────────────────────────────────
def main():
    if not API_KEY or not API_SECRET:
        print("ERROR: Set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars")
        sys.exit(1)

    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

    # verify connection
    account = api.get_account()
    print(f"Connected to Alpaca paper account: {account.account_number}")
    print(f"  Equity: ${float(account.equity):,.2f}  "
          f"Buying power: ${float(account.buying_power):,.2f}")

    init_csv()

    # state machine
    position = 0       # -1 short, 0 flat, 1 long
    entry_price = 0.0
    trade_count = 0

    # sync with any existing Alpaca position
    alpaca_pos = get_current_position(api)
    if alpaca_pos != 0:
        print(f"  Found existing {SYMBOL} position: {alpaca_pos} shares")
        position = 1 if alpaca_pos > 0 else -1

    bars_needed = Z_WIN + K + 5  # a few extra for safety

    print(f"\nStarting paper trader: {SYMBOL}  "
          f"entry_z={ENTRY_Z} exit_z={EXIT_Z} z_win={Z_WIN} k={K} "
          f"mom_thr={MOM_THR}")
    print(f"Polling every {POLL_INTERVAL}s for 1-min bars\n")

    while True:
        try:
            # write heartbeat so dashboard knows we're alive
            HEARTBEAT_FILE.write_text(datetime.utcnow().isoformat())

            # fetch recent 1-min bars
            barset = api.get_bars(
                SYMBOL,
                tradeapi.TimeFrame.Minute,
                limit=bars_needed,
                feed="iex",   # IEX = free feed; SIP requires paid subscription (was causing 401)
            )
            bars = [{"timestamp": b.t, "close": b.c} for b in barset]

            if len(bars) < bars_needed:
                print(f"[{datetime.utcnow():%H:%M:%S}] "
                      f"Only {len(bars)} bars available, need {bars_needed}. "
                      f"Market may be closed. Waiting...")
                time.sleep(POLL_INTERVAL)
                continue

            df = pd.DataFrame(bars)
            closes = df["close"].astype(float)

            z, mom, long_entry, short_entry, long_exit, short_exit = \
                compute_signals(closes)

            # use the second-to-last bar for signals (last bar may be incomplete)
            idx = len(df) - 2
            cur_z = z.iloc[idx]
            cur_mom = mom.iloc[idx]
            cur_price = closes.iloc[idx]
            le = bool(long_entry.iloc[idx])
            se = bool(short_entry.iloc[idx])
            lx = bool(long_exit.iloc[idx])
            sx = bool(short_exit.iloc[idx])

            now_str = datetime.utcnow().strftime("%H:%M:%S")
            print(f"[{now_str}] px={cur_price:.2f}  z={cur_z:.4f}  "
                  f"mom={cur_mom:.6f}  pos={position}  trades={trade_count}")

            # ── Exit logic ───────────────────────────────────────────
            if position == 1 and lx:
                close_position(api)
                pnl = (cur_price - entry_price) / entry_price
                log_trade("EXIT", "sell", cur_price, cur_z, cur_mom,
                          0, trade_count)
                print(f"  ** Closed LONG  pnl={pnl:+.4%}")
                position = 0
                entry_price = 0.0

            elif position == -1 and sx:
                close_position(api)
                pnl = (entry_price - cur_price) / entry_price
                log_trade("EXIT", "buy", cur_price, cur_z, cur_mom,
                          0, trade_count)
                print(f"  ** Closed SHORT  pnl={pnl:+.4%}")
                position = 0
                entry_price = 0.0

            # ── Flip logic (close + open same bar) ───────────────────
            if position == 1 and se:
                close_position(api)
                pnl = (cur_price - entry_price) / entry_price
                log_trade("EXIT", "sell", cur_price, cur_z, cur_mom,
                          0, trade_count)
                print(f"  ** Flipped LONG->SHORT  pnl={pnl:+.4%}")
                submit_order(api, "sell")
                trade_count += 1
                entry_price = cur_price
                position = -1
                log_trade("ENTRY", "short", cur_price, cur_z, cur_mom,
                          -1, trade_count)

            elif position == -1 and le:
                close_position(api)
                pnl = (entry_price - cur_price) / entry_price
                log_trade("EXIT", "buy", cur_price, cur_z, cur_mom,
                          0, trade_count)
                print(f"  ** Flipped SHORT->LONG  pnl={pnl:+.4%}")
                submit_order(api, "buy")
                trade_count += 1
                entry_price = cur_price
                position = 1
                log_trade("ENTRY", "long", cur_price, cur_z, cur_mom,
                          1, trade_count)

            # ── Entry logic (flat) ───────────────────────────────────
            if position == 0:
                if le:
                    submit_order(api, "buy")
                    trade_count += 1
                    entry_price = cur_price
                    position = 1
                    log_trade("ENTRY", "long", cur_price, cur_z, cur_mom,
                              1, trade_count)
                elif se:
                    submit_order(api, "sell")
                    trade_count += 1
                    entry_price = cur_price
                    position = -1
                    log_trade("ENTRY", "short", cur_price, cur_z, cur_mom,
                              -1, trade_count)

            # ── Unrealized P&L ───────────────────────────────────────
            if position != 0:
                if position == 1:
                    upnl = (cur_price - entry_price) / entry_price
                else:
                    upnl = (entry_price - cur_price) / entry_price
                print(f"  Unrealized P&L: {upnl:+.4%}")

        except KeyboardInterrupt:
            print("\nShutting down paper trader.")
            break
        except Exception as e:
            print(f"  !! Error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
