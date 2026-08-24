from __future__ import annotations
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import requests

app = FastAPI(title='NSE AI Trading Agent', version='4.0.0-paper')
CAPITAL = 10_000.0

# ---------------------------------------------------------------------------
# Persistent agent state (Vercel KV / Upstash Redis over the standard `redis`
# client, using the KV_URL env var Vercel injects once a KV store is
# connected to the project). Every serverless invocation is a fresh process,
# so without this the "autonomous" agent would forget its own portfolio
# between cron ticks. If KV_URL isn't set (or the connection fails), state
# falls back to an in-memory default for that single invocation only —
# the agent still runs, it just doesn't remember anything, and `persisted`
# in every response says so honestly.
# ---------------------------------------------------------------------------
STATE_KEY = 'agent:state:v1'
_kv_client = None
_kv_attempted = False

def get_kv():
    global _kv_client, _kv_attempted
    if _kv_attempted:
        return _kv_client
    _kv_attempted = True
    url = os.environ.get('KV_URL') or os.environ.get('REDIS_URL')
    if not url:
        return None
    try:
        import redis
        _kv_client = redis.from_url(url, socket_timeout=4, socket_connect_timeout=4, decode_responses=True)
        _kv_client.ping()
    except Exception:
        _kv_client = None
    return _kv_client

def default_state():
    return {'cash': CAPITAL, 'positions': {}, 'trades': [], 'log': [], 'last_decision': None}

def load_state():
    kv = get_kv()
    if kv is not None:
        try:
            raw = kv.get(STATE_KEY)
            if raw:
                state = json.loads(raw)
                for k, v in default_state().items():
                    state.setdefault(k, v)
                return state, True
        except Exception:
            pass
    return default_state(), False

def save_state(state: dict) -> bool:
    kv = get_kv()
    if kv is None:
        return False
    try:
        kv.set(STATE_KEY, json.dumps(state))
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Live market data (Yahoo Finance chart API — no key required).
# NSE tickers are suffixed ".NS"; a few common indices get friendly aliases.
# Every quote/history call is cached briefly and falls back to a synthetic
# price path if the upstream feed is unreachable or rate-limits us, so the
# dashboard degrades gracefully instead of erroring out.
# ---------------------------------------------------------------------------
YAHOO_CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
YAHOO_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; nse-paper-agent/1.0)'}
INDEX_ALIASES = {
    'NIFTY': '^NSEI', 'NIFTY50': '^NSEI', 'NIFTY_50': '^NSEI',
    'BANKNIFTY': '^NSEBANK', 'NIFTYBANK': '^NSEBANK',
    'SENSEX': '^BSESN',
}
_cache: dict[str, tuple[float, Any]] = {}
CACHE_TTL_INTRADAY = 10.0
CACHE_TTL_HISTORY = 900.0

def to_yahoo_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    if s in INDEX_ALIASES:
        return INDEX_ALIASES[s]
    if s.startswith('^') or '.' in s:
        return s
    return f'{s}.NS'

def _cached(key: str, ttl: float):
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    return None

def fetch_yahoo(symbol: str, interval: str, range_: str, ttl: float):
    yh = to_yahoo_symbol(symbol)
    key = f'{yh}:{interval}:{range_}'
    cached = _cached(key, ttl)
    if cached is not None:
        return cached
    resp = requests.get(
        YAHOO_CHART_URL.format(symbol=yh),
        params={'interval': interval, 'range': range_},
        headers=YAHOO_HEADERS, timeout=6,
    )
    resp.raise_for_status()
    result = resp.json()['chart']['result'][0]
    closes = result['indicators']['quote'][0]['close']
    timestamps = result['timestamp']
    pairs = [(t, c) for t, c in zip(timestamps, closes) if c is not None]
    if len(pairs) < 15:
        raise ValueError('not enough live bars returned')
    prices = np.array([c for _, c in pairs], dtype=float)
    last_ts = pairs[-1][0]
    out = (prices, last_ts)
    _cache[key] = (time.time(), out)
    return out

def synthetic_series(bars: int, seed: int):
    bars = max(80, min(int(bars), 5000))
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00015, 0.012, bars)
    for i in range(bars):
        returns[i] += 0.00035 if (i // 90) % 2 else -0.00012
    return 100 * np.exp(np.cumsum(returns))

def get_intraday_prices(symbol: str, min_bars: int = 60):
    try:
        prices, last_ts = fetch_yahoo(symbol, interval='2m', range_='5d', ttl=CACHE_TTL_INTRADAY)
        if len(prices) < min_bars:
            prices, last_ts = fetch_yahoo(symbol, interval='5m', range_='1mo', ttl=CACHE_TTL_INTRADAY)
        return prices, last_ts, 'live'
    except Exception:
        return synthetic_series(250, seed=abs(hash(symbol)) % 100000), None, 'synthetic_fallback'

def get_history_prices(symbol: str, min_bars: int = 200):
    try:
        prices, last_ts = fetch_yahoo(symbol, interval='1d', range_='2y', ttl=CACHE_TTL_HISTORY)
        if len(prices) < min_bars:
            raise ValueError('not enough daily history')
        return prices, last_ts, 'live'
    except Exception:
        return synthetic_series(1200, seed=abs(hash(symbol)) % 100000 + 1), None, 'synthetic_fallback'

class Request(BaseModel):
    symbol: str = 'RELIANCE'

def decision_for(symbol: str, prices: np.ndarray, fast_n: int = 5, slow_n: int = 20, mom_n: int = 10):
    n = len(prices)
    fast_n = max(2, min(fast_n, n))
    slow_n = max(fast_n + 1, min(slow_n, n))
    mom_n = max(1, min(mom_n, n - 1))
    fast = float(np.mean(prices[-fast_n:])); slow = float(np.mean(prices[-slow_n:]))
    vol = float(np.std(np.diff(np.log(prices[-slow_n:]))))
    momentum = float((prices[-1] / prices[-1 - mom_n]) - 1)
    score = 0.5; reasons = []
    if fast > slow: score += 0.18; reasons.append('short-term trend above baseline')
    else: score -= 0.18; reasons.append('short-term trend below baseline')
    if momentum > 0.01: score += 0.14; reasons.append('positive momentum')
    elif momentum < -0.01: score -= 0.14; reasons.append('negative momentum')
    if vol < 0.012: score += 0.06; reasons.append('controlled volatility')
    elif vol > 0.025: score -= 0.06; reasons.append('high volatility')
    score = float(np.clip(score, 0.05, 0.95))
    action = 'BUY' if score >= 0.62 else ('EXIT' if score <= 0.38 else 'WAIT')
    regime = 'TRENDING' if abs(momentum) > 0.015 else ('HIGH_VOLATILITY' if vol > 0.025 else 'RANGE')
    return {'symbol':symbol,'action':action,'score':round(score,4),'probability_success':round(score,4),
            'expected_return':round(momentum*100,4),'downside':round(max(vol*100,0.1),4),'regime':regime,
            'rationale':'; '.join(reasons),'scenarios':{'bull':round(float(np.clip(score+0.10,0,1)),4),'base':round(score,4),'bear':round(float(np.clip(1-score+0.10,0,1)),4)},'price':round(float(prices[-1]),4)}

# ---------------------------------------------------------------------------
# Nifty 100 scanner. Three horizons, each with its own bar interval/lookback
# and its own fast/slow window — an intraday scalp and a golden-cross-style
# investing read have nothing in common except sharing decision_for's shape.
# This is a static snapshot of Nifty 50 + Nifty Next 50 constituents (NSE's
# official list drifts on periodic index reconstitution — this isn't pulled
# live from NSE, so treat it as approximate).
# ---------------------------------------------------------------------------
NIFTY_100 = [
    'RELIANCE','TCS','HDFCBANK','ICICIBANK','INFY','HINDUNILVR','ITC','SBIN','BHARTIARTL','BAJFINANCE',
    'LT','KOTAKBANK','AXISBANK','ASIANPAINT','MARUTI','SUNPHARMA','TITAN','ULTRACEMCO','NESTLEIND','WIPRO',
    'ADANIENT','ONGC','NTPC','POWERGRID','M&M','TATAMOTORS','TATASTEEL','JSWSTEEL','HCLTECH','TECHM',
    'BAJAJFINSV','INDUSINDBK','GRASIM','DRREDDY','CIPLA','DIVISLAB','EICHERMOT','BPCL','COALINDIA','HEROMOTOCO',
    'BRITANNIA','SHREECEM','UPL','HDFCLIFE','SBILIFE','APOLLOHOSP','TATACONSUM','ADANIPORTS','BAJAJ-AUTO','HINDALCO',
    'DMART','PIDILITIND','GODREJCP','DABUR','HAVELLS','SIEMENS','AMBUJACEM','ICICIPRULI','ICICIGI','SBICARD',
    'BANKBARODA','PNB','CANBK','IOC','GAIL','INDIGO','VEDL','BOSCHLTD','MARICO','COLPAL',
    'BERGEPAINT','LUPIN','AUROPHARMA','TORNTPHARM','ALKEM','MOTHERSON','BEL','HAL','LTIM','MPHASIS',
    'PERSISTENT','NAUKRI','ZOMATO','PAYTM','IRCTC','PAGEIND','MUTHOOTFIN','CHOLAFIN','BAJAJHLDNG','ABB',
    'CUMMINSIND','TVSMOTOR','ASHOKLEY','ACC','JINDALSTEL','SAIL','NMDC','PIIND','INDUSTOWER','TRENT',
]

HORIZONS = {
    'intraday': {'fast': 5, 'slow': 20, 'mom': 10, 'interval': '5m', 'range': '5d', 'ttl': 20.0, 'min_bars': 25},
    'swing':    {'fast': 10, 'slow': 50, 'mom': 20, 'interval': '1d', 'range': '6mo', 'ttl': 300.0, 'min_bars': 55},
    'invest':   {'fast': 50, 'slow': 200, 'mom': 60, 'interval': '1d', 'range': '2y', 'ttl': 900.0, 'min_bars': 210},
}
SCAN_WORKERS = 16
SCAN_TIMEOUT_S = 25.0

def run_pool(fn, items, *args):
    """Run fn(item, *args) across items in a thread pool, returning whatever
    completes within SCAN_TIMEOUT_S. A slow straggler degrades the result to
    partial coverage instead of losing every already-finished result to an
    uncaught as_completed() timeout."""
    results = []
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        futures = {pool.submit(fn, item, *args): item for item in items}
        try:
            for fut in as_completed(futures, timeout=SCAN_TIMEOUT_S):
                try:
                    r = fut.result()
                    if r is not None:
                        results.append(r)
                except Exception:
                    continue
        except FuturesTimeoutError:
            pass
    return results

def scan_one(symbol: str, params: dict):
    try:
        prices, last_ts = fetch_yahoo(symbol, params['interval'], params['range'], params['ttl'])
        source = 'live'
        if len(prices) < params['min_bars']:
            raise ValueError('not enough live bars for this horizon')
    except Exception:
        prices = synthetic_series(max(params['min_bars'] + 10, 300), seed=abs(hash(symbol)) % 100000)
        last_ts = None
        source = 'synthetic_fallback'
    d = decision_for(symbol, prices, params['fast'], params['slow'], params['mom'])
    d['data_source'] = source
    d['as_of'] = last_ts
    return d

def run_scan(horizon: str):
    horizon = horizon if horizon in HORIZONS else 'intraday'
    params = HORIZONS[horizon]
    cache_key = f'scan:{horizon}'
    cached = _cached(cache_key, params['ttl'])
    if cached is not None:
        return cached
    results = run_pool(scan_one, NIFTY_100, params)
    results.sort(key=lambda d: d['score'], reverse=True)
    out = {
        'horizon': horizon, 'universe': 'NIFTY_100', 'count': len(results),
        'live_count': sum(1 for r in results if r['data_source'] == 'live'),
        'generated_at': time.time(), 'results': results,
    }
    _cache[cache_key] = (time.time(), out)
    return out

@app.get('/api/scan')
def scan(horizon: str = 'intraday'):
    return run_scan(horizon)

# ---------------------------------------------------------------------------
# Portfolio-wide backtest: does this strategy actually beat just holding the
# index? Walk-forward over ~2y of daily closes for every NIFTY 100 symbol
# (decision_for at bar i only ever sees prices[:i+1] — no lookahead), each
# compared against its own buy-and-hold return, plus the NIFTY 50 index
# (^NSEI — the closest liquid, reliably-available benchmark; NIFTY 100 has
# no equally reliable free index ticker) as the market-level bar to clear.
# cost_bps defaults to 0 (frictionless) — pass e.g. ?cost_bps=10 for a
# realistic ~0.10%-per-leg NSE cost estimate; a frictionless number is not
# what real trading would return, it's only a first check for any edge at
# all before costs are even considered.
# ---------------------------------------------------------------------------
BENCHMARK_SYMBOL = 'NIFTY'
PORTFOLIO_BACKTEST_TTL = 900.0

def backtest_symbol(symbol: str, params: dict, cost_bps: float):
    prices, _, source = get_history_prices(symbol)
    n = len(prices)
    cost_frac = cost_bps / 10000.0
    equity = CAPITAL; peak = equity; max_dd = 0.0; in_pos = False; entry = 0.0; trades_count = 0
    start_i = max(25, params['slow'] + 1)
    for i in range(min(start_i, n), n):
        d = decision_for(symbol, prices[:i + 1], params['fast'], params['slow'], params['mom'])
        px = float(prices[i])
        if d['action'] == 'BUY' and not in_pos:
            entry = px * (1 + cost_frac); in_pos = True; trades_count += 1
        elif d['action'] == 'EXIT' and in_pos:
            equity *= (px * (1 - cost_frac)) / entry
            in_pos = False; trades_count += 1
        peak = max(peak, equity); max_dd = max(max_dd, (peak - equity) / peak * 100)
    if in_pos:
        equity *= (float(prices[-1]) * (1 - cost_frac)) / entry
    strategy_return_pct = (equity / CAPITAL - 1) * 100
    buyhold_return_pct = (float(prices[-1]) / float(prices[0]) - 1) * 100
    return {
        'symbol': symbol, 'bars': n, 'trades': trades_count,
        'strategy_return_pct': round(strategy_return_pct, 2),
        'buyhold_return_pct': round(buyhold_return_pct, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'beat_buyhold': strategy_return_pct > buyhold_return_pct,
        'data_source': source,
    }

@app.get('/api/backtest_portfolio')
def backtest_portfolio(horizon: str = 'swing', cost_bps: float = 0.0):
    horizon = horizon if horizon in ('swing', 'invest') else 'swing'
    params = HORIZONS[horizon]
    cache_key = f'btport:{horizon}:{cost_bps}'
    cached = _cached(cache_key, PORTFOLIO_BACKTEST_TTL)
    if cached is not None:
        return cached

    results = run_pool(backtest_symbol, NIFTY_100, params, cost_bps)

    if not results:
        return {'horizon': horizon, 'universe': 'NIFTY_100', 'count': 0, 'results': []}

    bench_prices, _, bench_source = get_history_prices(BENCHMARK_SYMBOL, min_bars=50)
    benchmark_return_pct = round((float(bench_prices[-1]) / float(bench_prices[0]) - 1) * 100, 2)

    strat_returns = sorted(r['strategy_return_pct'] for r in results)
    bh_returns = [r['buyhold_return_pct'] for r in results]
    results.sort(key=lambda r: r['strategy_return_pct'], reverse=True)

    out = {
        'horizon': horizon, 'universe': 'NIFTY_100', 'count': len(results),
        'live_count': sum(1 for r in results if r['data_source'] == 'live'),
        'cost_bps_per_leg': cost_bps, 'frictionless': cost_bps == 0.0,
        'summary': {
            'avg_strategy_return_pct': round(sum(strat_returns) / len(strat_returns), 2),
            'median_strategy_return_pct': round(strat_returns[len(strat_returns) // 2], 2),
            'avg_buyhold_return_pct': round(sum(bh_returns) / len(bh_returns), 2),
            'win_rate_vs_buyhold_pct': round(sum(1 for r in results if r['beat_buyhold']) / len(results) * 100, 2),
            'benchmark_symbol': 'NIFTY 50 (^NSEI)', 'benchmark_return_pct': benchmark_return_pct,
            'benchmark_source': bench_source,
            'avg_trades_per_symbol': round(sum(r['trades'] for r in results) / len(results), 1),
            'avg_max_drawdown_pct': round(sum(r['max_drawdown_pct'] for r in results) / len(results), 2),
        },
        'results': results, 'generated_at': time.time(),
    }
    _cache[cache_key] = (time.time(), out)
    return out

# ---------------------------------------------------------------------------
# Autonomous tick. Vercel Cron hits GET /api/tick on a fixed schedule (see
# vercel.json) — this is what makes the agent act on its own instead of only
# reacting to button clicks. Cron only fires against the Production
# deployment, never PR previews, and only inside NSE market hours (checked
# here in IST, independent of whatever coarser window the cron schedule
# itself covers).
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))
MAX_OPEN_POSITIONS = 8
NEW_POSITION_CASH_FRACTION = 0.15

def market_open(now_ist: datetime) -> bool:
    if now_ist.weekday() >= 5:
        return False
    minutes = now_ist.hour * 60 + now_ist.minute
    return 9 * 60 + 15 <= minutes <= 15 * 60 + 30

@app.get('/api/tick')
def tick():
    now_ist = datetime.now(IST)
    if not market_open(now_ist):
        return {'status': 'market_closed', 'ist_time': now_ist.isoformat()}

    state, _ = load_state()
    scan_result = run_scan('intraday')
    by_symbol = {r['symbol']: r for r in scan_result['results']}
    executed = []

    for sym in list(state['positions'].keys()):
        d = by_symbol.get(sym)
        if d is None:
            prices, _, source = get_intraday_prices(sym)
            d = decision_for(sym, prices); d['data_source'] = source
        if d['action'] == 'EXIT':
            p = state['positions'].pop(sym)
            price = d['price']
            pnl = round((price - p['avg_price']) * p['qty'], 2)
            state['cash'] += p['qty'] * price
            state['trades'].append({'symbol': sym, 'side': 'SELL', 'qty': p['qty'], 'price': price, 'pnl': pnl, 'ts': time.time()})
            state['log'].append({'ts': time.time(), 'symbol': sym, 'action': 'EXIT', 'score': d['score'],
                                  'note': f"Exited {sym} x{p['qty']} @ ₹{price} (pnl ₹{pnl})"})
            executed.append({'symbol': sym, 'side': 'SELL', 'price': price, 'pnl': pnl})

    candidates = [r for r in scan_result['results'] if r['action'] == 'BUY' and r['symbol'] not in state['positions']]
    for d in candidates:
        if len(state['positions']) >= MAX_OPEN_POSITIONS:
            break
        sym, price = d['symbol'], d['price']
        qty = max(1, int((state['cash'] * NEW_POSITION_CASH_FRACTION) // price))
        cost = qty * price
        if qty > 0 and cost <= state['cash']:
            state['cash'] -= cost
            state['positions'][sym] = {'symbol': sym, 'qty': qty, 'avg_price': price, 'last_price': price, 'pnl': 0.0}
            state['log'].append({'ts': time.time(), 'symbol': sym, 'action': 'BUY', 'score': d['score'],
                                  'note': f"Bought {sym} x{qty} @ ₹{price} — {d['rationale']}"})
            executed.append({'symbol': sym, 'side': 'BUY', 'qty': qty, 'price': price})

    state['log'] = state['log'][-200:]
    persisted = save_state(state)
    return {'status': 'ok', 'ist_time': now_ist.isoformat(), 'executed': executed,
            'open_positions': len(state['positions']), 'persisted': persisted}

@app.get('/health')
def health(): return {'status':'ok','mode':'paper','live_trading':False}

@app.get('/api/quote')
def quote(symbol: str = 'RELIANCE'):
    prices, last_ts, source = get_intraday_prices(symbol)
    return {'symbol': symbol.upper(), 'yahoo_symbol': to_yahoo_symbol(symbol),
            'price': round(float(prices[-1]), 4), 'as_of': last_ts, 'data_source': source}

@app.get('/api/status')
def status():
    state, persisted = load_state()
    equity = state['cash'] + sum(p['qty'] * p['last_price'] for p in state['positions'].values())
    return {
        'paper_trading': True, 'live_trading': False, 'autonomous': True,
        'persisted': persisted, 'kv_connected': get_kv() is not None,
        'cash': round(state['cash'], 2), 'equity': round(equity, 2),
        'daily_pnl': round(equity - CAPITAL, 2), 'trades_today': len(state['trades']),
        'positions': list(state['positions'].values()), 'last_decision': state.get('last_decision'),
        'trade_log': state['trades'][-10:], 'activity_log': list(reversed(state['log'][-30:])),
    }

@app.post('/api/simulate')
def simulate(req: Request):
    state, _ = load_state()
    symbol = req.symbol.upper().replace('.NS', '')
    prices, last_ts, source = get_intraday_prices(symbol)
    d = decision_for(symbol, prices); d['data_source'] = source; d['as_of'] = last_ts
    state['last_decision'] = d
    price = float(prices[-1]); execution = {'status': 'NO_TRADE'}
    if d['action'] == 'BUY' and symbol not in state['positions']:
        qty = max(1, int((state['cash'] * 0.25) // price)); cost = qty * price
        if qty > 0 and cost <= state['cash']:
            state['cash'] -= cost
            state['positions'][symbol] = {'symbol': symbol, 'qty': qty, 'avg_price': round(price, 4), 'last_price': round(price, 4), 'pnl': 0.0}
            execution = {'status': 'FILLED', 'side': 'BUY', 'qty': qty, 'price': round(price, 4)}
            state['log'].append({'ts': time.time(), 'symbol': symbol, 'action': 'BUY', 'score': d['score'], 'note': f"Manual buy {symbol} x{qty} @ ₹{round(price,4)}"})
    elif symbol in state['positions']:
        p = state['positions'][symbol]; p['last_price'] = round(price, 4); p['pnl'] = round((price - p['avg_price']) * p['qty'], 2)
        if d['action'] == 'EXIT':
            state['cash'] += p['qty'] * price
            execution = {'status': 'FILLED', 'side': 'SELL', 'qty': p['qty'], 'price': round(price, 4), 'realized_pnl': p['pnl']}
            state['trades'].append({'symbol': symbol, 'side': 'SELL', 'qty': p['qty'], 'price': round(price, 4), 'pnl': p['pnl']})
            state['log'].append({'ts': time.time(), 'symbol': symbol, 'action': 'EXIT', 'score': d['score'], 'note': f"Manual exit {symbol} @ ₹{round(price,4)} (pnl ₹{p['pnl']})"})
            del state['positions'][symbol]
    persisted = save_state(state)
    return {'symbol': symbol, 'price': price, 'data_source': source, 'candidate': d, 'execution': execution, 'persisted': persisted}

@app.post('/api/backtest')
def backtest(req: Request):
    prices, last_ts, source = get_history_prices(req.symbol)
    equity=CAPITAL; peak=equity; max_dd=0.0; in_pos=False; entry=0.0; trades_count=0
    for i in range(25,len(prices)):
        d=decision_for(req.symbol.upper(),prices[:i+1]); px=float(prices[i])
        if d['action']=='BUY' and not in_pos: in_pos=True; entry=px; trades_count+=1
        elif d['action']=='EXIT' and in_pos: equity*=px/entry; in_pos=False; trades_count+=1
        peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak*100)
    if in_pos: equity*=prices[-1]/entry
    return {'start_equity':CAPITAL,'end_equity':round(equity,2),'trades':trades_count,'return_pct':round((equity/CAPITAL-1)*100,2),'max_drawdown_pct':round(max_dd,2),'bars_used':len(prices),'data_source':source}

@app.post('/api/reset')
def reset():
    persisted = save_state(default_state())
    return {'status': 'reset', 'capital': CAPITAL, 'persisted': persisted}
