from __future__ import annotations
import time
from typing import Any
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import requests

app = FastAPI(title='NSE AI Trading Agent', version='2.0.0-paper')
CAPITAL = 10_000.0
cash = CAPITAL
positions: dict[str, dict[str, float]] = {}
trades: list[dict[str, Any]] = []
last_decision: dict[str, Any] | None = None

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

def decision_for(symbol: str, prices: np.ndarray):
    global last_decision
    fast = float(np.mean(prices[-5:])); slow = float(np.mean(prices[-20:]))
    vol = float(np.std(np.diff(np.log(prices[-20:]))))
    momentum = float((prices[-1] / prices[-10]) - 1)
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
    result = {'symbol':symbol,'action':action,'score':round(score,4),'probability_success':round(score,4),
              'expected_return':round(momentum*100,4),'downside':round(max(vol*100,0.1),4),'regime':regime,
              'rationale':'; '.join(reasons),'scenarios':{'bull':round(float(np.clip(score+0.10,0,1)),4),'base':round(score,4),'bear':round(float(np.clip(1-score+0.10,0,1)),4)},'price':round(float(prices[-1]),4)}
    last_decision = result
    return result

@app.get('/')
def root():
    return FileResponse(Path(__file__).resolve().parents[1] / 'public' / 'index.html')

@app.get('/health')
def health(): return {'status':'ok','mode':'paper','live_trading':False}

@app.get('/api/quote')
def quote(symbol: str = 'RELIANCE'):
    prices, last_ts, source = get_intraday_prices(symbol)
    return {'symbol': symbol.upper(), 'yahoo_symbol': to_yahoo_symbol(symbol),
            'price': round(float(prices[-1]), 4), 'as_of': last_ts, 'data_source': source}

@app.get('/api/status')
def status():
    equity = cash + sum(p['qty']*p['last_price'] for p in positions.values())
    return {'paper_trading':True,'live_trading':False,'cash':round(cash,2),'equity':round(equity,2),'daily_pnl':round(equity-CAPITAL,2),'trades_today':len(trades),'positions':list(positions.values()),'last_decision':last_decision,'trade_log':trades[-10:]}

@app.post('/api/simulate')
def simulate(req: Request):
    global cash
    symbol=req.symbol.upper().replace('.NS','')
    prices, last_ts, source = get_intraday_prices(symbol)
    d=decision_for(symbol,prices); d['data_source']=source; d['as_of']=last_ts
    price=float(prices[-1]); execution={'status':'NO_TRADE'}
    if d['action']=='BUY' and symbol not in positions:
        qty=max(1,int((cash*0.25)//price)); cost=qty*price
        if qty>0 and cost<=cash:
            cash-=cost; positions[symbol]={'symbol':symbol,'qty':qty,'avg_price':round(price,4),'last_price':round(price,4),'pnl':0.0}; execution={'status':'FILLED','side':'BUY','qty':qty,'price':round(price,4)}
    elif symbol in positions:
        p=positions[symbol]; p['last_price']=round(price,4); p['pnl']=round((price-p['avg_price'])*p['qty'],2)
        if d['action']=='EXIT':
            cash+=p['qty']*price; execution={'status':'FILLED','side':'SELL','qty':p['qty'],'price':round(price,4),'realized_pnl':p['pnl']}; trades.append({'symbol':symbol,'side':'SELL','qty':p['qty'],'price':round(price,4),'pnl':p['pnl']}); del positions[symbol]
    return {'symbol':symbol,'price':price,'data_source':source,'candidate':d,'execution':execution}

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
    global cash,positions,trades,last_decision
    cash=CAPITAL; positions={}; trades=[]; last_decision=None
    return {'status':'reset','capital':CAPITAL}
