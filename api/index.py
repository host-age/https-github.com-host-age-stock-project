from __future__ import annotations
from typing import Any
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np

app = FastAPI(title='NSE AI Trading Agent', version='1.1.0-paper')
CAPITAL = 10_000.0
cash = CAPITAL
positions: dict[str, dict[str, float]] = {}
trades: list[dict[str, Any]] = []
last_decision: dict[str, Any] | None = None

class Request(BaseModel):
    symbol: str = 'DEMO'
    bars: int = 250
    seed: int = 17

def series(bars: int, seed: int):
    bars = max(80, min(int(bars), 5000))
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00015, 0.012, bars)
    for i in range(bars):
        returns[i] += 0.00035 if (i // 90) % 2 else -0.00012
    return 100 * np.exp(np.cumsum(returns))

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

@app.get('/api/status')
def status():
    equity = cash + sum(p['qty']*p['last_price'] for p in positions.values())
    return {'paper_trading':True,'live_trading':False,'cash':round(cash,2),'equity':round(equity,2),'daily_pnl':round(equity-CAPITAL,2),'trades_today':len(trades),'positions':list(positions.values()),'last_decision':last_decision,'trade_log':trades[-10:]}

@app.post('/api/simulate')
def simulate(req: Request):
    global cash
    symbol=req.symbol.upper().replace('.NS',''); prices=series(req.bars,req.seed); d=decision_for(symbol,prices); price=float(prices[-1]); execution={'status':'NO_TRADE'}
    if d['action']=='BUY' and symbol not in positions:
        qty=max(1,int((cash*0.25)//price)); cost=qty*price
        if qty>0 and cost<=cash:
            cash-=cost; positions[symbol]={'symbol':symbol,'qty':qty,'avg_price':round(price,4),'last_price':round(price,4),'pnl':0.0}; execution={'status':'FILLED','side':'BUY','qty':qty,'price':round(price,4)}
    elif symbol in positions:
        p=positions[symbol]; p['last_price']=round(price,4); p['pnl']=round((price-p['avg_price'])*p['qty'],2)
        if d['action']=='EXIT':
            cash+=p['qty']*price; execution={'status':'FILLED','side':'SELL','qty':p['qty'],'price':round(price,4),'realized_pnl':p['pnl']}; trades.append({'symbol':symbol,'side':'SELL','qty':p['qty'],'price':round(price,4),'pnl':p['pnl']}); del positions[symbol]
    return {'symbol':symbol,'price':price,'candidate':d,'execution':execution}

@app.post('/api/backtest')
def backtest(req: Request):
    prices=series(max(req.bars,500),req.seed); equity=CAPITAL; peak=equity; max_dd=0.0; in_pos=False; entry=0.0; trades_count=0
    for i in range(25,len(prices)):
        d=decision_for(req.symbol.upper(),prices[:i+1]); px=float(prices[i])
        if d['action']=='BUY' and not in_pos: in_pos=True; entry=px; trades_count+=1
        elif d['action']=='EXIT' and in_pos: equity*=px/entry; in_pos=False; trades_count+=1
        peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak*100)
    if in_pos: equity*=prices[-1]/entry
    return {'start_equity':CAPITAL,'end_equity':round(equity,2),'trades':trades_count,'return_pct':round((equity/CAPITAL-1)*100,2),'max_drawdown_pct':round(max_dd,2)}

@app.post('/api/reset')
def reset():
    global cash,positions,trades,last_decision
    cash=CAPITAL; positions={}; trades=[]; last_decision=None
    return {'status':'reset','capital':CAPITAL}
