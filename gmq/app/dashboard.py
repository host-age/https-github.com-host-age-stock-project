"""Session dashboard: a self-contained HTML report built from a run report.

Design note: the visual language borrows from chess *analysis* interfaces --
an evaluation bar, ranked candidate moves with scores, a move log -- because
that is genuinely what this engine does, not because it is decorative. The
eval bar at the top answers the only question that matters at a glance: was
this run sound, and if not, which check failed.

No external assets. Charts are inline SVG. The page is theme-aware and works
in light, dark and the un-stamped system default.
"""
from __future__ import annotations

import html
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# small SVG helpers
# --------------------------------------------------------------------------


def _sparkline(values: Sequence[float], w: int = 720, h: int = 180,
               base: Optional[float] = None) -> str:
    """Equity curve with an area fill and an emphasised endpoint."""
    if len(values) < 2:
        return '<p class="muted">not enough data</p>'
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.12
    lo -= pad
    hi += pad
    n = len(values)

    def x(i): return 6 + i * (w - 12) / (n - 1)
    def y(v): return h - 8 - (v - lo) / (hi - lo) * (h - 22)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    area = (f'M {x(0):.1f},{h - 8:.1f} L ' +
            " L ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values)) +
            f" L {x(n-1):.1f},{h - 8:.1f} Z")
    base_line = ""
    if base is not None and lo <= base <= hi:
        by = y(base)
        base_line = (f'<line x1="6" y1="{by:.1f}" x2="{w-6}" y2="{by:.1f}" '
                     f'class="grid-base"/>')
    last_x, last_y = x(n - 1), y(values[-1])
    up = values[-1] >= (base if base is not None else values[0])
    cls = "up" if up else "down"
    return f"""<svg viewBox="0 0 {w} {h}" class="chart {cls}" role="img"
 aria-label="equity curve" preserveAspectRatio="none">
  <path d="{area}" class="area"/>
  {base_line}
  <polyline points="{pts}" class="line"/>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.5" class="endpoint"/>
</svg>"""


def _bars(pairs: List[Tuple[str, float]], w: int = 620, bar_h: int = 22,
          fmt: str = "{:.2f}", diverging: bool = False) -> str:
    if not pairs:
        return '<p class="muted">no data</p>'
    vals = [v for _k, v in pairs]
    mx = max(abs(v) for v in vals) or 1.0
    h = len(pairs) * (bar_h + 8) + 6
    rows = []
    label_w = 168
    track = w - label_w - 76
    for i, (k, v) in enumerate(pairs):
        yy = 6 + i * (bar_h + 8)
        if diverging:
            mid = label_w + track / 2
            bw = abs(v) / mx * (track / 2)
            bx = mid if v >= 0 else mid - bw
            cls = "pos" if v >= 0 else "neg"
        else:
            bx = label_w
            bw = abs(v) / mx * track
            cls = "pos" if v >= 0 else "neg"
        rows.append(
            f'<text x="0" y="{yy + bar_h * 0.72:.0f}" class="blabel">'
            f'{html.escape(k[:24])}</text>'
            f'<rect x="{bx:.1f}" y="{yy}" width="{max(bw,1.5):.1f}" '
            f'height="{bar_h}" rx="2" class="bar {cls}"/>'
            f'<text x="{w - 4}" y="{yy + bar_h * 0.72:.0f}" class="bval">'
            f'{fmt.format(v)}</text>')
    mid_line = ""
    if diverging:
        mid = label_w + track / 2
        mid_line = f'<line x1="{mid}" y1="0" x2="{mid}" y2="{h}" class="grid-base"/>'
    return (f'<svg viewBox="0 0 {w} {h}" class="barchart" role="img">'
            f'{mid_line}{"".join(rows)}</svg>')


def _calibration_plot(rows: List[dict], w: int = 340, h: int = 300) -> str:
    """Predicted vs observed. The diagonal is the truth; distance from it is
    the size of the model's self-deception."""
    if not rows:
        return '<p class="muted">not enough resolved outcomes yet</p>'
    pad = 38
    def px(v): return pad + v * (w - pad - 12)
    def py(v): return h - pad - v * (h - pad - 12)
    pts = []
    for r in rows:
        p, o, n = r.get("predicted", r.get("pred", 0)), \
            r.get("observed", r.get("obs", 0)), r.get("n", 1)
        rad = 3 + min(math.sqrt(n), 9)
        pts.append(f'<circle cx="{px(p):.1f}" cy="{py(o):.1f}" r="{rad:.1f}" '
                   f'class="calpt"><title>predicted {p:.2f} / observed '
                   f'{o:.2f} (n={n})</title></circle>')
    path = " L ".join(f"{px(r.get('predicted', r.get('pred',0))):.1f},"
                      f"{py(r.get('observed', r.get('obs',0))):.1f}"
                      for r in rows)
    return f"""<svg viewBox="0 0 {w} {h}" class="chart cal" role="img"
 aria-label="calibration">
  <line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" class="diag"/>
  <line x1="{pad}" y1="{h-pad}" x2="{w-12}" y2="{h-pad}" class="axis"/>
  <line x1="{pad}" y1="{h-pad}" x2="{pad}" y2="12" class="axis"/>
  <path d="M {path}" class="calline"/>
  {''.join(pts)}
  <text x="{(w+pad)/2}" y="{h-8}" class="axlabel" text-anchor="middle">predicted</text>
  <text x="12" y="{(h)/2}" class="axlabel" text-anchor="middle"
   transform="rotate(-90 12 {h/2})">observed</text>
</svg>"""


# --------------------------------------------------------------------------


def _fmt(v: Any, nd: int = 2, pct: bool = False, sign: bool = False) -> str:
    if v is None:
        return "&mdash;"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "&mdash;"
        s = f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}"
        return s + ("%" if pct else "")
    return html.escape(str(v))


def _tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="tile {tone}"><span class="tl">{label}</span>'
            f'<span class="tv">{value}</span>'
            f'<span class="ts">{sub}</span></div>')


def _table(headers: List[str], rows: List[List[str]],
           cls: str = "") -> str:
    if not rows:
        return '<p class="muted">nothing recorded</p>'
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                 for r in rows)
    return (f'<div class="scroll"><table class="{cls}">'
            f"<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>")


def _verdict(rep: dict) -> Tuple[str, str, str, float]:
    """Overall read on the run. Returns (label, tone, explanation, eval -1..1).

    Deliberately conservative: a short simulated run with few trades is
    reported as inconclusive, not as a result. The most common way a system
    like this misleads its author is by presenting a small, lucky sample as
    evidence.
    """
    t = rep.get("trades", {}) or {}
    e = rep.get("equity", {}) or {}
    n = t.get("trades", 0)
    sh = e.get("sharpe", 0.0) or 0.0
    dd = e.get("max_drawdown_pct", 0.0) or 0.0
    halts = [h.get("reason") for h in (rep.get("halts") or [])]
    best_share = t.get("best_trade_share", 0.0) or 0.0

    if n < 30:
        return ("Inconclusive", "warn",
                f"{n} closed trades is far too small a sample to distinguish "
                f"skill from luck. Nothing here is evidence either way.", 0.0)
    if halts:
        return ("Halted", "warn",
                f"The risk engine stopped trading ({', '.join(sorted(set(halts)))}). "
                f"That is the control working, but the run is not a clean read.",
                -0.15)
    if best_share > 0.45:
        return ("Single-trade dependent", "warn",
                f"{best_share:.0%} of the P&L came from one trade. That is a "
                f"sample of one wearing the costume of a strategy.", -0.1)
    if sh > 0.8 and dd < 5:
        return ("Positive, unverified", "good",
                "Risk-adjusted return is positive and drawdown contained in "
                "this simulated sample. It has not been validated against real "
                "market data.", min(0.75, sh / 3))
    if sh > 0:
        return ("Marginal", "neutral",
                "A small positive edge in simulation, within the range noise "
                "produces at this sample size.", min(0.35, sh / 4))
    return ("Negative", "bad",
            "The strategy lost money in this run after costs.",
            max(-0.8, sh / 3))


def build_dashboard(rep: dict, out_path: str,
                    journal: Any = None,
                    equity: Optional[List[Tuple[int, float]]] = None,
                    extras: Optional[dict] = None) -> str:
    t = rep.get("trades", {}) or {}
    e = rep.get("equity", {}) or {}
    lat = rep.get("latency", {}) or {}
    ex = rep.get("execution", {}) or {}
    risk = rep.get("risk", {}) or {}
    models = rep.get("models", {}) or {}
    extras = extras or {}

    label, tone, why, ev = _verdict(rep)
    ev_pct = (ev + 1) / 2 * 100

    # ---- equity series
    series = extras.get("equity_series") or []
    if not series and equity:
        series = [v for _ts, v in equity]
    eq_svg = _sparkline(series, base=series[0] if series else None) \
        if len(series) > 2 else '<p class="muted">equity curve not captured</p>'

    # ---- honest model skill
    honest = models.get("honest_skill", {}) or {}
    skill_rows = []
    for h, s in sorted(honest.items(), key=lambda kv: float(kv[0])):
        skill_rows.append([
            f"{int(float(h))}s",
            _fmt(s.get("n"), 0),
            _fmt(s.get("brier"), 4),
            f'<span class="{"pos" if (s.get("skill") or 0) > 0 else "neg"}">'
            f'{_fmt(s.get("skill"), 4, sign=True)}</span>',
            _fmt((s.get("accuracy") or 0) * 100, 1, pct=True),
        ])

    # ---- per-regime
    reg_rows = []
    for k, v in sorted((rep.get("by_regime") or {}).items(),
                       key=lambda kv: -(kv[1].get("trades", 0))):
        if not v.get("trades"):
            continue
        reg_rows.append([
            html.escape(k),
            _fmt(v.get("trades"), 0),
            _fmt((v.get("win_rate") or 0) * 100, 1, pct=True),
            f'<span class="{"pos" if (v.get("expectancy_r") or 0) > 0 else "neg"}">'
            f'{_fmt(v.get("expectancy_r"), 3, sign=True)}R</span>',
            _fmt(v.get("total_pnl"), 0, sign=True),
        ])

    # ---- loss-cause classification
    causes = rep.get("loss_causes", {}) or {}
    cause_bars = _bars([(k.replace("_", " ").title(), float(v))
                        for k, v in sorted(causes.items(),
                                           key=lambda kv: -kv[1])],
                       fmt="{:.0f}") if causes else \
        '<p class="muted">no adverse-move diagnoses recorded</p>'

    # ---- excursions
    exc = rep.get("excursions", {}) or {}

    # ---- calibration
    cal = rep.get("decision_calibration") or []
    cal_svg = _calibration_plot(cal)

    # ---- verification / limitations
    checks = extras.get("checks") or []
    check_rows = [[
        ("✓" if c.get("pass") else "✗"),
        html.escape(c.get("check", "").replace("_", " ")),
        html.escape(c.get("detail", "")),
    ] for c in checks]

    title = extras.get("title", "Grandmaster Engine Report")
    run_id = rep.get("run_id", "run")
    days = rep.get("days", 0)
    wall = rep.get("wall_s", 0)

    css = _CSS
    body = f"""
<header class="top">
  <div class="brand">
    <span class="mark" aria-hidden="true">&#9822;</span>
    <div>
      <h1>Grandmaster Engine</h1>
      <p class="sub">Run <code>{html.escape(str(run_id))}</code> &middot;
      {_fmt(days,0)} simulated session{'s' if (days or 0) != 1 else ''} &middot;
      {_fmt(wall,0)}s wall clock</p>
    </div>
  </div>
  <div class="verdict {tone}">
    <span class="vlabel">{html.escape(label)}</span>
    <div class="evalbar" role="img"
         aria-label="overall evaluation {ev:+.2f} on a -1 to +1 scale">
      <div class="evalfill" style="height:{ev_pct:.1f}%"></div>
    </div>
  </div>
</header>

<p class="banner">
  <strong>Simulated results.</strong> Every number on this page comes from a
  synthetic market built for this project. It validates that the code works,
  not that the strategy makes money. No real orders were placed and none can
  be without a deliberate, multi-part opt-in.
</p>

<p class="why {tone}">{html.escape(why)}</p>

<section>
  <h2>Result</h2>
  <div class="tiles">
    {_tile("Return", _fmt(e.get('total_return_pct'), 3, pct=True, sign=True),
           "net of all costs",
           "good" if (e.get('total_return_pct') or 0) > 0 else "bad")}
    {_tile("Sharpe", _fmt(e.get('sharpe'), 2), "annualised")}
    {_tile("Sortino", _fmt(e.get('sortino'), 2), "downside only")}
    {_tile("Max drawdown", _fmt(e.get('max_drawdown_pct'), 2, pct=True),
           "peak to trough", "bad" if (e.get('max_drawdown_pct') or 0) > 5 else "")}
    {_tile("Trades", _fmt(t.get('trades'), 0),
           f"{_fmt((t.get('win_rate') or 0)*100,1)}% won")}
    {_tile("Expectancy", _fmt(t.get('expectancy_r'), 3, sign=True) + "R",
           "per trade, in risk units",
           "good" if (t.get('expectancy_r') or 0) > 0 else "bad")}
    {_tile("Profit factor", _fmt(t.get('profit_factor'), 2), "gross win / gross loss")}
    {_tile("Worst trade share", _fmt((t.get('best_trade_share') or 0)*100, 1, pct=True),
           "of total P&amp;L from one trade",
           "warn" if (t.get('best_trade_share') or 0) > 0.4 else "")}
  </div>
  <div class="card">
    <h3>Equity</h3>
    {eq_svg}
  </div>
</section>

<section>
  <h2>Did the models actually know anything?</h2>
  <p class="lead">Brier <em>skill</em>, not accuracy. Skill is measured against
  the base rate, so a model that is 88% accurate in a market that rises 88% of
  the time scores zero &mdash; which is the correct answer. Samples are spaced a
  full horizon apart so their label windows never overlap; the overlapping
  figure would be roughly twice as flattering and entirely fictional.</p>
  <div class="split">
    <div class="card">
      <h3>Forecast quality by horizon</h3>
      {_table(["Horizon", "n", "Brier", "Skill", "Accuracy"], skill_rows)}
    </div>
    <div class="card">
      <h3>Decision calibration</h3>
      <p class="note">When the engine said 70%, how often was it right? Points
      on the diagonal are honest; points below it mean overconfidence, which
      makes every expected value downstream too high.</p>
      {cal_svg}
    </div>
  </div>
</section>

<section>
  <h2>How it behaved</h2>
  <div class="split">
    <div class="card">
      <h3>Performance by detected regime</h3>
      <p class="note">A strategy that only works in one regime is fine. One
      that is allowed to trade in all of them is not.</p>
      {_table(["Regime", "Trades", "Win rate", "Expectancy", "P&amp;L"], reg_rows)}
    </div>
    <div class="card">
      <h3>Why positions went against it</h3>
      <p class="note">Every adverse move is classified before the engine
      responds. &ldquo;It&rsquo;s down&rdquo; is not a diagnosis, and normal
      volatility and an invalidated thesis call for opposite actions.</p>
      {cause_bars}
    </div>
  </div>
</section>

<section>
  <h2>Execution and risk</h2>
  <div class="tiles">
    {_tile("Decisions", _fmt(rep.get('decisions'), 0),
           f"{_fmt(rep.get('actions'),0)} acted on")}
    {_tile("Search latency", _fmt(lat.get('mean_search_ms'), 2) + " ms",
           f"p99 {_fmt(lat.get('p99_search_ms'),2)} ms")}
    {_tile("Fill rate", _fmt((ex.get('fill_rate') or 0)*100, 1, pct=True),
           f"{_fmt(ex.get('rejected'),0)} rejected")}
    {_tile("Slippage", _fmt(ex.get('mean_slippage_bps'), 2, sign=True) + " bps",
           f"p95 {_fmt(ex.get('p95_slippage_bps'),2)} bps")}
    {_tile("Risk vetoes", _fmt(risk.get('vetoes'), 0),
           f"{_fmt(risk.get('scale_downs'),0)} orders cut down")}
    {_tile("Position mismatches", _fmt(ex.get('discrepancies'), 0),
           "belief vs broker", "bad" if (ex.get('discrepancies') or 0) else "good")}
    {_tile("Halts", str(len(rep.get('halts') or [])),
           ", ".join(sorted({h.get('reason','') for h in (rep.get('halts') or [])})) or "none",
           "warn" if rep.get('halts') else "good")}
    {_tile("Fees paid", _fmt(extras.get('fees'), 0),
           "STT, brokerage, impact")}
  </div>
  <div class="card">
    <h3>Where the stops belong</h3>
    <p class="note">Maximum adverse excursion on winners versus maximum
    favourable excursion on losers. If winners routinely take more heat than
    the stop allows, the stop is converting winners into losers &mdash; and no
    aggregate P&amp;L number will ever show you that.</p>
    {_table(["Measure", "Value"],
            [[html.escape(k.replace('_', ' ')), _fmt(v, 2)]
             for k, v in sorted(exc.items())])}
  </div>
</section>

{'<section><h2>What was verified</h2><p class="lead">Structural checks on the '
 'machinery, not the results &mdash; a leaked backtest looks excellent, which '
 'is exactly the problem with it.</p>' +
 _table(["", "Check", "How"], check_rows) + '</section>' if check_rows else ''}

<section class="limits">
  <h2>What this does not show</h2>
  <ul>
    <li><strong>The market is a model.</strong> Prices emerge from a real
      matching engine driven by simulated market makers and informed traders.
      It has no earnings gaps, no exchange outages, no auction mechanics, no
      settlement and no corporate actions.</li>
    <li><strong>Costs are modelled, not measured.</strong> STT, brokerage,
      stamp duty, GST and square-root impact are all charged, and stop fills
      are slipped past their trigger &mdash; but every one of those constants
      is an estimate, not an observation from a live account.</li>
    <li><strong>The sample is small.</strong> A few simulated sessions cannot
      separate a real edge from a lucky one. The walk-forward and Monte Carlo
      tooling in the repository exists precisely because a single backtest
      number is not evidence.</li>
    <li><strong>Nothing here transfers to live trading unexamined.</strong>
      Latency, fill probability and impact are calibrated to this simulator.
      They would all have to be re-measured against a real venue first.</li>
  </ul>
</section>

<footer>
  <p>Generated by the Grandmaster Engine &middot; every decision on this page
  has a full audit record in <code>{html.escape(str(extras.get('journal_path', 'runs/&lt;run&gt;/*.jsonl')))}</code></p>
</footer>
"""
    doc = f"""<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>{css}</style>
<main class="wrap">{body}</main>
"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(doc)
    return out_path


_CSS = r"""
:root{
  --ground:#F3F5F7; --panel:#FFFFFF; --panel-2:#EAEEF2;
  --ink:#111820; --ink-2:#43505E; --ink-3:#6E7C8B;
  --rule:#D5DDE5; --rule-2:#C2CDD8;
  --accent:#0E7C86; --accent-soft:#D6EDEF;
  --pos:#0F7A52; --neg:#B23A32; --warn:#9A6B08;
  --pos-soft:#DBEFE5; --neg-soft:#F6DEDC; --warn-soft:#F6EBD3;
  --shadow:0 1px 2px rgba(17,24,32,.06),0 8px 24px rgba(17,24,32,.05);
}
:root:not([data-theme="light"]){
  @media (prefers-color-scheme: dark){
    --ground:#0E1116; --panel:#161B22; --panel-2:#1D242D;
    --ink:#E6EDF3; --ink-2:#A9B6C3; --ink-3:#79879A;
    --rule:#262E38; --rule-2:#333D49;
    --accent:#3FB6C0; --accent-soft:#12333A;
    --pos:#3DBE8B; --neg:#E8776D; --warn:#D8A63C;
    --pos-soft:#10281F; --neg-soft:#2E1917; --warn-soft:#2A2113;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0E1116; --panel:#161B22; --panel-2:#1D242D;
  --ink:#E6EDF3; --ink-2:#A9B6C3; --ink-3:#79879A;
  --rule:#262E38; --rule-2:#333D49;
  --accent:#3FB6C0; --accent-soft:#12333A;
  --pos:#3DBE8B; --neg:#E8776D; --warn:#D8A63C;
  --pos-soft:#10281F; --neg-soft:#2E1917; --warn-soft:#2A2113;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.35);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1060px; margin:0 auto; padding:28px 22px 64px;
  display:flex; flex-direction:column; gap:34px}

h1,h2,h3{font-family:"Bricolage Grotesque","Public Sans",sans-serif;
  text-wrap:balance; margin:0; letter-spacing:-.015em}
h1{font-size:clamp(1.5rem,3.2vw,2rem); font-weight:800; line-height:1.1}
h2{font-size:1.28rem; font-weight:600; margin-bottom:2px}
h3{font-size:.95rem; font-weight:600; color:var(--ink-2);
  text-transform:uppercase; letter-spacing:.07em}
p{margin:0}
code,.mono,td.num,th.num{font-family:"JetBrains Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
code{font-size:.86em; background:var(--panel-2); padding:.1em .38em;
  border-radius:3px}

/* ---------- header ---------- */
.top{display:flex; justify-content:space-between; align-items:stretch;
  gap:20px; flex-wrap:wrap; padding-bottom:20px;
  border-bottom:1px solid var(--rule)}
.brand{display:flex; gap:14px; align-items:center}
.mark{font-size:2.4rem; line-height:1; color:var(--accent)}
.sub{color:var(--ink-3); font-size:.86rem; margin-top:3px}
.verdict{display:flex; align-items:center; gap:12px}
.vlabel{font-family:"Bricolage Grotesque",sans-serif; font-weight:600;
  font-size:.95rem; padding:.34em .8em; border-radius:100px;
  border:1px solid var(--rule-2); background:var(--panel);
  white-space:nowrap}
.verdict.good .vlabel{background:var(--pos-soft); color:var(--pos);
  border-color:transparent}
.verdict.bad .vlabel{background:var(--neg-soft); color:var(--neg);
  border-color:transparent}
.verdict.warn .vlabel{background:var(--warn-soft); color:var(--warn);
  border-color:transparent}
/* the chess eval bar: white advantage rises from the bottom */
.evalbar{width:16px; height:56px; border-radius:4px; overflow:hidden;
  background:var(--ink); border:1px solid var(--rule-2);
  display:flex; align-items:flex-end}
.evalfill{width:100%; background:var(--panel);
  border-top:2px solid var(--accent); transition:height .5s ease}

.banner{background:var(--warn-soft); color:var(--warn);
  border-left:3px solid var(--warn); padding:12px 16px; border-radius:0 6px 6px 0;
  font-size:.9rem}
.banner strong{color:var(--warn)}
.why{font-size:1.02rem; color:var(--ink-2); max-width:66ch}
.why.good{color:var(--pos)} .why.bad{color:var(--neg)} .why.warn{color:var(--warn)}
.lead{color:var(--ink-2); max-width:70ch; font-size:.94rem; margin-top:6px}
.note{color:var(--ink-3); font-size:.84rem; max-width:62ch; margin:2px 0 10px}
.muted{color:var(--ink-3); font-size:.88rem; font-style:italic}

section{display:flex; flex-direction:column; gap:14px}

/* ---------- tiles ---------- */
.tiles{display:grid; gap:12px;
  grid-template-columns:repeat(auto-fit,minmax(168px,1fr))}
.tile{background:var(--panel); border:1px solid var(--rule); border-radius:8px;
  padding:13px 15px; display:flex; flex-direction:column; gap:3px;
  box-shadow:var(--shadow)}
.tl{font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--ink-3); font-weight:600}
.tv{font-family:"JetBrains Mono",monospace; font-size:1.32rem; font-weight:500;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em}
.ts{font-size:.76rem; color:var(--ink-3)}
.tile.good .tv{color:var(--pos)} .tile.bad .tv{color:var(--neg)}
.tile.warn .tv{color:var(--warn)}
.tile.good{border-left:3px solid var(--pos)}
.tile.bad{border-left:3px solid var(--neg)}
.tile.warn{border-left:3px solid var(--warn)}

/* ---------- cards ---------- */
.card{background:var(--panel); border:1px solid var(--rule); border-radius:8px;
  padding:16px 18px; display:flex; flex-direction:column; gap:8px;
  box-shadow:var(--shadow); min-width:0}
.split{display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}

/* ---------- tables ---------- */
.scroll{overflow-x:auto; margin:0 -2px}
table{width:100%; border-collapse:collapse; font-size:.87rem}
th{text-align:left; font-weight:600; font-size:.72rem; text-transform:uppercase;
  letter-spacing:.07em; color:var(--ink-3); padding:6px 10px 6px 0;
  border-bottom:1px solid var(--rule-2); white-space:nowrap}
td{padding:7px 10px 7px 0; border-bottom:1px solid var(--rule);
  font-family:"JetBrains Mono",monospace; font-variant-numeric:tabular-nums;
  font-size:.84rem; vertical-align:top}
td:nth-child(2),td:nth-child(3){white-space:normal}
tbody tr:last-child td{border-bottom:none}
.pos{color:var(--pos)} .neg{color:var(--neg)}

/* ---------- charts ---------- */
.chart{width:100%; height:auto; display:block; overflow:visible}
.chart .area{fill:var(--accent); opacity:.10}
.chart .line{fill:none; stroke:var(--accent); stroke-width:1.8;
  stroke-linejoin:round; vector-effect:non-scaling-stroke}
.chart.down .line{stroke:var(--neg)} .chart.down .area{fill:var(--neg)}
.chart .endpoint{fill:var(--accent); stroke:var(--panel); stroke-width:2}
.chart.down .endpoint{fill:var(--neg)}
.grid-base{stroke:var(--rule-2); stroke-width:1; stroke-dasharray:3 4}
.cal .diag{stroke:var(--ink-3); stroke-dasharray:4 4; stroke-width:1}
.cal .axis{stroke:var(--rule-2); stroke-width:1}
.cal .calline{fill:none; stroke:var(--accent); stroke-width:1.6}
.cal .calpt{fill:var(--accent); opacity:.75}
.axlabel{fill:var(--ink-3); font-size:11px;
  font-family:"Public Sans",sans-serif}
.barchart{width:100%; height:auto; display:block}
.barchart .bar.pos{fill:var(--accent)} .barchart .bar.neg{fill:var(--neg)}
.blabel{fill:var(--ink-2); font-size:12px; font-family:"Public Sans",sans-serif}
.bval{fill:var(--ink-3); font-size:11px; text-anchor:end;
  font-family:"JetBrains Mono",monospace}

/* ---------- limits ---------- */
.limits ul{margin:0; padding-left:1.1rem; display:flex; flex-direction:column;
  gap:9px; color:var(--ink-2); max-width:74ch; font-size:.92rem}
.limits strong{color:var(--ink)}
footer{border-top:1px solid var(--rule); padding-top:16px; color:var(--ink-3);
  font-size:.82rem}

@media (max-width:640px){
  .wrap{padding:18px 14px 44px}
  .top{flex-direction:column; gap:14px}
  .tv{font-size:1.15rem}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""
