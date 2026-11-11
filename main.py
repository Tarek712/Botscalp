import asyncio
import json
import os
import time
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from http import HTTPStatus
from typing import List, Optional, Set, Dict
import numpy as np
import websockets

# ---------------------------------------------------------
# 1. الموديلات وهياكل البيانات
# ---------------------------------------------------------
@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    def to_dict(self) -> Dict:
        return {
            "time": self.timestamp,
            "open": round(self.open, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "close": round(self.close, 2),
            "volume": round(self.volume, 2)
        }

class BotState(Enum):
    IDLE = "IDLE"
    SCANNING_M5 = "SCANNING_M5"
    WAITING_M1 = "WAITING_M1"
    HUNTING_S5 = "HUNTING_S5"
    ORDER_PENDING = "ORDER_PENDING"
    IN_TRADE = "IN_TRADE"
    HALTED = "HALTED"

# ---------------------------------------------------------
# 2. محرك الاستراتيجية
# ---------------------------------------------------------
class StrategyEngine:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.fixed_lot = 0.01
        self.state = BotState.IDLE
        self.current_bias: Optional[str] = None
        self.dxy_status = "NEUTRAL"
        self.active_order = None
        self.active_position = None
        self.m1_start_time = 0
        self.s5_start_time = 0
        self.order_start_time = 0

    @staticmethod
    def calculate_atr(candles: List[Candle], period: int = 14) -> float:
        if len(candles) < period + 1:
            return 1.20
        trs = [max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close)) 
               for c, prev in zip(candles[1:], candles[:-1])]
        return float(np.mean(trs[-period:]))

    @staticmethod
    def calculate_ema(candles: List[Candle], period: int = 20) -> float:
        closes = [c.close for c in candles]
        if len(closes) < period:
            return closes[-1] if closes else 0.0
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        a = np.convolve(closes, weights, mode='full')[:len(closes)]
        a[:period] = a[period]
        return float(a[-1])

    def get_m5_market_structure(self, m5_candles: List[Candle]) -> str:
        if len(m5_candles) < 25:
            return "NEUTRAL"
        swing_highs, swing_lows = [], []
        for i in range(2, len(m5_candles) - 2):
            c = m5_candles[i]
            if c.high > m5_candles[i-1].high and c.high > m5_candles[i-2].high and c.high > m5_candles[i+1].high and c.high > m5_candles[i+2].high:
                swing_highs.append(c.high)
            if c.low < m5_candles[i-1].low and c.low < m5_candles[i-2].low and c.low < m5_candles[i+1].low and c.low < m5_candles[i+2].low:
                swing_lows.append(c.low)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "NEUTRAL"
        last_close = m5_candles[-1].close
        ema = self.calculate_ema(m5_candles, 20)
        if (swing_lows[-1] > swing_lows[-2] and last_close > ema) or (last_close > swing_highs[-1]):
            return "BULLISH"
        if (swing_highs[-1] < swing_highs[-2] and last_close < ema) or (last_close < swing_lows[-1]):
            return "BEARISH"
        return "NEUTRAL"

    def check_dxy(self, dxy_m5: List[Candle], gold_m5: List[Candle]) -> str:
        if len(dxy_m5) < 20 or len(gold_m5) < 20:
            return "NEUTRAL"
        dxy_roc = (dxy_m5[-1].close - dxy_m5[-4].close) / max(dxy_m5[-4].close, 1e-5)
        dxy_ema = self.calculate_ema(dxy_m5, 20)
        if dxy_m5[-1].close < dxy_ema and dxy_roc < 0:
            return "DXY_BEARISH"
        elif dxy_m5[-1].close > dxy_ema and dxy_roc > 0:
            return "DXY_BULLISH"
        return "NEUTRAL"

    def detect_m1(self, m1_candles: List[Candle], bias: str) -> bool:
        if len(m1_candles) < 10:
            return False
        curr, prev = m1_candles[-1], m1_candles[-2]
        if bias == "BULLISH":
            return (curr.is_bullish and prev.is_bearish and curr.close > prev.open) or (curr.lower_wick >= 0.6 * curr.total_range)
        elif bias == "BEARISH":
            return (curr.is_bearish and prev.is_bullish and curr.close < prev.open) or (curr.upper_wick >= 0.6 * curr.total_range)
        return False

    def detect_s5(self, s5_candles: List[Candle], bias: str):
        if len(s5_candles) < 6:
            return None
        curr = s5_candles[-1]
        recent_lows = [c.low for c in s5_candles[-6:-1]]
        recent_highs = [c.high for c in s5_candles[-6:-1]]
        if bias == "BULLISH" and curr.low <= min(recent_lows) and curr.lower_wick >= 1.8 * max(curr.body, 0.04) and curr.is_bullish:
            return {"limit_price": round(curr.low + 0.5 * curr.lower_wick, 2), "extreme": curr.low}
        if bias == "BEARISH" and curr.high >= max(recent_highs) and curr.upper_wick >= 1.8 * max(curr.body, 0.04) and curr.is_bearish:
            return {"limit_price": round(curr.high - 0.5 * curr.upper_wick, 2), "extreme": curr.high}
        return None

    def evaluate(self, now: int, bid: float, ask: float, gold_m5, dxy_m5, gold_m1, gold_s5):
        if self.state == BotState.ORDER_PENDING and self.active_order:
            if now - self.order_start_time > 30:
                self.active_order = None
                self.state = BotState.SCANNING_M5
            return

        if self.state in [BotState.IDLE, BotState.SCANNING_M5]:
            self.state = BotState.SCANNING_M5
            m5_bias = self.get_m5_market_structure(gold_m5)
            self.dxy_status = self.check_dxy(dxy_m5, gold_m5)
            if m5_bias == "BULLISH" and self.dxy_status == "DXY_BEARISH":
                self.current_bias = "BULLISH"
                self.state = BotState.WAITING_M1
                self.m1_start_time = now
            elif m5_bias == "BEARISH" and self.dxy_status == "DXY_BULLISH":
                self.current_bias = "BEARISH"
                self.state = BotState.WAITING_M1
                self.m1_start_time = now

        elif self.state == BotState.WAITING_M1:
            if now - self.m1_start_time > 900:
                self.state = BotState.SCANNING_M5
            elif self.detect_m1(gold_m1, self.current_bias):
                self.state = BotState.HUNTING_S5
                self.s5_start_time = now

        elif self.state == BotState.HUNTING_S5:
            if now - self.s5_start_time > 180:
                self.state = BotState.SCANNING_M5
            else:
                sig = self.detect_s5(gold_s5, self.current_bias)
                if sig:
                    atr = self.calculate_atr(gold_m5, 14)
                    entry = sig["limit_price"]
                    extreme = sig["extreme"]
                    if self.current_bias == "BULLISH":
                        dist = max(1.5 * atr, abs(entry - extreme) + 0.20)
                        sl, tp = round(entry - dist, 2), round(entry + 3.0 * dist, 2)
                        otype = "BUY_LIMIT"
                    else:
                        dist = max(1.5 * atr, abs(extreme - entry) + 0.20)
                        sl, tp = round(entry + dist, 2), round(entry - 3.0 * dist, 2)
                        otype = "SELL_LIMIT"
                    self.active_order = {
                        "id": f"ORD-{now}", "type": otype, "symbol": "XAUUSD", "lot": 0.01,
                        "price": entry, "sl": sl, "tp": tp, "risk_dist": round(dist, 2), "created_at": now
                    }
                    self.order_start_time = now
                    self.state = BotState.ORDER_PENDING

# ---------------------------------------------------------
# 3. مجمع الشموع والمحاكي
# ---------------------------------------------------------
class TimeBucket:
    def __init__(self, interval: int, max_len: int = 150):
        self.interval = interval
        self.max_len = max_len
        self.candles: List[Candle] = []
        self.curr: Optional[Candle] = None

    def update(self, ts: int, price: float):
        b_ts = (ts // self.interval) * self.interval
        if not self.curr:
            self.curr = Candle(b_ts, price, price, price, price)
            return False
        if b_ts > self.curr.timestamp:
            self.candles.append(self.curr)
            if len(self.candles) > self.max_len:
                self.candles.pop(0)
            self.curr = Candle(b_ts, price, price, price, price)
            return True
        self.curr.high = max(self.curr.high, price)
        self.curr.low = min(self.curr.low, price)
        self.curr.close = price
        return False

# ---------------------------------------------------------
# 4. كود واجهة الويب HTML المدمجة داخلياً كـ String
# ---------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>XAU/USD S5 Execution Terminal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
    body { font-family: 'IBM Plex Sans Arabic', sans-serif; background: #08090D; color: #F5F5F7; }
    .card { background: rgba(18, 20, 29, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; }
    .mono { font-family: monospace; }
  </style>
</head>
<body class="min-h-screen p-3 md:p-5 flex flex-col gap-4">
  <header class="card px-5 py-3 flex flex-wrap items-center justify-between gap-3">
    <div class="flex items-center gap-3">
      <span id="dot" class="w-3 h-3 rounded-full bg-amber-500 animate-pulse"></span>
      <div>
        <h1 class="text-base md:text-lg font-bold">XAU/USD S5 Order-Flow Terminal</h1>
        <p class="text-xs text-zinc-400">Live Auto-Execution System</p>
      </div>
    </div>
    <div class="flex flex-wrap items-center gap-5 mono text-xs md:text-sm">
      <div><span class="text-zinc-500 text-[10px] block">السعر</span><span id="price" class="font-bold text-white">--- / ---</span></div>
      <div><span class="text-zinc-500 text-[10px] block">السبريد</span><span id="spread" class="font-semibold text-emerald-400">---</span></div>
      <div><span class="text-zinc-500 text-[10px] block">الرصيد</span><span id="equity" class="font-bold text-white">$5,000.00</span></div>
      <div><span class="text-zinc-500 text-[10px] block">الربح اليومي</span><span id="pnl" class="font-semibold text-emerald-400">0.00%</span></div>
    </div>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1">
    <div class="lg:col-span-8 card p-3 flex flex-col min-h-[420px]">
      <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-2">
          <span class="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20 mono">XAUUSD • S5</span>
          <span id="bias" class="px-2 py-0.5 rounded text-[11px] bg-zinc-800 text-zinc-300 mono">BIAS: ---</span>
          <span id="dxy" class="px-2 py-0.5 rounded text-[11px] bg-zinc-800 text-zinc-300 mono">DXY: ---</span>
        </div>
        <span id="state" class="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30 mono">CONNECTING</span>
      </div>
      <div id="chart" class="w-full flex-1 rounded-lg overflow-hidden min-h-[360px]"></div>
    </div>

    <div class="lg:col-span-4 flex flex-col gap-4">
      <div class="card p-4">
        <h2 class="text-xs font-semibold text-zinc-400 mb-2">الأمر المعلق / الصفقة الحالية</h2>
        <div id="active-box" class="text-xs text-zinc-500 py-6 text-center border border-dashed border-zinc-800 rounded-lg mono">
          لا توجد أوامر حالياً
        </div>
      </div>
      <div class="card p-4 flex-1 flex flex-col">
        <h2 class="text-xs font-semibold text-zinc-400 mb-2">سجل الصفقات المنفذة</h2>
        <div id="history" class="space-y-1.5 overflow-y-auto text-xs flex-1 max-h-[220px]">
          <p class="text-zinc-600 text-center py-4">في انتظار أول صفقة...</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {
      layout: { background: { color: '#0F1118' }, textColor: '#8E94A4' },
      grid: { vertLines: { color: '#171A24' }, horzLines: { color: '#171A24' } },
      timeScale: { timeVisible: true, secondsVisible: true }
    });
    const series = chart.addCandlestickSeries({
      upColor: '#10B981', downColor: '#EF4444', borderVisible: false, wickUpColor: '#10B981', wickDownColor: '#EF4444'
    });

    let eline = null, sline = null, tline = null;
    function updateLines(obj) {
      if (eline) { series.removePriceLine(eline); eline = null; }
      if (sline) { series.removePriceLine(sline); sline = null; }
      if (tline) { series.removePriceLine(tline); tline = null; }
      if (!obj) return;
      const p = obj.price || obj.entry;
      eline = series.createPriceLine({ price: p, color: '#3B82F6', lineWidth: 2, title: 'ENTRY' });
      sline = series.createPriceLine({ price: obj.sl, color: '#EF4444', lineWidth: 2, title: 'SL' });
      tline = series.createPriceLine({ price: obj.tp, color: '#10B981', lineWidth: 2, title: 'TP' });
    }

    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}`;
    let socket = null;

    function connect() {
      socket = new WebSocket(wsUrl);
      socket.onopen = () => { document.getElementById('dot').className = 'w-3 h-3 rounded-full bg-emerald-500'; };
      socket.onclose = () => {
        document.getElementById('dot').className = 'w-3 h-3 rounded-full bg-rose-500';
        setTimeout(connect, 2000);
      };
      socket.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.type === 'INIT' && d.candles) series.setData(d.candles);
        if (d.type === 'TICK') {
          if (d.s5) series.update(d.s5);
          document.getElementById('price').textContent = `${d.bid.toFixed(2)} / ${d.ask.toFixed(2)}`;
          document.getElementById('spread').textContent = `$${d.spread.toFixed(2)}`;
          document.getElementById('equity').textContent = `$${d.equity.toFixed(2)}`;
          document.getElementById('pnl').textContent = `${d.pnl_pct.toFixed(2)}%`;
          document.getElementById('state').textContent = d.state;
          document.getElementById('bias').textContent = `BIAS: ${d.bias}`;
          document.getElementById('dxy').textContent = `DXY: ${d.dxy}`;
          updateLines(d.pos || d.ord);

          const abox = document.getElementById('active-box');
          if (d.pos) {
            abox.innerHTML = `<div class="space-y-1"><span class="text-emerald-400 font-bold">${d.pos.side} @ ${d.pos.entry}</span><br>SL: ${d.pos.sl} | TP: ${d.pos.tp}</div>`;
          } else if (d.ord) {
            abox.innerHTML = `<div class="space-y-1"><span class="text-amber-400 font-bold">${d.ord.type} @ ${d.ord.price}</span><br>SL: ${d.ord.sl} | TP: ${d.ord.tp}</div>`;
          } else {
            abox.textContent = 'لا توجد أوامر حالياً';
          }

          if (d.history && d.history.length > 0) {
            document.getElementById('history').innerHTML = d.history.map(t => `
              <div class="p-2 rounded bg-zinc-900 border border-zinc-800 flex justify-between mono">
                <span class="${t.side==='BUY'?'text-emerald-400':'text-rose-400'} font-bold">${t.side}</span>
                <span>${t.pnl >= 0 ? '+' : ''}$${t.pnl} (${t.reason})</span>
              </div>
            `).join('');
          }
        }
      };
    }
    connect();
  </script>
</body>
</html>
"""

# ---------------------------------------------------------
# 5. السيرفر الموحد (WebSocket + HTTP في آن واحد)
# ---------------------------------------------------------
class SingleApp:
    def __init__(self):
        self.engine = StrategyEngine()
        self.clients: Set = set()
        self.gold_s5 = TimeBucket(5)
        self.gold_m1 = TimeBucket(60)
        self.gold_m5 = TimeBucket(300)
        self.dxy_m5 = TimeBucket(300)
        self.balance = 5000.0
        self.equity = 5000.0
        self.trades = []
        self._seed()

    def _seed(self):
        now = int(time.time())
        for i in range(35, 0, -1):
            ts = ((now - i * 300) // 300) * 300
            p = 2650.0 + (35 - i) * 0.3
            c = Candle(ts, p, p + 0.8, p - 0.7, p + 0.3)
            self.gold_m5.candles.append(c)
            self.dxy_m5.candles.append(Candle(ts, 104.5 - i * 0.01, 104.5, 104.4, 104.5 - i * 0.01))
        for i in range(20, 0, -1):
            ts = ((now - i * 60) // 60) * 60
            p = 2660.0 + (20 - i) * 0.1
            self.gold_m1.candles.append(Candle(ts, p, p + 0.4, p - 0.3, p + 0.2))

    async def register(self, ws, *args):
        self.clients.add(ws)
        try:
            init_data = json.dumps({
                "type": "INIT",
                "candles": [c.to_dict() for c in self.gold_s5.candles[-60:]]
            })
            await ws.send(init_data)
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)

    def process_broker(self, now: int, bid: float, ask: float):
        eng = self.engine
        # Fill Limit Orders
        if eng.state == BotState.ORDER_PENDING and eng.active_order:
            ord_ = eng.active_order
            if (ord_["type"] == "BUY_LIMIT" and ask <= ord_["price"]) or (ord_["type"] == "SELL_LIMIT" and bid >= ord_["price"]):
                side = "BUY" if ord_["type"] == "BUY_LIMIT" else "SELL"
                entry = ask if side == "BUY" else bid
                eng.active_position = {
                    "side": side, "entry": entry, "sl": ord_["sl"], "tp": ord_["tp"],
                    "risk_dist": ord_["risk_dist"], "be_moved": False
                }
                eng.active_order = None
                eng.state = BotState.IN_TRADE

        # Manage Open Trade
        if eng.state == BotState.IN_TRADE and eng.active_position:
            pos = eng.active_position
            pnl = (bid - pos["entry"]) if pos["side"] == "BUY" else (pos["entry"] - ask)
            self.equity = round(self.balance + pnl, 2)
            # Break-Even
            if not pos["be_moved"] and pnl >= 1.5 * pos["risk_dist"]:
                pos["sl"] = round(pos["entry"] + (0.15 if pos["side"] == "BUY" else -0.15), 2)
                pos["be_moved"] = True
            # SL / TP
            hit_sl = (pos["side"] == "BUY" and bid <= pos["sl"]) or (pos["side"] == "SELL" and ask >= pos["sl"])
            hit_tp = (pos["side"] == "BUY" and bid >= pos["tp"]) or (pos["side"] == "SELL" and ask <= pos["tp"])
            if hit_sl or hit_tp:
                close_p = pos["sl"] if hit_sl else pos["tp"]
                final_pnl = round(((close_p - pos["entry"]) if pos["side"] == "BUY" else (pos["entry"] - close_p)), 2)
                self.balance = round(self.balance + final_pnl, 2)
                self.equity = self.balance
                self.trades.append({"side": pos["side"], "pnl": final_pnl, "reason": "SL_HIT" if hit_sl else "TP_HIT"})
                eng.active_position = None
                eng.state = BotState.SCANNING_M5

    async def run(self):
        gold = 2662.50
        dxy = 104.10
        count = 0
        while True:
            now = int(time.time())
            count += 1
            if count % 80 in (65, 66, 67):
                gold -= 0.60
            elif count % 80 in (68, 69, 70):
                gold += 0.70
            else:
                gold += random.uniform(-0.15, 0.18)
            dxy += random.uniform(-0.01, 0.008)

            bid, ask = round(gold - 0.08, 2), round(gold + 0.08, 2)
            self.gold_s5.update(now, gold)
            self.gold_m1.update(now, gold)
            self.gold_m5.update(now, gold)
            self.dxy_m5.update(now, dxy)

            self.process_broker(now, bid, ask)
            self.engine.evaluate(now, bid, ask, self.gold_m5.candles, self.dxy_m5.candles, self.gold_m1.candles, self.gold_s5.candles)

            if self.clients:
                payload = json.dumps({
                    "type": "TICK",
                    "bid": bid, "ask": ask, "spread": 0.16,
                    "s5": self.gold_s5.curr.to_dict() if self.gold_s5.curr else None,
                    "equity": self.equity, "pnl_pct": ((self.equity - 5000.0) / 5000.0) * 100,
                    "state": self.engine.state.value, "bias": self.engine.current_bias or "NEUTRAL",
                    "dxy": self.engine.dxy_status,
                    "ord": self.engine.active_order, "pos": self.engine.active_position,
                    "history": self.trades[-5:]
                })
                await asyncio.gather(*[c.send(payload) for c in list(self.clients)], return_exceptions=True)

            await asyncio.sleep(0.3)

def serve_http(first, second):
    try:
        req = second if hasattr(first, 'respond') else first
        headers = second.headers if hasattr(second, 'headers') else second
        if headers.get("Upgrade", "").lower() == "websocket":
            return None
        body = DASHBOARD_HTML.encode('utf-8')
        if hasattr(first, 'respond'):
            return first.respond(HTTPStatus.OK, body)
        return (HTTPStatus.OK, [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))], body)
    except Exception:
        return None

async def main():
    port = int(os.environ.get("SERVER_PORT", 25644))
    app = SingleApp()
    print(f"[*] Server starting on port {port}...")
    async with websockets.serve(app.register, "0.0.0.0", port, process_request=serve_http):
        await app.run()

if __name__ == "__main__":
    asyncio.run(main())
