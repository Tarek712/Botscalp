import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


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


@dataclass
class StrategyConfig:
    symbol: str = "XAUUSD"
    fixed_lot: float = 0.01
    initial_balance: float = 5000.0
    max_daily_dd_pct: float = 0.02          # 2% Daily Drawdown Limit
    max_trades_per_day: int = 3             # Max trades per day
    stop_on_first_win: bool = True          # Stop trading for the day after first TP hit
    max_spread_usd: float = 0.30            # Max allowed spread ($0.30 / 3 pips)
    atr_period: int = 14
    sl_atr_multiplier: float = 1.5
    rr_target: float = 3.0                  # Target 1:3 R:R
    breakeven_trigger_rr: float = 1.5       # Move SL to BE at 1:1.5 R:R
    m1_wait_timeout_sec: int = 900          # 15 min timeout waiting for M1 trigger
    s5_hunt_timeout_sec: int = 180          # 3 min timeout hunting S5 exhaustion
    pending_order_ttl_sec: int = 30         # 30 sec Limit Order Time-To-Live
    enforce_london_ny_window: bool = True   # 13:00 to 17:00 GMT filter


class XAUUSDScalpingEngine:
    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.state = BotState.IDLE
        self.current_bias: Optional[str] = None
        self.dxy_status: str = "NEUTRAL"

        # Timers
        self.m1_state_start_time: int = 0
        self.s5_state_start_time: int = 0
        self.pending_order_time: int = 0

        # Active Order / Position Refs
        self.active_order: Optional[Dict] = None
        self.active_position: Optional[Dict] = None

    @staticmethod
    def calculate_atr(candles: List[Candle], period: int = 14) -> float:
        if len(candles) < period + 1:
            return 1.20
        trs = []
        for i in range(1, len(candles)):
            c = candles[i]
            prev_c = candles[i - 1]
            tr = max(c.high - c.low, abs(c.high - prev_c.close), abs(c.low - prev_c.close))
            trs.append(tr)
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
        """Determines M5 Market Structure using 5-Bar Fractals and EMA20."""
        if len(m5_candles) < 25:
            return "NEUTRAL"

        swing_highs = []
        swing_lows = []

        for i in range(2, len(m5_candles) - 2):
            if (m5_candles[i].high > m5_candles[i-1].high and
                m5_candles[i].high > m5_candles[i-2].high and
                m5_candles[i].high > m5_candles[i+1].high and
                m5_candles[i].high > m5_candles[i+2].high):
                swing_highs.append(m5_candles[i].high)

            if (m5_candles[i].low < m5_candles[i-1].low and
                m5_candles[i].low < m5_candles[i-2].low and
                m5_candles[i].low < m5_candles[i+1].low and
                m5_candles[i].low < m5_candles[i+2].low):
                swing_lows.append(m5_candles[i].low)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "NEUTRAL"

        last_close = m5_candles[-1].close
        ema_20 = self.calculate_ema(m5_candles, 20)

        if (swing_lows[-1] > swing_lows[-2] and last_close > ema_20) or (last_close > swing_highs[-1]):
            if last_close > swing_lows[-1]:
                return "BULLISH"

        if (swing_highs[-1] < swing_highs[-2] and last_close < ema_20) or (last_close < swing_lows[-1]):
            if last_close < swing_highs[-1]:
                return "BEARISH"

        return "NEUTRAL"

    def check_dxy_filter(self, dxy_m5: List[Candle], gold_m5: List[Candle]) -> str:
        """Checks DXY inverse correlation and detects liquidity traps."""
        if len(dxy_m5) < 20 or len(gold_m5) < 20:
            return "INVALID"

        dxy_ema20 = self.calculate_ema(dxy_m5, 20)
        dxy_roc = (dxy_m5[-1].close - dxy_m5[-4].close) / max(dxy_m5[-4].close, 1e-5)
        gold_roc = (gold_m5[-1].close - gold_m5[-4].close) / max(gold_m5[-4].close, 1e-5)

        if (dxy_roc > 0.0005 and gold_roc > 0.0005) or (dxy_roc < -0.0005 and gold_roc < -0.0005):
            return "ANOMALY_TRAP"

        if dxy_m5[-1].close < dxy_ema20 and dxy_roc < 0:
            return "DXY_BEARISH"
        elif dxy_m5[-1].close > dxy_ema20 and dxy_roc > 0:
            return "DXY_BULLISH"

        return "NEUTRAL"

    def detect_m1_trigger(self, m1_candles: List[Candle], bias: str) -> bool:
        """Detects M1 Engulfing or Rejection Pinbar at POI."""
        if len(m1_candles) < 15:
            return False

        curr = m1_candles[-1]
        prev = m1_candles[-2]
        atr_m1 = self.calculate_atr(m1_candles, 14)

        if curr.total_range < (0.45 * atr_m1):
            return False

        if bias == "BULLISH":
            is_engulfing = curr.is_bullish and prev.is_bearish and (curr.close > prev.open) and (curr.open <= prev.close)
            is_pinbar = (curr.lower_wick >= 0.60 * curr.total_range) and (curr.upper_wick <= 0.25 * curr.total_range)
            return is_engulfing or is_pinbar

        elif bias == "BEARISH":
            is_engulfing = curr.is_bearish and prev.is_bullish and (curr.close < prev.open) and (curr.open >= prev.close)
            is_pinbar = (curr.upper_wick >= 0.60 * curr.total_range) and (curr.lower_wick <= 0.25 * curr.total_range)
            return is_engulfing or is_pinbar

        return False

    def detect_s5_exhaustion(self, s5_candles: List[Candle], bias: str) -> Optional[Dict]:
        """Detects S5 micro-sweep exhaustion wick and calculates Limit price."""
        if len(s5_candles) < 8:
            return None

        curr = s5_candles[-1]
        recent_lows = [c.low for c in s5_candles[-6:-1]]
        recent_highs = [c.high for c in s5_candles[-6:-1]]

        if bias == "BULLISH":
            swept_liquidity = curr.low <= min(recent_lows)
            has_exhaustion_wick = curr.lower_wick >= (1.8 * max(curr.body, 0.04))
            if swept_liquidity and has_exhaustion_wick and curr.is_bullish:
                limit_price = round(curr.low + (0.5 * curr.lower_wick), 2)
                return {"limit_price": limit_price, "wick_extreme": curr.low}

        elif bias == "BEARISH":
            swept_liquidity = curr.high >= max(recent_highs)
            has_exhaustion_wick = curr.upper_wick >= (1.8 * max(curr.body, 0.04))
            if swept_liquidity and has_exhaustion_wick and curr.is_bearish:
                limit_price = round(curr.high - (0.5 * curr.upper_wick), 2)
                return {"limit_price": limit_price, "wick_extreme": curr.high}

        return None

    def evaluate(
        self,
        current_time_gmt_hour: int,
        current_timestamp: int,
        bid: float,
        ask: float,
        daily_pnl_pct: float,
        trades_today: int,
        won_today: bool,
        gold_m5: List[Candle],
        dxy_m5: List[Candle],
        gold_m1: List[Candle],
        gold_s5: List[Candle]
    ) -> Dict:
        spread = ask - bid

        # 1. Daily Risk & Halt Checks
        if daily_pnl_pct <= -self.cfg.max_daily_dd_pct:
            self.state = BotState.HALTED
            return {"action": "HALT", "reason": "Daily Drawdown Limit Reached (-2%)"}

        if self.cfg.stop_on_first_win and won_today:
            self.state = BotState.HALTED
            return {"action": "HALT", "reason": "Daily Target Secured (1st Win)"}

        if trades_today >= self.cfg.max_trades_per_day:
            self.state = BotState.HALTED
            return {"action": "HALT", "reason": "Max Daily Trades Reached"}

        # 2. In-Trade Management
        if self.state == BotState.IN_TRADE and self.active_position:
            return {"action": "MONITOR_TRADE", "state": self.state.value}

        # 3. Pending Order TTL Check
        if self.state == BotState.ORDER_PENDING and self.active_order:
            if current_timestamp - self.pending_order_time > self.cfg.pending_order_ttl_sec:
                self.active_order = None
                self.state = BotState.SCANNING_M5
                return {"action": "CANCEL_LIMIT_ORDER", "reason": "TTL Expired (30s)"}
            return {"action": "WAIT_FILL", "order": self.active_order}

        # 4. Time & Spread Filters
        if self.cfg.enforce_london_ny_window and not (13 <= current_time_gmt_hour < 17):
            self.state = BotState.IDLE
            return {"action": "NONE", "reason": "Outside 13:00-17:00 GMT Window"}

        if spread > self.cfg.max_spread_usd:
            return {"action": "NONE", "reason": f"Spread Too High (${spread:.2f})"}

        # 5. State: SCANNING_M5
        if self.state in [BotState.IDLE, BotState.SCANNING_M5]:
            self.state = BotState.SCANNING_M5
            m5_bias = self.get_m5_market_structure(gold_m5)
            self.dxy_status = self.check_dxy_filter(dxy_m5, gold_m5)

            if self.dxy_status == "ANOMALY_TRAP":
                return {"action": "NONE", "reason": "Gold & DXY Moving Together (Liquidity Trap)"}

            if m5_bias == "BULLISH" and self.dxy_status == "DXY_BEARISH":
                self.current_bias = "BULLISH"
                self.state = BotState.WAITING_M1
                self.m1_state_start_time = current_timestamp
                return {"action": "STATE_CHANGE", "new_state": "WAITING_M1", "bias": "BULLISH"}

            elif m5_bias == "BEARISH" and self.dxy_status == "DXY_BULLISH":
                self.current_bias = "BEARISH"
                self.state = BotState.WAITING_M1
                self.m1_state_start_time = current_timestamp
                return {"action": "STATE_CHANGE", "new_state": "WAITING_M1", "bias": "BEARISH"}

        # 6. State: WAITING_M1
        if self.state == BotState.WAITING_M1:
            if current_timestamp - self.m1_state_start_time > self.cfg.m1_wait_timeout_sec:
                self.state = BotState.SCANNING_M5
                return {"action": "RESET_TO_M5", "reason": "M1 Wait Timeout (15m)"}

            if self.detect_m1_trigger(gold_m1, self.current_bias):
                self.state = BotState.HUNTING_S5
                self.s5_state_start_time = current_timestamp
                return {"action": "STATE_CHANGE", "new_state": "HUNTING_S5", "bias": self.current_bias}

        # 7. State: HUNTING_S5
        if self.state == BotState.HUNTING_S5:
            if current_timestamp - self.s5_state_start_time > self.cfg.s5_hunt_timeout_sec:
                self.state = BotState.SCANNING_M5
                return {"action": "RESET_TO_M5", "reason": "S5 Hunt Timeout (3m)"}

            s5_signal = self.detect_s5_exhaustion(gold_s5, self.current_bias)
            if s5_signal:
                atr_m5 = self.calculate_atr(gold_m5, self.cfg.atr_period)
                entry_price = s5_signal["limit_price"]
                wick_extreme = s5_signal["wick_extreme"]

                if self.current_bias == "BULLISH":
                    sl_distance = max(self.cfg.sl_atr_multiplier * atr_m5, abs(entry_price - wick_extreme) + 0.20)
                    sl_price = round(entry_price - sl_distance, 2)
                    tp_price = round(entry_price + (self.cfg.rr_target * sl_distance), 2)
                    order_type = "BUY_LIMIT"
                else:
                    sl_distance = max(self.cfg.sl_atr_multiplier * atr_m5, abs(wick_extreme - entry_price) + 0.20)
                    sl_price = round(entry_price + sl_distance, 2)
                    tp_price = round(entry_price - (self.cfg.rr_target * sl_distance), 2)
                    order_type = "SELL_LIMIT"

                self.active_order = {
                    "id": f"ORD-{current_timestamp}",
                    "type": order_type,
                    "symbol": self.cfg.symbol,
                    "lot": self.cfg.fixed_lot,
                    "price": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "risk_dist": round(sl_distance, 2),
                    "created_at": current_timestamp
                }
                self.pending_order_time = current_timestamp
                self.state = BotState.ORDER_PENDING
                return {"action": "PLACE_LIMIT_ORDER", "order": self.active_order}

        return {"action": "MONITORING", "state": self.state.value}
