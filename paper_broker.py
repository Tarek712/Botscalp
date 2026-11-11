from typing import Dict, List, Optional
from strategy_engine import StrategyConfig, BotState, XAUUSDScalpingEngine


class PaperMatchingBroker:
    """
    Realistic Paper-Trading Execution Engine for XAU/USD Limit Orders.
    - BUY_LIMIT fills ONLY when Ask <= Limit_Price
    - SELL_LIMIT fills ONLY when Bid >= Limit_Price
    - Break-Even and SL/TP are evaluated on real Bid/Ask sides
    """
    def __init__(self, config: StrategyConfig, engine: XAUUSDScalpingEngine):
        self.cfg = config
        self.engine = engine
        self.initial_balance = config.initial_balance
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.floating_pnl = 0.0

        self.trades_today = 0
        self.won_today = False
        self.trade_history: List[Dict] = []

    @property
    def daily_pnl_pct(self) -> float:
        return (self.equity - self.initial_balance) / self.initial_balance

    def process_tick(self, timestamp: int, bid: float, ask: float) -> Optional[Dict]:
        event = None

        # 1. Check Pending Limit Order Fill
        if self.engine.state == BotState.ORDER_PENDING and self.engine.active_order:
            order = self.engine.active_order
            if order["type"] == "BUY_LIMIT" and ask <= order["price"]:
                event = self._fill_order(order, fill_price=ask, side="BUY", timestamp=timestamp)
            elif order["type"] == "SELL_LIMIT" and bid >= order["price"]:
                event = self._fill_order(order, fill_price=bid, side="SELL", timestamp=timestamp)

        # 2. Manage Active Position (Floating PnL, Break-Even, SL/TP)
        if self.engine.state == BotState.IN_TRADE and self.engine.active_position:
            pos = self.engine.active_position
            lot = pos["lot"]
            # On XAUUSD, 1.00 lot = 100 oz -> $1 move = $100 -> 0.01 lot = $1 per $1 move
            contract_multiplier = 100.0 * lot

            if pos["side"] == "BUY":
                price_diff = bid - pos["entry"]
                self.floating_pnl = round(price_diff * contract_multiplier, 2)
                self.equity = round(self.balance + self.floating_pnl, 2)

                # Check SL
                if bid <= pos["sl"]:
                    return self._close_position(pos, close_price=pos["sl"], reason="SL_HIT", timestamp=timestamp)
                # Check TP
                if bid >= pos["tp"]:
                    return self._close_position(pos, close_price=pos["tp"], reason="TP_HIT", timestamp=timestamp)
                # Check Break-Even Trigger (1:1.5 R:R)
                if not pos["be_moved"] and price_diff >= (self.cfg.breakeven_trigger_rr * pos["risk_dist"]):
                    pos["sl"] = round(pos["entry"] + 0.15, 2)
                    pos["be_moved"] = True
                    return {"event": "BREAKEVEN_ACTIVATED", "new_sl": pos["sl"], "position": pos}

            elif pos["side"] == "SELL":
                price_diff = pos["entry"] - ask
                self.floating_pnl = round(price_diff * contract_multiplier, 2)
                self.equity = round(self.balance + self.floating_pnl, 2)

                # Check SL
                if ask >= pos["sl"]:
                    return self._close_position(pos, close_price=pos["sl"], reason="SL_HIT", timestamp=timestamp)
                # Check TP
                if ask <= pos["tp"]:
                    return self._close_position(pos, close_price=pos["tp"], reason="TP_HIT", timestamp=timestamp)
                # Check Break-Even Trigger (1:1.5 R:R)
                if not pos["be_moved"] and price_diff >= (self.cfg.breakeven_trigger_rr * pos["risk_dist"]):
                    pos["sl"] = round(pos["entry"] - 0.15, 2)
                    pos["be_moved"] = True
                    return {"event": "BREAKEVEN_ACTIVATED", "new_sl": pos["sl"], "position": pos}

        return event

    def _fill_order(self, order: Dict, fill_price: float, side: str, timestamp: int) -> Dict:
        self.engine.active_order = None
        self.engine.state = BotState.IN_TRADE
        self.engine.active_position = {
            "id": order["id"],
            "side": side,
            "lot": order["lot"],
            "entry": round(fill_price, 2),
            "sl": order["sl"],
            "tp": order["tp"],
            "risk_dist": order["risk_dist"],
            "be_moved": False,
            "opened_at": timestamp
        }
        return {"event": "ORDER_FILLED", "position": self.engine.active_position}

    def _close_position(self, pos: Dict, close_price: float, reason: str, timestamp: int) -> Dict:
        contract_multiplier = 100.0 * pos["lot"]
        if pos["side"] == "BUY":
            pnl = round((close_price - pos["entry"]) * contract_multiplier, 2)
        else:
            pnl = round((pos["entry"] - close_price) * contract_multiplier, 2)

        self.balance = round(self.balance + pnl, 2)
        self.floating_pnl = 0.0
        self.equity = self.balance
        self.trades_today += 1

        if pnl > 0 and reason == "TP_HIT":
            self.won_today = True

        closed_record = {
            **pos,
            "close_price": round(close_price, 2),
            "pnl": pnl,
            "reason": reason,
            "closed_at": timestamp
        }
        self.trade_history.append(closed_record)
        self.engine.active_position = None
        self.engine.state = BotState.SCANNING_M5

        return {"event": "POSITION_CLOSED", "trade": closed_record}
