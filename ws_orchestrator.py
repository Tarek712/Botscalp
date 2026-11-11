import asyncio
import json
import os
import time
import random
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Set
import websockets

from strategy_engine import StrategyConfig, XAUUSDScalpingEngine, Candle
from tick_aggregator import MultiTimeframeAggregator
from paper_broker import PaperMatchingBroker

HOST_DOMAIN = "fi10.bot-hosting.cloud"
HOST_PORT = int(os.environ.get("SERVER_PORT", 25644))


class WebSocketOrchestrator:
    """
    Asynchronous WebSocket & HTTP Server for fi10.bot-hosting.cloud:25644
    - Serves dashboard.html directly when visiting http://fi10.bot-hosting.cloud:25644 in a browser
    - Broadcasts real-time telemetry, candles, bot state, and order lines over ws://fi10.bot-hosting.cloud:25644
    """
    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.engine = XAUUSDScalpingEngine(config)
        self.aggregator = MultiTimeframeAggregator()
        self.broker = PaperMatchingBroker(config, self.engine)
        self.connected_clients: Set = set()

        self._seed_initial_market_history()

    def _seed_initial_market_history(self):
        now = int(time.time())
        base_gold = 2650.00
        base_dxy = 104.50

        gold_m5_seed = []
        dxy_m5_seed = []
        for i in range(40, 0, -1):
            ts = ((now - i * 300) // 300) * 300
            wave = (40 - i) * 0.25 + (1.2 if i % 4 == 0 else -0.8)
            g_open = base_gold + wave
            g_close = g_open + (0.6 if i % 2 == 0 else -0.3)
            gold_m5_seed.append(Candle(ts, g_open, max(g_open, g_close) + 0.8, min(g_open, g_close) - 0.7, g_close))

            d_wave = -(40 - i) * 0.015
            d_open = base_dxy + d_wave
            d_close = d_open - 0.02
            dxy_m5_seed.append(Candle(ts, d_open, max(d_open, d_close) + 0.03, min(d_open, d_close) - 0.03, d_close))

        self.aggregator.seed_historical_candles("XAUUSD", "M5", gold_m5_seed)
        self.aggregator.seed_historical_candles("DXY", "M5", dxy_m5_seed)

        gold_m1_seed = []
        for i in range(25, 0, -1):
            ts = ((now - i * 60) // 60) * 60
            g_open = 2660.0 + (25 - i) * 0.1
            g_close = g_open + 0.35
            gold_m1_seed.append(Candle(ts, g_open, g_close + 0.4, g_open - 0.5, g_close))
        self.aggregator.seed_historical_candles("XAUUSD", "M1", gold_m1_seed)

    async def register_client(self, websocket, *args):
        self.connected_clients.add(websocket)
        try:
            init_payload = {
                "type": "INIT_SNAPSHOT",
                "candles_s5": [c.to_dict() for c in self.aggregator.streams["XAUUSD"]["S5"].get_closed_candles()[-60:]],
                "candles_m1": [c.to_dict() for c in self.aggregator.streams["XAUUSD"]["M1"].get_closed_candles()[-60:]],
                "account": {
                    "balance": self.broker.balance,
                    "equity": self.broker.equity,
                    "daily_pnl_pct": round(self.broker.daily_pnl_pct * 100, 2)
                }
            }
            await websocket.send(json.dumps(init_payload))
            async for _ in websocket:
                pass
        finally:
            self.connected_clients.discard(websocket)

    async def broadcast(self, payload: dict):
        if not self.connected_clients:
            return
        message = json.dumps(payload)
        await asyncio.gather(
            *[client.send(message) for client in list(self.connected_clients)],
            return_exceptions=True
        )

    async def process_incoming_tick(self, symbol: str, timestamp_sec: int, bid: float, ask: float):
        closed_flags = self.aggregator.on_tick(symbol, timestamp_sec, bid, ask)

        if symbol == "XAUUSD":
            broker_event = self.broker.process_tick(timestamp_sec, bid, ask)

            gmt_hour = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc).hour
            effective_hour = gmt_hour if self.cfg.enforce_london_ny_window else 14

            decision = self.engine.evaluate(
                current_time_gmt_hour=effective_hour,
                current_timestamp=timestamp_sec,
                bid=bid,
                ask=ask,
                daily_pnl_pct=self.broker.daily_pnl_pct,
                trades_today=self.broker.trades_today,
                won_today=self.broker.won_today,
                gold_m5=self.aggregator.streams["XAUUSD"]["M5"].get_closed_candles(),
                dxy_m5=self.aggregator.streams["DXY"]["M5"].get_closed_candles(),
                gold_m1=self.aggregator.streams["XAUUSD"]["M1"].get_closed_candles(),
                gold_s5=self.aggregator.streams["XAUUSD"]["S5"].get_closed_candles()
            )

            current_s5 = self.aggregator.streams["XAUUSD"]["S5"].current_candle
            telemetry = {
                "type": "LIVE_TELEMETRY",
                "timestamp": timestamp_sec,
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "spread": round(ask - bid, 2),
                "current_s5": current_s5.to_dict() if current_s5 else None,
                "s5_closed": closed_flags.get("S5", False),
                "bot_state": self.engine.state.value,
                "bias": self.engine.current_bias or "NEUTRAL",
                "dxy_status": self.engine.dxy_status,
                "decision": decision,
                "broker_event": broker_event,
                "active_order": self.engine.active_order,
                "active_position": self.engine.active_position,
                "account": {
                    "balance": self.broker.balance,
                    "equity": self.broker.equity,
                    "floating_pnl": self.broker.floating_pnl,
                    "daily_pnl_pct": round(self.broker.daily_pnl_pct * 100, 2),
                    "trades_today": self.broker.trades_today
                },
                "trade_history": self.broker.trade_history[-5:]
            }
            await self.broadcast(telemetry)

    async def run_simulated_market_stream(self):
        gold_mid = 2662.50
        dxy_mid = 104.10
        tick_counter = 0

        print(f"[*] Live Dashboard URL : http://{HOST_DOMAIN}:{HOST_PORT}")
        print(f"[*] WebSocket Endpoint : ws://{HOST_DOMAIN}:{HOST_PORT}")
        print(f"[*] Listening on port  : {HOST_PORT}")

        while True:
            now = int(time.time())
            tick_counter += 1

            if tick_counter % 100 in (85, 86, 87):
                gold_mid -= 0.65
            elif tick_counter % 100 in (88, 89, 90):
                gold_mid += 0.75
            else:
                gold_mid += random.uniform(-0.18, 0.20)

            dxy_mid += random.uniform(-0.012, 0.009)

            gold_spread = 0.16
            gold_bid = round(gold_mid - gold_spread / 2, 2)
            gold_ask = round(gold_mid + gold_spread / 2, 2)

            await self.process_incoming_tick("DXY", now, dxy_mid - 0.01, dxy_mid + 0.01)
            await self.process_incoming_tick("XAUUSD", now, gold_bid, gold_ask)

            await asyncio.sleep(0.25)


def serve_dashboard_http(first_arg, second_arg):
    """
    Compatible HTTP handler for both websockets <=13 (path, request_headers)
    and websockets >=14 (connection, request) so opening http://fi10.bot-hosting.cloud:25644
    directly in the browser renders dashboard.html!
    """
    try:
        # Check if it's a WebSocket upgrade request
        if hasattr(second_arg, "headers"):
            # websockets >= 14
            headers = second_arg.headers
            if headers.get("Upgrade", "").lower() == "websocket":
                return None
            html_bytes = Path("dashboard.html").read_bytes()
            return first_arg.respond(HTTPStatus.OK, html_bytes)
        else:
            # websockets <= 13
            headers = second_arg
            if headers.get("Upgrade", "").lower() == "websocket":
                return None
            html_bytes = Path("dashboard.html").read_bytes()
            return (
                HTTPStatus.OK,
                [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(html_bytes)))],
                html_bytes,
            )
    except Exception:
        return None


async def main():
    config = StrategyConfig(
        symbol="XAUUSD",
        fixed_lot=0.01,
        initial_balance=5000.0,
        enforce_london_ny_window=False
    )
    orchestrator = WebSocketOrchestrator(config)

    async with websockets.serve(
        orchestrator.register_client,
        "0.0.0.0",
        HOST_PORT,
        process_request=serve_dashboard_http
    ):
        await orchestrator.run_simulated_market_stream()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Bot stopped by user.")
