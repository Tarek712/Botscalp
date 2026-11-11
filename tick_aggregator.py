from collections import deque
from typing import Dict, List, Optional
from strategy_engine import Candle


class TimeframeBucket:
    def __init__(self, interval_seconds: int, max_candles: int = 200):
        self.interval = interval_seconds
        self.candles: deque[Candle] = deque(maxlen=max_candles)
        self.current_candle: Optional[Candle] = None

    def update(self, timestamp_sec: int, price: float, volume: float = 1.0) -> bool:
        """
        Updates the current bucket with a new tick.
        Returns True if a candle just closed, False otherwise.
        """
        bucket_ts = (timestamp_sec // self.interval) * self.interval
        closed_new_candle = False

        if self.current_candle is None:
            self.current_candle = Candle(
                timestamp=bucket_ts,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume
            )
            return False

        if bucket_ts > self.current_candle.timestamp:
            # Close previous candle and push to history
            self.candles.append(self.current_candle)
            closed_new_candle = True
            # Open new candle
            self.current_candle = Candle(
                timestamp=bucket_ts,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume
            )
        else:
            # Update current open candle
            self.current_candle.high = max(self.current_candle.high, price)
            self.current_candle.low = min(self.current_candle.low, price)
            self.current_candle.close = price
            self.current_candle.volume += volume

        return closed_new_candle

    def get_closed_candles(self) -> List[Candle]:
        return list(self.candles)


class MultiTimeframeAggregator:
    """
    Zero-latency RAM aggregator that converts raw ticks for XAUUSD and DXY
    into synchronized S5, M1, and M5 candle streams.
    """
    def __init__(self):
        self.streams: Dict[str, Dict[str, TimeframeBucket]] = {
            "XAUUSD": {
                "S5": TimeframeBucket(5, max_candles=300),
                "M1": TimeframeBucket(60, max_candles=200),
                "M5": TimeframeBucket(300, max_candles=200),
            },
            "DXY": {
                "M1": TimeframeBucket(60, max_candles=200),
                "M5": TimeframeBucket(300, max_candles=200),
            }
        }

    def on_tick(self, symbol: str, timestamp_sec: int, bid: float, ask: float, volume: float = 1.0) -> Dict[str, bool]:
        mid_price = round((bid + ask) / 2.0, 3)
        closed_flags = {}

        if symbol not in self.streams:
            return closed_flags

        for tf_name, bucket in self.streams[symbol].items():
            closed_flags[tf_name] = bucket.update(timestamp_sec, mid_price, volume)

        return closed_flags

    def seed_historical_candles(self, symbol: str, timeframe: str, historical_candles: List[Candle]):
        if symbol in self.streams and timeframe in self.streams[symbol]:
            bucket = self.streams[symbol][timeframe]
            bucket.candles.clear()
            for c in historical_candles:
                bucket.candles.append(c)
