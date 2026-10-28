"""Alpaca SDK adapter; importing this module never constructs a client."""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (GetOrderByIdRequest, GetOrdersRequest,
    LimitOrderRequest, TakeProfitRequest, StopLossRequest)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from order_journal import OrderBlocked


class PaperTradingClient(TradingClient):
    """Pinned SDK 0.44.0: one HTTP attempt, finite connect/read timeouts.

    The SDK's default transport can retry POSTs. These private transport hooks
    are intentionally version-pinned and regression-tested; review on upgrades.
    """
    def __init__(self, api_key, secret_key):
        super().__init__(api_key, secret_key, paper=True)
        self._retry = 0

    def _one_request(self, method, url, opts, retry):
        return super()._one_request(method, url, dict(opts, timeout=(5, 10)), 0)


def lookup_nested(broker, cid, broker_id=None):
    if broker_id is None:
        broker_id = broker.get_order_by_client_id(cid).id
    return broker.get_order_by_id(broker_id, filter=GetOrderByIdRequest(nested=True))


def open_orders(broker):
    orders = broker.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.OPEN, nested=True, limit=500))
    if len(orders) >= 500:
        raise OrderBlocked("Open-order page may be truncated")
    return orders


def protective_request(cid, symbol, qty, stop, target):
    return LimitOrderRequest(client_order_id=cid, symbol=symbol, qty=qty,
        side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
        order_class=OrderClass.OCO, limit_price=target,
        take_profit=TakeProfitRequest(limit_price=target),
        stop_loss=StopLossRequest(stop_price=stop))
