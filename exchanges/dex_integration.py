class DexIntegration:
    def __init__(self, config=None):
        self.config = config

    async def get_balance(self):
        return {"total_usd": 500}

    async def get_ticker(self, symbol):
        return {"last": 300.0}

    async def place_market_order(self, symbol, side, qty):
        return {"id": "order789", "status": "filled", "price": 300.0}

    async def place_limit_order(self, symbol, side, qty, price):
        return {"id": "order789", "status": "filled", "price": price}

    def supports_symbol(self, symbol):
        return True

    def get_balance_sync(self):
        return {"total_usd": 500}

    async def close(self):
        pass