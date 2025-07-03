class UniswapClient:
    def __init__(self, config=None):
        self.config = config

    async def get_balance(self):
        return {"total_usd": 250}

    async def get_ticker(self, symbol):
        return {"last": 400.0}

    async def place_market_order(self, symbol, side, qty):
        return {"id": "order999", "status": "filled", "price": 400.0}

    async def place_limit_order(self, symbol, side, qty, price):
        return {"id": "order999", "status": "filled", "price": price}

    def supports_symbol(self, symbol):
        return True

    def get_balance_sync(self):
        return {"total_usd": 250}

    async def close(self):
        pass