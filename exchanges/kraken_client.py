class KrakenClient:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret

    async def get_balance(self):
        return {"total_usd": 2000}

    async def get_ticker(self, symbol):
        return {"last": 200.0}

    async def place_market_order(self, symbol, side, qty):
        return {"id": "order456", "status": "filled", "price": 200.0}

    async def place_limit_order(self, symbol, side, qty, price):
        return {"id": "order456", "status": "filled", "price": price}

    def supports_symbol(self, symbol):
        return True

    def get_balance_sync(self):
        return {"total_usd": 2000}

    async def close(self):
        pass