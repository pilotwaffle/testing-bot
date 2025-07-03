class CoinbaseClient:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret

    async def get_balance(self):
        # Dummy implementation
        return {"total_usd": 1000}

    async def get_ticker(self, symbol):
        # Dummy implementation
        return {"last": 100.0}

    async def place_market_order(self, symbol, side, qty):
        # Dummy implementation
        return {"id": "order123", "status": "filled", "price": 100.0}

    async def place_limit_order(self, symbol, side, qty, price):
        # Dummy implementation
        return {"id": "order123", "status": "filled", "price": price}

    def supports_symbol(self, symbol):
        return True

    def get_balance_sync(self):
        return {"total_usd": 1000}

    async def close(self):
        pass