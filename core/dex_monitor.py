import logging
import time
from typing import List, Dict, Any, Optional
import requests

class DexEvent:
    """
    Represents a DEX (Decentralized Exchange) event, e.g. a swap, liquidity change, or new token listing.
    """
    def __init__(
        self, 
        event_type: str,
        tx_hash: str,
        block_number: int,
        timestamp: float,
        token_in: str,
        token_out: str,
        amount_in: float,
        amount_out: float,
        sender: str,
        recipient: str,
        raw: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type
        self.tx_hash = tx_hash
        self.block_number = block_number
        self.timestamp = timestamp
        self.token_in = token_in
        self.token_out = token_out
        self.amount_in = amount_in
        self.amount_out = amount_out
        self.sender = sender
        self.recipient = recipient
        self.raw = raw or {}

    def __repr__(self):
        return (
            f"DexEvent({self.event_type}, {self.tx_hash}, {self.block_number}, "
            f"{self.token_in}->{self.token_out} {self.amount_in}->{self.amount_out})"
        )

class DexMonitor:
    """
    Monitors decentralized exchanges (DEXes) for swaps, liquidity changes, and other key events.
    Can be extended for Uniswap, Pancakeswap, Curve, etc.
    """
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        dex_graph_api: Optional[str] = None,
        polling_interval: float = 10.0,
        tokens_of_interest: Optional[List[str]] = None
    ):
        """
        Args:
            rpc_url: Optional web3 RPC URL for on-chain fetching.
            dex_graph_api: Optional subgraph API endpoint for DEX event monitoring.
            polling_interval: Time in seconds between polling DEX events.
            tokens_of_interest: List of tokens to monitor, or None for all.
        """
        self.logger = logging.getLogger("DexMonitor")
        self.rpc_url = rpc_url
        self.dex_graph_api = dex_graph_api
        self.polling_interval = polling_interval
        self.tokens_of_interest = set(tokens_of_interest) if tokens_of_interest else None
        self.last_block = 0
        self.last_checked_timestamp = 0
        self.running = False
        self.event_log: List[DexEvent] = []

    def start(self):
        """
        Start the DEX monitor polling loop. This is blocking.
        """
        self.running = True
        self.logger.info("Starting DexMonitor polling loop.")
        while self.running:
            try:
                self.poll()
            except Exception as e:
                self.logger.error(f"Error during poll: {e}", exc_info=True)
            time.sleep(self.polling_interval)

    def stop(self):
        """Stop the polling loop."""
        self.running = False
        self.logger.info("Stopped DexMonitor.")

    def poll(self):
        """
        Poll for new DEX events using the configured method (subgraph API or on-chain).
        """
        if self.dex_graph_api:
            events = self._get_events_from_graph()
            if events:
                for event in events:
                    self.process_event(event)
        # You can add on-chain polling logic here if needed (e.g., via web3.py)

    def _get_events_from_graph(self) -> List[DexEvent]:
        """
        Fetches DEX events using a GraphQL subgraph endpoint.
        Returns:
            List of DexEvent objects.
        """
        # Example: Uniswap v2/v3 subgraph query for Swaps
        # This is a template; adapt the query for your DEX and subgraph
        query = """
        {
            swaps(orderBy: timestamp, orderDirection: desc, first: 10
                where: {timestamp_gt: %d}
            ) {
                transaction {
                    id
                    timestamp
                }
                pair {
                    token0 { symbol }
                    token1 { symbol }
                }
                amount0In
                amount1In
                amount0Out
                amount1Out
                sender
                to
            }
        }
        """ % self.last_checked_timestamp
        try:
            response = requests.post(
                self.dex_graph_api,
                json={"query": query}
            )
            response.raise_for_status()
            data = response.json()
            swaps = data.get("data", {}).get("swaps", [])
            events = []
            for swap in swaps:
                tx_id = swap["transaction"]["id"]
                timestamp = int(swap["transaction"]["timestamp"])
                token0 = swap["pair"]["token0"]["symbol"]
                token1 = swap["pair"]["token1"]["symbol"]
                # Determine direction
                if float(swap["amount0In"]) > 0:
                    token_in, token_out = token0, token1
                    amount_in = float(swap["amount0In"])
                    amount_out = float(swap["amount1Out"])
                else:
                    token_in, token_out = token1, token0
                    amount_in = float(swap["amount1In"])
                    amount_out = float(swap["amount0Out"])
                # Filter for tokens of interest
                if self.tokens_of_interest and not (
                    token_in in self.tokens_of_interest or token_out in self.tokens_of_interest
                ):
                    continue
                event = DexEvent(
                    event_type="swap",
                    tx_hash=tx_id,
                    block_number=0,  # block_number not available from subgraph in this template
                    timestamp=timestamp,
                    token_in=token_in,
                    token_out=token_out,
                    amount_in=amount_in,
                    amount_out=amount_out,
                    sender=swap.get("sender"),
                    recipient=swap.get("to"),
                    raw=swap
                )
                events.append(event)
                # Update last timestamp
                if timestamp > self.last_checked_timestamp:
                    self.last_checked_timestamp = timestamp
            self.logger.info(f"Fetched {len(events)} swap events from subgraph.")
            return events
        except Exception as e:
            self.logger.error(f"Failed to fetch DEX events from subgraph: {e}")
            return []

    def process_event(self, event: DexEvent):
        """
        Handle a new DEX event: log it, update state, trigger alerts, etc.
        """
        self.logger.info(f"New DEX event: {event}")
        self.event_log.append(event)
        # You may extend this to trigger notifications, metrics, or custom handlers

    def get_recent_events(self, count: int = 10) -> List[DexEvent]:
        """
        Return the N most recent DEX events.
        """
        return self.event_log[-count:]

    def find_large_swaps(self, min_amount: float) -> List[DexEvent]:
        """
        Return all swap events with amount_in above min_amount.
        """
        return [e for e in self.event_log if e.event_type == "swap" and e.amount_in >= min_amount]

    def monitor_new_listings(self):
        """
        Optionally, add logic to monitor for new token listings (requires DEX-specific support).
        """
        # Placeholder for extension.
        pass

    # You can add more advanced analytics, e.g. aggregate volume, on-chain health checks, etc.
