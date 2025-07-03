import contextlib
import aiosqlite
from dataclasses import dataclass, asdict
from typing import List, Optional, AsyncGenerator
from datetime import datetime

DATABASE_PATH = "trade_bot.sqlite"


@dataclass
class Trade:
    id: Optional[int]
    symbol: str
    quantity: float
    price: float
    timestamp: datetime

    @staticmethod
    def from_row(row):
        return Trade(
            id=row[0],
            symbol=row[1],
            quantity=row[2],
            price=row[3],
            timestamp=datetime.fromisoformat(row[4])
        )


@dataclass
class Position:
    id: Optional[int]
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float

    @staticmethod
    def from_row(row):
        return Position(
            id=row[0],
            symbol=row[1],
            quantity=row[2],
            avg_entry_price=row[3],
            current_price=row[4],
            unrealized_pnl=row[5]
        )


@contextlib.asynccontextmanager
async def get_database() -> AsyncGenerator[aiosqlite.Connection, None]:
    db = await aiosqlite.connect(DATABASE_PATH)
    try:
        await db.execute("PRAGMA foreign_keys = ON;")
        yield db
    finally:
        await db.close()


async def initialize_database():
    async with get_database() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl REAL NOT NULL
            );
        """)
        await db.commit()


# CRUD for Trades
async def insert_trade(trade: Trade) -> int:
    async with get_database() as db:
        cursor = await db.execute(
            "INSERT INTO trades (symbol, quantity, price, timestamp) VALUES (?, ?, ?, ?)",
            (trade.symbol, trade.quantity, trade.price, trade.timestamp.isoformat())
        )
        await db.commit()
        return cursor.lastrowid

async def fetch_trades(symbol: Optional[str] = None) -> List[Trade]:
    async with get_database() as db:
        if symbol:
            cursor = await db.execute(
                "SELECT id, symbol, quantity, price, timestamp FROM trades WHERE symbol=? ORDER BY timestamp DESC", (symbol,)
            )
        else:
            cursor = await db.execute(
                "SELECT id, symbol, quantity, price, timestamp FROM trades ORDER BY timestamp DESC"
            )
        rows = await cursor.fetchall()
        return [Trade.from_row(row) for row in rows]

# CRUD for Positions
async def upsert_position(position: Position) -> int:
    async with get_database() as db:
        cursor = await db.execute(
            "SELECT id FROM positions WHERE symbol=?", (position.symbol,)
        )
        row = await cursor.fetchone()
        if row:
            await db.execute(
                """UPDATE positions SET quantity=?, avg_entry_price=?, current_price=?, unrealized_pnl=?
                   WHERE symbol=?""",
                (position.quantity, position.avg_entry_price, position.current_price, position.unrealized_pnl, position.symbol)
            )
            await db.commit()
            return row[0]
        else:
            cursor = await db.execute(
                """INSERT INTO positions (symbol, quantity, avg_entry_price, current_price, unrealized_pnl)
                   VALUES (?, ?, ?, ?, ?)""",
                (position.symbol, position.quantity, position.avg_entry_price, position.current_price, position.unrealized_pnl)
            )
            await db.commit()
            return cursor.lastrowid

async def fetch_positions() -> List[Position]:
    async with get_database() as db:
        cursor = await db.execute(
            "SELECT id, symbol, quantity, avg_entry_price, current_price, unrealized_pnl FROM positions"
        )
        rows = await cursor.fetchall()
        return [Position.from_row(row) for row in rows]

async def delete_trade(trade_id: int):
    async with get_database() as db:
        await db.execute("DELETE FROM trades WHERE id=?", (trade_id,))
        await db.commit()

async def delete_position(position_id: int):
    async with get_database() as db:
        await db.execute("DELETE FROM positions WHERE id=?", (position_id,))
        await db.commit()


# Utility: convert dataclass to dict for serialization
def trade_asdict(trade: Trade) -> dict:
    d = asdict(trade)
    d["timestamp"] = d["timestamp"].isoformat()
    return d

def position_asdict(position: Position) -> dict:
    return asdict(position)