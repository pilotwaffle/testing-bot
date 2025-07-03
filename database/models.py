from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    # Add more trade fields as needed

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    # Add more position fields as needed

# Define additional models here as needed, e.g. Order, PairLock, GlobalLock, etc.