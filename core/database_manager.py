from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from typing import Generator, Optional
import logging

from .database import Base  # Your SQLAlchemy models and declarative base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the SQLAlchemy engine, scoped sessions (thread/request-safe), and table creation.
    Compatible with SQLite, PostgreSQL, MySQL, etc.
    """
    def __init__(self, db_url: str, echo: bool = False):
        # Extra kwargs for SQLite memory db or special cases
        kwargs = {}
        if db_url == "sqlite://":
            # In-memory DB, use StaticPool
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
        if db_url.startswith("sqlite://"):
            kwargs.setdefault("connect_args", {"check_same_thread": False})

        self.engine = create_engine(db_url, echo=echo, future=True, **kwargs)
        self.SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine, future=True)
        # Use scoped_session for thread/request safety (FastAPI, etc.)
        self.Session = scoped_session(self.SessionFactory)
        self.create_tables()

    def create_tables(self):
        """
        Create all tables defined in Base's metadata (models).
        """
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")

    def get_session(self) -> Generator[Session, None, None]:
        """
        Yields a thread-local/session-local SQLAlchemy session.
        Use with a `with` statement or as a generator in dependency injection.
        """
        session = self.Session()
        try:
            yield session
        finally:
            session.close()

    def remove_session(self):
        """
        Removes the current thread's session. Use in async web apps after requests.
        """
        self.Session.remove()

def init_db(db_url: str, echo: bool = False) -> DatabaseManager:
    """
    Centralized DB initialization.
    Returns a DatabaseManager instance ready for use.
    """
    return DatabaseManager(db_url, echo=echo)