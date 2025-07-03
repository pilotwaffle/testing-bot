from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from typing import Generator, Optional

from .models import Base, Trade, Position  # Add more as needed

class DatabaseManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, echo=False, future=True)
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self.Session = scoped_session(self.session_factory)

    def init_db(self):
        """Creates all tables in the database."""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        """Yields a SQLAlchemy session and closes it after use."""
        session: Optional[Session] = None
        try:
            session = self.Session()
            yield session
        finally:
            if session:
                session.close()
                self.Session.remove()

def init_db(db_url: str) -> DatabaseManager:
    manager = DatabaseManager(db_url)
    manager.init_db()
    return manager

def get_database(db_url: str = "sqlite:///example.db") -> DatabaseManager:
    """Returns a DatabaseManager instance."""
    return DatabaseManager(db_url)