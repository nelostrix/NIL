"""Explicit local SQLite transactions shared by durable dataset/search APIs."""
from pathlib import Path
import sqlite3
import time


def connect(path):
    if str(path)==':memory:' or '://' in str(path) or str(path).startswith('file:'):
        raise ValueError('A durable SQLite file path is required, not an in-memory database or URI')
    path=Path(path)
    if path.is_symlink(): raise ValueError('Ledger path must not be a symlink')
    path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=5,isolation_level=None)
    try:
        deadline=time.monotonic()+5
        while True:
            try: db.execute('PRAGMA journal_mode=WAL'); break
            except sqlite3.OperationalError as error:
                if getattr(error,'sqlite_errorcode',0)&255 not in {5,6} or time.monotonic()>=deadline: raise
                time.sleep(.02)
        db.execute('PRAGMA synchronous=FULL')
        return db
    except BaseException:
        db.close(); raise
