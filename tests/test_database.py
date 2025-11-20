import pytest
import os
from src.database import Database

@pytest.fixture
def db():
    db_path = "tests/test_data/test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    database = Database(db_path)
    yield database
    
    database.close()
    if os.path.exists(db_path):
        os.remove(db_path)

def test_session_creation(db):
    session_id = db.create_session()
    assert session_id is not None
    assert isinstance(session_id, int)

def test_message_storage(db):
    session_id = db.create_session()
    db.add_message(session_id, "user", "Hello")
    db.add_message(session_id, "assistant", "Hi there")
    
    history = db.get_history(session_id)
    assert len(history) == 2
    assert history[0]['role'] == "user"
    assert history[0]['content'] == "Hello"
    assert history[1]['role'] == "assistant"
    assert history[1]['content'] == "Hi there"

def test_caching(db):
    query = "What is RAG?"
    response = "Retrieval Augmented Generation"
    
    # Cache miss
    assert db.get_cached_response(query) is None
    
    # Cache hit
    db.cache_response(query, response)
    cached = db.get_cached_response(query)
    assert cached == response
    
    # Case insensitivity
    cached_upper = db.get_cached_response(query.upper())
    assert cached_upper == response
