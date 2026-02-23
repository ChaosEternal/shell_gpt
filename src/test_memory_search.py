import asyncio
import os
import sys
import uuid
import json
import sqlite3
import numpy as np
from typing import List, Any
import logging

# Ensure we can import from src
# Assuming this script is run from src/ or similar
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    # Also link to agents if needed, assuming user env has it PYTHONPATH configured.

# Try to import necessary classes
try:
    from sadk.memory import Memory, EmbedGear
except ImportError:
    # If sadk is not in path, try adding specific path
    # This block handles if user runs from root
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shell_adk/src"))
    from sadk.memory import Memory, EmbedGear

# 1. Mock EmbedGear
class MockEmbedGear:
    def __init__(self):
        self.embedding_model = "mock-model"

    async def get_embedding(self, text: str) -> str:
        # Generate deterministic embedding based on text
        # Return hex string representing 768 bits (96 bytes -> 192 hex chars)
        import hashlib
        # Use simple hash to get bytes
        h = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
        # Repeat to fill 96 bytes (768 bits)
        full_bytes = h * 3 
        return full_bytes.hex()

async def main():
    db_path = "/tmp/test_sadk_memory.db"
    
    # Clean up previous run
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"Initializing Memory at {db_path}...")
    # Initialize Memory
    try:
        mem = Memory("test_session_scope", db_path)
        mem.init_database()
    except Exception as e:
        print(f"Failed to initialize memory: {e}")
        return

    # 2. Insert Test Data
    print("Inserting test data...")
    conn = mem._get_connection()
    cursor = conn.cursor()

    # Data to insert
    test_session_id = str(uuid.uuid4())
    print(f"Test Session ID: {test_session_id}")

    # Insert Session
    # Attempt to insert into sessions table.
    # We rely on Memory inherited properties for table names.
    # Note: SQLiteSession defines self.sessions_table usually as "sessions" or similar.
    # We'll use the property from the instance.
    try:
        cursor.execute(
            f"INSERT INTO {mem.sessions_table} (session_id, created_at, rank) VALUES (?, datetime('now'), ?)", 
            (test_session_id, 1)
        )
    except sqlite3.OperationalError as e:
        print(f"Error inserting session: {e}")
        # Debug schema
        res = cursor.execute(f"PRAGMA table_info({mem.sessions_table})")
        columns = [row[1] for row in res.fetchall()]
        print(f"Session Table Columns: {columns}")
        raise e

    # Insert Messages
    messages = [
        {"role": "user", "content": "How do I reverse a list in python?"},
        {"role": "assistant", "content": "You can use list.reverse() or slice [::-1]"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "system", "content": "System initialization complete."}
    ]

    for i, msg in enumerate(messages):
        msg_json = json.dumps(msg)
        try:
             # We assume messages_table has: id, session_id, message_data, created_at, embedding (from patch)
             cursor.execute(
                f"INSERT INTO {mem.messages_table} (session_id, message_data, created_at) VALUES (?, ?, datetime('now', '+{i} seconds'))",
                (test_session_id, msg_json)
            )
        except sqlite3.OperationalError as e:
             print(f"Error inserting message: {e}")
             res = cursor.execute(f"PRAGMA table_info({mem.messages_table})")
             columns = [row[1] for row in res.fetchall()]
             print(f"Messages Table Columns: {columns}")
             raise e

    conn.commit()
    print("Data inserted.")
    
    # 3. Run Embeddings
    print("Running embed_memories...")
    mock_embed = MockEmbedGear()
    await mem.embed_memories(mock_embed)
    print("Embeddings generated.")

    # 4. Search
    print("Searching for 'python reverse'...")
    search_queries = [
        "python list reverse",
        "how to reverse array in python", 
        "python reverse list syntax"
    ]
    
    # Generate query embeddings using the SAME mock embedder so we get matches.
    # Since our mock is deterministic based on text, we need queries that 'match' the content?
    # Actually, our mock hash is just a hash. 'python list reverse' hash will be completely different
    # from 'How do I reverse a list in python?'.
    # So valid semantic search won't work with this mock unless we fake the similarity.
    # BUT, the `search_similar` code uses `hamming_dist`.
    # It calculates bitwise difference.
    # If we want to find a match, we should search for something close to the inserted content hash.
    # For a UNIT TEST: we can use the exact same text to ensure distance is 0 (perfect match).
    
    test_search_text = "How do I reverse a list in python?"
    print(f"Using exact match query: '{test_search_text}'")
    
    exact_query_embedding = await mock_embed.get_embedding(test_search_text)
    
    # We pass 3 embeddings. We can pass the same one 3 times for strong signal,
    # or variations.
    query_embeddings = [exact_query_embedding] * 3

    results = await mem.search_similar(query_embeddings, topk=5, no_payload=False)
    
    print(f"Found {len(results)} results:")
    for res in results:
        # res tuple: (id, session_id, message_data, distance, rank)
        # Note: the order depends on SQL select
        # SELECT m.id, m.session_id, message_data, distance, s.rank
        msg_id, sess_id, msg_data_str, dist, rank = res
        msg_data = json.loads(msg_data_str)
        print(f" - [Dist: {dist:.4f}] {msg_data.get('content')[:50]}...")

    # verify specific hit
    found_target = any(test_search_text in json.loads(r[2])['content'] for r in results)
    
    if found_target:
        print("SUCCESS: Found relevant memory.")
    else:
        print("FAILURE: Did not find relevant memory.")

if __name__ == "__main__":
    asyncio.run(main())
