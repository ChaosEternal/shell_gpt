from agents import (
    Agent,
    Runner,
    RunConfig,
    RunContextWrapper,    
    Session,
    SQLiteSession,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from sadk.memory import search_in_memory_raw, create_memory, MemoryContext, EmbedGear



memory = create_memory()
c = AsyncOpenAI(base_url="http://127.0.0.1:11435/v1", api_key="no")
embeder = EmbedGear(c, "nomic-embed-text:latest")

ctx = RunContextWrapper(context=MemoryContext(session=memory, embeder=embeder))

import asyncio
asyncio.run(search_in_memory_raw(ctx, ["a", "b", "c"]))


