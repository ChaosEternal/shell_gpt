import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Dict

from agents import (
    Agent,
    Runner,
    RunConfig,
    RunContextWrapper,
    function_tool,
)
from agents.run import CallModelData, ModelInputData

# 1. Define the Context
# This acts as our "Session Database" and the bridge between Tool -> Filter
@dataclass
class SessionContext:
    # The "Database" of memories
    data_store: Dict[str, str]
    
    # A temporary holding area for items fetched during this turn
    staged_items: List[str] = field(default_factory=list)

# 2. The Tool
# It finds the item but returns a generic status. 
# The actual content is moved to 'staged_items' in the context.
@function_tool
def fetch_memory_item(
    ctx: RunContextWrapper[SessionContext], 
    lookup_key: str
) -> str:
    """
    Fetches a specific item from the session database by its key.
    """
    # Access our "Session DB" from the context
    store = ctx.context.data_store
    value = store.get(lookup_key)
    
    if value:
        # Found it! Stage it for injection.
        print(f"\n[Tool] Found item for '{lookup_key}'. Staging for injection.")
        ctx.context.staged_items.append(value)
        return "Item found and staged."
    
    return f"No item found for key: {lookup_key}"

# 3. The Input Filter
# This runs RIGHT BEFORE the LLM is called. 
# It takes any 'staged_items' and injects them into the input list 'as is'.
def inject_staged_items_filter(data: CallModelData[SessionContext]) -> ModelInputData:
    context = data.context
    current_input_list = data.model_data.input
    
    # If we have items waiting to be injected...
    if context and context.staged_items:
        print(f"[Filter] Injecting {len(context.staged_items)} items into model input.")
        
        injected_messages = []
        for item_content in context.staged_items:
            # We inject the item "as is" (wrapped in a system message for proper formatting)
            injected_messages.append({
                "role": "system",
                "content": f"CONTEXTUAL MEMORY: {item_content}"
            })
            
        # Combine: Current History + Injected Items
        # We append them so they appear at the very end of the context window
        new_input_list = list(current_input_list) + injected_messages
        
        # CLEAR the stage so they don't persist permanently in the context object
        # (The filter adds them freshly each time they are needed)
        context.staged_items.clear()
        
        return ModelInputData(
            input=new_input_list,
            instructions=data.model_data.instructions
        )

    return data.model_data

# 4. Execution
async def main():
    # Populate our session database
    session_data = {
        "user_preference": "The user prefers answers in bullet points.",
        "project_deadline": "The project deadline is Friday at 5 PM.",
        "secret_codeword": "BlueFalcon",
    }
    
    # Initialize context
    ctx = SessionContext(data_store=session_data)

    agent = Agent(
        name="MemoryAgent",
        instructions="You are a helpful assistant. Use fetch_memory_item to look up facts.",
        tools=[fetch_memory_item],
    )

    # Configure the run to use our filter
    run_config = RunConfig(
        call_model_input_filter=inject_staged_items_filter
    )

    print("--- Turn 1: User asks a question requiring memory ---")
    # The agent will call the tool, which returns "Item found". 
    # Then the filter will inject the actual text "BlueFalcon" into the prompt.
    result = await Runner.run(
        agent, 
        input="What is the secret codeword?", 
        context=ctx,
        run_config=run_config
    )
    
    print(f"\nAgent Response: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
