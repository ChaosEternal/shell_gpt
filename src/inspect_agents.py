import sys
import os

try:
    import agents
    from agents import RunConfig, Agent
    
    print("RunConfig attributes:")
    print(dir(RunConfig))
    
    print("\nRunConfig annotations:")
    if hasattr(RunConfig, "__annotations__"):
        print(RunConfig.__annotations__)
        
    print("\nAgent attributes:")
    print(dir(Agent))

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
