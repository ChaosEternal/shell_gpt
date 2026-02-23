import re
from typing import List

from .agent_hive import AgentHive
from agents import function_tool

tutorial_dir = AgentHive().config_dir / "tutorials"


def err_list_tutorials() -> str:
    """
    List all available tutorials.
    """
    tuts = [f"- {x.name.rstrip(".md")}" for x in tutorial_dir.glob("*.md")]
    message = [
        "The tutorial you specified does not exist.",
        "Available tutorials:"
    ] + tuts

    return "\n".join(message)

def read_tutorial_raw(tutorial_name: str) -> str:
    """
    Read the content of a tutorial.

    Args:
        tutorial_name: the name of the tutorial to read.

    Returns:
        The content of the tutorial.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", tutorial_name):
        return err_list_tutorials()

    tut = tutorial_dir / f"{tutorial_name}.md"
    if not tut.exists():
        return err_list_tutorials()

    with open(tut, "r") as f:
        return f.read()

read_tutorial = function_tool(func=read_tutorial_raw, name_override="read_tutorial")
