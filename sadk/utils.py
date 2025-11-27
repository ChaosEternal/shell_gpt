from rich.console import Console
from rich.markdown import Markdown
import toml

class _Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(_Singleton, cls).__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Render(_Singleton):
    _no_md = None
    _theme = None

    def __init__(self, no_md=False):
        if self._no_md is None:
            self._no_md = no_md
        if self._theme is None:
            self._theme = "dracula"

    def output(self, text):

        if self._no_md:
            print(text)
        else:
            console = Console()
            console.print(Markdown(markup=text, code_theme=self._theme))

    def set_no_md(self, o):
        self._no_md = not o

    def set_theme(self, o):
        self._theme = o

def simple_handle_response_input_item_param(response):
    """
    Convert response: ResponseInputItemParam to text.

    """
    d = dict(response)
    if "content" in d:
        if isinstance(d["content"], list):
            return "\n".join(k["text"] for k in d["content"])
        return str(d["content"])

    if "id" in d:
        d.pop("id")

    n = d.get("name")
    t = d.get("type")
    return f"""
```toml
{t} = {n}
```

"""
