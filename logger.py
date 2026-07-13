from rich.console import Console

#0B6623 Forest green
#00DF00 Grass green

class Logger:
    def __init__(self):
        self.console = Console()
    
    def __str__(self):
        return "This is a Logger instance."

    def info(self, msg: str):
        self.console.print(f"[#0B6623 on white]INFO:[/#0B6623 on white] [black on #00DF00]{msg}[/black on #00DF00]")

    def print_array(self, arr: list):
        self.console.print("\n######################ARRAY/TUPLE########################", style="#0B6623 on white")
        for i, entry in enumerate(arr):
            self.console.print(f"[#0B6623 on white]Entry {i}[/#0B6623 on white]: {entry}")
        self.console.print("######################ARRAY/TUPLE########################\n", style="#0B6623 on white")