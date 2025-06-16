from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown


console = Console()


def print_header() -> None:
    """Print the application header."""
    console.print(
        Panel.fit(
            "[bold blue]FLAN-T5 CLI[/]\n"
            "Type your message and press Enter to get a response.\n"
            "Commands: [bold]/model[/] - Change model, [bold]/quit[/] - Exit",
            title="Welcome"
        )
    ) 
    
def print_help() -> None:
    """Print help information."""
    help_text = """
        [bold]Available Commands:[/]
        [bold]/model[/] - Switch to a different FLAN-T5 model
        [bold]/help[/]  - Show this help message
        [bold]/quit[/]  - Exit the application
        
        [bold]Examples:[/]
        Summarize quantum computing in simple terms
        Translate 'Hello, how are you?' to French
        Explain the concept of machine learning
        """
    console.print(Panel(help_text, title="Help"))

def print_error(message: str) -> None:
    """Print an error message.
    
    Args:
        message: Error message to display.
    """
    console.print(f"[red]Error: {message}[/]")

def print_success(message: str) -> None:
    """Print a success message.
    
    Args:
        message: Success message to display.
    """
    console.print(f"[green]✓ {message}[/]")

def print_model_info(model_name: str, device: str) -> None:
    """Print information about the loaded model.
    
    Args:
        model_name: Name of the loaded model.
        device: Device the model is running on.
    """
    console.print(f"[bold]Model:[/] {model_name}")
    console.print(f"[bold]Device:[/] {device.upper()}\n")