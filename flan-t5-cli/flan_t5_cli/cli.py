import argparse
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from .model import FlanT5Model
from .utils import (
    console,
    print_header,
    print_help,
    print_error,
    print_success,
    print_model_info,
)

class FLANT5CLI:
    """FLAN-T5 Command Line Interface."""
    
    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[str] = None):
        """Initialize the CLI.
        
        Args:
            model_name: Name of the model to load.
            device: Device to run the model on.
        """
        self.model = FlanT5Model(model_name, device)
        self.running = True
        print_header()
        print_model_info(self.model.model_name, self.model.device)
        
    def process_command(self, command: str) -> None:
        """Process a user command.
        
        Args:
            command: User input command.
        """
        if not command.strip():
            return
            
        if command.lower() == '/quit':
            self.running = False
            print_success("Goodbye!")
            
        elif command.lower() == '/help':
            print_help()
            
        elif command.lower() == '/model':
            self.switch_model()
            
        else:
            self.generate_response(command)
            
    def switch_model(self) -> None:
        """Switch to a different model."""
        new_model = console.input("Enter model name (e.g., 'google/flan-t5-large'): ")
        try:
            with console.status(f"[bold green]Loading {new_model}...") as status:
                self.model.switch_model(new_model)
            print_success(f"Switched to {new_model}")
            print_model_info(self.model.model_name, self.model.device)
        except Exception as e:
            print_error(f"Failed to switch model: {str(e)}")
        
  
    def generate_response(self, prompt: str) -> None:
        """Generate a response to the user's prompt."""
        try:
            with console.status("[bold green]Generating response...") as status:
                response = self.model.generate(prompt)
            console.print(f"\n[bold]Response:[/]")
            console.print(response)
        except Exception as e:
            print_error(f"Failed to generate response: {str(e)}")      
    
    def run(self) -> None:
        """Run the CLI."""
        while self.running:
            try:
                user_input = console.input("\nYou: ")
                self.process_command(user_input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /quit to exit[/]")
            except Exception as e:
                print_error(str(e))
    
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FLAN-T5 Command Line Interface")
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-base",
        help="Model to use (default: google/flan-t5-base)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    cli = FLANT5CLI(model_name=args.model, device=args.device)
    cli.run()

if __name__ == "__main__":
    main()