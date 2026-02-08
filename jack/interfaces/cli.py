"""
Jack CLI - Command Line Interface

Interactive interface for controlling Jack.
Supports both single commands and interactive mode.

Usage:
    jack run "list all python files"
    jack train --phase 0 --samples 1000
    jack interactive
"""

import sys
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

# Handle imports gracefully
try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    typer = None

# Import Jack components
from ..core import JackBrain, JackConfig, Executor, Verifier, Memory
from ..adapters import get_adapter
from ..training import Phase0Trainer, DigitalSandbox


class JackCLI:
    """
    Command-line interface for Jack.

    Provides:
    - Single command execution
    - Interactive mode
    - Training management
    - System inspection
    """

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize Jack CLI"""
        self.adapter = get_adapter()

        # Set up config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = self.adapter.get_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.memory = Memory(db_path=str(self.config_dir / "memory.db"))
        self.executor = Executor()
        self.verifier = Verifier()

        # Brain (transformer that predicts actions)
        self.brain = JackBrain(JackConfig())

        # Console for output
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None

    def print(self, message: str, style: str = "") -> None:
        """Print message to console"""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)

    def print_panel(self, content: str, title: str = "") -> None:
        """Print a panel"""
        if self.console:
            self.console.print(Panel(content, title=title))
        else:
            print(f"\n=== {title} ===")
            print(content)
            print("=" * (len(title) + 8))

    def print_error(self, message: str) -> None:
        """Print error message"""
        self.print(f"[bold red]Error:[/bold red] {message}" if HAS_RICH else f"Error: {message}")

    def print_success(self, message: str) -> None:
        """Print success message"""
        self.print(f"[bold green]Success:[/bold green] {message}" if HAS_RICH else f"Success: {message}")

    def run_task(self, task: str, verbose: bool = False) -> bool:
        """
        Execute a task using Jack.

        Currently runs as direct shell command. Future: use trained brain for planning.

        Args:
            task: Natural language task description or shell command
            verbose: Show detailed output

        Returns:
            True if successful
        """
        self.print_panel(task, "Task")

        try:
            # For now, treat simple commands directly
            # Future: use transformer brain for planning
            if verbose:
                self.print("\n[dim]Analyzing task...[/dim]" if HAS_RICH else "\nAnalyzing task...")

            # Simple task patterns
            task_lower = task.lower()

            if "list" in task_lower and "file" in task_lower:
                cmd = "dir" if os.name == "nt" else "ls -la"
                if "python" in task_lower:
                    cmd = "dir *.py /s /b" if os.name == "nt" else "find . -name '*.py'"
            elif "create" in task_lower and "script" in task_lower:
                self.print("Creating a script requires code generation (use LLM integration)")
                return False
            else:
                # Treat as direct command
                cmd = task

            if verbose:
                self.print(f"\n[cyan]Command:[/cyan] {cmd}" if HAS_RICH else f"\nCommand: {cmd}")

            # Verify and execute
            is_safe, reason = self.verifier.check_shell(cmd)
            if not is_safe:
                self.print_error(f"Command blocked: {reason}")
                return False

            result = self.executor.shell_run(cmd)
            if result.success:
                if result.output:
                    self.print(result.output[:2000])
                self.print_success("Task completed")
                return True
            else:
                self.print_error(result.error or "Command failed")
                return False

        except Exception as e:
            self.print_error(str(e))
            return False

    def run_shell(self, command: str) -> None:
        """Execute a shell command directly"""
        # Verify first
        is_safe, reason = self.verifier.check_shell(command)
        if not is_safe:
            self.print_error(f"Command blocked: {reason}")
            return

        result = self.executor.shell_run(command)
        if result.success:
            if result.output:
                self.print(result.output)
        else:
            self.print_error(result.error or "Command failed")

    def train(self, phase: int = 0, samples: int = 500, epochs: int = 5) -> None:
        """
        Run training.

        Args:
            phase: Training phase (0 = digital physics)
            samples: Number of samples to collect
            epochs: Number of training epochs
        """
        if phase != 0:
            self.print_error("Only Phase 0 training is currently implemented")
            return

        self.print_panel(f"Phase {phase}: Digital Physics Training", "Training")

        sandbox = DigitalSandbox(base_dir=str(self.config_dir / "sandbox"))
        trainer = Phase0Trainer(
            sandbox=sandbox,
            buffer_path=str(self.config_dir / "transitions.json"),
            world_model=self.brain,  # Pass transformer brain
        )

        try:
            if HAS_RICH:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Collecting experiences...", total=None)
                    stats = trainer.collect_phase(samples)
                    progress.update(task, description="Training complete!")
            else:
                self.print("Collecting experiences...")
                stats = trainer.collect_phase(samples)
                self.print("Training complete!")

            # Show stats
            self.print("\n[bold]Training Statistics:[/bold]" if HAS_RICH else "\nTraining Statistics:")
            for key, value in stats.items():
                self.print(f"  {key}: {value}")

            final_stats = trainer.get_stats()
            self.print(f"\n  Total transitions: {final_stats['total_transitions']}")
            self.print(f"  Success rate: {final_stats['successful_transitions'] / max(1, final_stats['total_transitions']):.1%}")

        finally:
            trainer.cleanup()

    def show_status(self) -> None:
        """Show Jack's current status"""
        if HAS_RICH:
            table = Table(title="Jack Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")

            # OS Info
            system_info = self.adapter.get_system_summary()
            table.add_row("OS", system_info['os'], f"{system_info['hostname']}")
            table.add_row("User", system_info['username'], "")
            table.add_row("CPU", "Active", f"{system_info['cpu_cores']} cores")
            table.add_row(
                "Memory",
                f"{system_info['memory_available_gb']:.1f} GB free",
                f"of {system_info['memory_total_gb']:.1f} GB"
            )

            # Jack components
            table.add_row("Brain", "Ready", "Transformer (6 layers, 8 heads)")
            table.add_row("Memory", "Active", str(self.config_dir / "memory.db"))
            table.add_row("Verifier", "Active", f"{len(self.verifier.blocked_patterns)} safety rules")

            self.console.print(table)
        else:
            self.print("\n=== Jack Status ===")
            system_info = self.adapter.get_system_summary()
            self.print(f"OS: {system_info['os']}")
            self.print(f"Host: {system_info['hostname']}")
            self.print(f"User: {system_info['username']}")
            self.print(f"CPU: {system_info['cpu_cores']} cores")
            self.print(f"Memory: {system_info['memory_available_gb']:.1f} / {system_info['memory_total_gb']:.1f} GB")
            self.print(f"\nConfig: {self.config_dir}")

    def interactive(self) -> None:
        """Run interactive mode"""
        self.print_panel(
            "Type commands or tasks. Use 'help' for options, 'exit' to quit.",
            "Jack Interactive Mode"
        )

        while True:
            try:
                if HAS_RICH:
                    user_input = Prompt.ask("\n[bold cyan]jack>[/bold cyan]")
                else:
                    user_input = input("\njack> ").strip()

                if not user_input:
                    continue

                # Handle special commands
                cmd_lower = user_input.lower().strip()

                if cmd_lower in ('exit', 'quit', 'q'):
                    self.print("Goodbye!")
                    break

                elif cmd_lower == 'help':
                    self._show_help()

                elif cmd_lower == 'status':
                    self.show_status()

                elif cmd_lower.startswith('!'):
                    # Direct shell command
                    self.run_shell(user_input[1:].strip())

                elif cmd_lower.startswith('train'):
                    # Training command
                    parts = cmd_lower.split()
                    samples = 500
                    for i, p in enumerate(parts):
                        if p == '--samples' and i + 1 < len(parts):
                            samples = int(parts[i + 1])
                    self.train(phase=0, samples=samples)

                elif cmd_lower == 'memory':
                    self._show_memory()

                else:
                    # Treat as task
                    self.run_task(user_input, verbose=True)

            except KeyboardInterrupt:
                self.print("\n[dim]Use 'exit' to quit[/dim]" if HAS_RICH else "\nUse 'exit' to quit")
            except EOFError:
                break

    def _show_help(self) -> None:
        """Show help information"""
        help_text = """
Commands:
  help          - Show this help
  status        - Show Jack's status
  exit          - Exit interactive mode
  !<command>    - Run shell command directly (e.g., !dir)
  train         - Run Phase 0 training
  memory        - Show memory contents

Or just type a task in natural language:
  "list all python files"
  "create a hello world script"
  "what processes are using the most memory?"
"""
        self.print(help_text)

    def _show_memory(self) -> None:
        """Show memory contents"""
        short_term = self.memory.get_short_term()
        tools = self.memory.list_tools()

        self.print("\n[bold]Short-term Memory:[/bold]" if HAS_RICH else "\nShort-term Memory:")
        if short_term:
            for item in short_term[-5:]:
                self.print(f"  - {item.content[:100]}")
        else:
            self.print("  (empty)")

        self.print("\n[bold]Saved Tools:[/bold]" if HAS_RICH else "\nSaved Tools:")
        if tools:
            for tool in tools[:10]:
                self.print(f"  - {tool['name']}: {tool['description'][:50]}")
        else:
            self.print("  (none)")


def run_cli():
    """Entry point for CLI"""
    if typer is None:
        print("Error: typer and rich are required for CLI. Install with:")
        print("  pip install typer rich")
        sys.exit(1)

    app = typer.Typer(
        name="jack",
        help="Jack - Universal AI Agent for Digital and Physical Worlds",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        task: str = typer.Argument(..., help="Task to execute"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    ):
        """Execute a task"""
        cli = JackCLI()
        success = cli.run_task(task, verbose=verbose)
        raise typer.Exit(0 if success else 1)

    @app.command()
    def shell(command: str = typer.Argument(..., help="Shell command to run")):
        """Run a shell command directly"""
        cli = JackCLI()
        cli.run_shell(command)

    @app.command()
    def train(
        phase: int = typer.Option(0, "--phase", "-p", help="Training phase"),
        samples: int = typer.Option(500, "--samples", "-s", help="Number of samples"),
    ):
        """Run training"""
        cli = JackCLI()
        cli.train(phase=phase, samples=samples)

    @app.command()
    def status():
        """Show Jack's status"""
        cli = JackCLI()
        cli.show_status()

    @app.command()
    def interactive():
        """Run in interactive mode"""
        cli = JackCLI()
        cli.interactive()

    @app.command("i")
    def interactive_short():
        """Run in interactive mode (shorthand)"""
        cli = JackCLI()
        cli.interactive()

    app()


if __name__ == "__main__":
    run_cli()
