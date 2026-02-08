"""
Jack - Universal AI Agent

Entry point for running Jack from command line:
    python -m jack

Or after installation:
    jack
"""

from .interfaces.cli import run_cli

if __name__ == "__main__":
    run_cli()
