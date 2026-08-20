"""Entry shim so `uv run python main.py` keeps working. See jtech_cli/cli.py."""

from jtech_cli.cli import main

if __name__ == "__main__":
    main()
