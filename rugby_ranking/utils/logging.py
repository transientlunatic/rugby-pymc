"""Logging and formatting utilities."""

from typing import Optional


def setup_logging(verbose: bool = True) -> None:
    """Configure logging for CLI output."""
    import logging

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def print_section(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print("=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"✗ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ  {message}")
