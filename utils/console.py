"""
Console output utilities using Rich library
Provides progress bars, tables, and formatted output
"""

import logging
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn
)
from rich.table import Table

logger = logging.getLogger(__name__)

# Global console instance
console = Console()


def create_progress_bar(description: str = "Processing", total: Optional[int] = None) -> Progress:
    """
    Create a Rich progress bar with standardized formatting.

    Args:
        description: Description text for the progress bar
        total: Total number of items (None for indeterminate progress)

    Returns:
        Progress object (use as context manager)

    Example:
        with create_progress_bar("Processing files", total=len(files)) as progress:
            task = progress.add_task(description, total=len(files))
            for file in files:
                # ... process file ...
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("{task.fields[status]}", style="green"),
        console=console
    )


def print_summary_table(title: str, data: List[Dict[str, Any]],
                       columns: Optional[List[str]] = None) -> None:
    """
    Print a formatted table with summary data.

    Args:
        title: Table title
        data: List of dictionaries with row data
        columns: Optional list of column names (uses dict keys if None)

    Example:
        data = [
            {'File': 'audio1.wav', 'Duration': '30s', 'Status': 'OK'},
            {'File': 'audio2.wav', 'Duration': '45s', 'Status': 'OK'}
        ]
        print_summary_table("Processing Results", data)
    """
    if not data:
        console.print(f"[yellow]No data to display for {title}[/yellow]")
        return

    # Determine columns
    if columns is None:
        columns = list(data[0].keys())

    # Create table
    table = Table(title=title, box=box.ROUNDED)

    # Add columns
    for col in columns:
        table.add_column(col, style="cyan", no_wrap=False)

    # Add rows
    for row in data:
        table.add_row(*[str(row.get(col, '')) for col in columns])

    console.print(table)


def print_status_panel(title: str, content: str, style: str = "green") -> None:
    """
    Print a formatted panel with status information.

    Args:
        title: Panel title
        content: Panel content (supports Rich markup)
        style: Border style (green, yellow, red, blue)

    Example:
        print_status_panel(
            "Processing Complete",
            "Processed [bold]42[/bold] files successfully",
            style="green"
        )
    """
    panel = Panel(content, title=title, border_style=style)
    console.print(panel)


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the configuration.

    Args:
        config: Configuration dictionary
    """
    table = Table(title="Configuration Summary", box=box.SIMPLE)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # AI Models
    table.add_row("GPT Model", config.get('ai_models', {}).get('openai', {}).get('gpt_model', 'N/A'))
    table.add_row("Whisper Model", config.get('ai_models', {}).get('whisper', {}).get('model_name', 'N/A'))

    # Weather API
    table.add_row("Weather Provider", config.get('weather_api', {}).get('provider', 'N/A'))

    # Geocoding
    table.add_row("Geocoding Provider", config.get('geocoding_api', {}).get('provider', 'N/A'))

    # File paths
    table.add_row("Output JSON", config.get('file_paths', {}).get('output_json', 'N/A'))

    # Config source
    if '_meta' in config and 'source' in config['_meta']:
        table.add_row("Config Source", config['_meta']['source'])

    console.print(table)


def print_dry_run_summary(title: str, actions: List[str]) -> None:
    """
    Print a summary of actions that would be performed in dry-run mode.

    Args:
        title: Summary title
        actions: List of action descriptions
    """
    content = "[yellow]DRY-RUN MODE - No changes will be made[/yellow]\n\n"
    content += "The following actions would be performed:\n\n"

    for i, action in enumerate(actions, 1):
        content += f"  {i}. {action}\n"

    content += "\n[green]Run without --dry-run to execute these actions[/green]"

    print_status_panel(title, content, style="yellow")


def print_file_list(files: List[str], title: str = "Files", max_display: int = 10) -> None:
    """
    Print a formatted list of files.

    Args:
        files: List of file paths
        title: List title
        max_display: Maximum number of files to display (shows total if exceeded)
    """
    console.print(f"\n[bold]{title}:[/bold] {len(files)} files")

    if len(files) <= max_display:
        for f in files:
            console.print(f"  • {f}")
    else:
        for f in files[:max_display]:
            console.print(f"  • {f}")
        console.print(f"  ... and {len(files) - max_display} more files")


def print_error(message: str, title: str = "Error") -> None:
    """
    Print an error message in a formatted panel.

    Args:
        message: Error message
        title: Error title
    """
    print_status_panel(title, f"[red]{message}[/red]", style="red")


def print_warning(message: str) -> None:
    """
    Print a warning message.

    Args:
        message: Warning message
    """
    console.print(f"[yellow]⚠️  {message}[/yellow]")


def print_success(message: str) -> None:
    """
    Print a success message.

    Args:
        message: Success message
    """
    console.print(f"[green]✓ {message}[/green]")


def print_info(message: str) -> None:
    """
    Print an informational message.

    Args:
        message: Info message
    """
    console.print(f"[blue]ℹ️  {message}[/blue]")


def print_header(text: str) -> None:
    """
    Print a formatted header.

    Args:
        text: Header text
    """
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{text.center(60)}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def print_section(text: str) -> None:
    """
    Print a section divider.

    Args:
        text: Section text
    """
    console.print(f"\n[bold]{text}[/bold]")
    console.print("[dim]" + "-" * len(text) + "[/dim]")


if __name__ == '__main__':
    # Test console utilities
    print_header("FRAMAI Console Utilities Test")

    # Test summary table
    test_data = [
        {'File': 'recording1.wav', 'Duration': '30s', 'Status': '✓ OK'},
        {'File': 'recording2.wav', 'Duration': '45s', 'Status': '✓ OK'},
        {'File': 'recording3.wav', 'Duration': '60s', 'Status': '✓ OK'},
    ]
    print_summary_table("Processing Results", test_data)

    # Test status messages
    print("\n")
    print_success("Files processed successfully")
    print_warning("Some files had no GPS data")
    print_error("Failed to load API key", title="Configuration Error")
    print_info("Using default configuration")

    # Test sections
    print_section("Configuration")
    print("  • GPT Model: gpt-4")
    print("  • Whisper Model: base.en")

    # Test file list
    test_files = [f"file{i}.wav" for i in range(15)]
    print_file_list(test_files, "Audio Files", max_display=5)

    # Test dry-run summary
    test_actions = [
        "Process 10 audio files",
        "Extract GPS from 5 images",
        "Fetch weather data for 3 locations",
        "Generate output JSON"
    ]
    print_dry_run_summary("Planned Actions", test_actions)

    # Test progress bar
    import time
    print_section("Progress Bar Test")
    with create_progress_bar("Processing test files", total=5) as progress:
        task = progress.add_task("Processing", total=5, status="Starting...")
        for i in range(5):
            time.sleep(0.5)
            progress.update(task, status=f"File {i+1}/5")
            progress.advance(task)
        progress.update(task, status="Complete!")
