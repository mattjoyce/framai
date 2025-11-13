#!/usr/bin/env python3
"""
FRAMAI - Field Recording Audio/Media Analysis & Integration
Main CLI entry point using Click
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional

# Import configuration and utilities
from config import load_config, print_config_summary
from utils.console import (
    console,
    print_header,
    print_error,
    print_success,
    print_warning,
    print_config_summary as print_config_table
)

# Version
__version__ = "1.0.0"


# Custom Click context class to hold shared state
class FramContext:
    def __init__(self):
        self.config = None
        self.verbose = False
        self.dry_run = False


# Pass context between commands
pass_context = click.make_pass_decorator(FramContext, ensure=True)


@click.group()
@click.version_option(version=__version__, prog_name="fram-cli")
@click.option('--config', '-c', 'config_path',
              type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
@click.option('--dry-run', is_flag=True,
              help='Show what would be done without executing')
@click.pass_context
def cli(ctx, config_path: Optional[str], verbose: bool, dry_run: bool):
    """
    FRAMAI - Field Recording Audio/Media Analysis & Integration

    A comprehensive tool for processing field recordings:
    - Extract GPS and metadata from images
    - Fetch historical weather data
    - Transcribe audio with Whisper
    - Refine transcriptions with GPT
    - Post-process audio (trim, fade)

    Examples:
        fram-cli images ./recordings/
        fram-cli transcribe ./audio/ --duration 30
        fram-cli refine fram.json
        fram-cli postprocess ./audio/ --fade 30
    """
    # Ensure context object exists
    ctx.ensure_object(FramContext)

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    # Store settings in context
    ctx.obj.verbose = verbose
    ctx.obj.dry_run = dry_run

    # Load configuration
    try:
        ctx.obj.config = load_config(config_path)

        if verbose:
            print_config_table(ctx.obj.config)

    except Exception as e:
        print_error(f"Failed to load configuration: {e}", title="Configuration Error")
        sys.exit(1)

    # Show dry-run warning
    if dry_run:
        print_warning("DRY-RUN MODE: No changes will be made")


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output', '-o', default=None,
              help='Output JSON file (default: from config)')
@click.option('--weather/--no-weather', default=True,
              help='Fetch weather data (default: enabled)')
@click.option('--geocode/--no-geocode', default=True,
              help='Fetch location names (default: enabled)')
@click.option('--threshold', type=int, default=None,
              help='Location clustering threshold in meters (default: from config)')
@click.option('--extensions', default=None,
              help='Comma-separated image extensions (e.g., jpg,png)')
@pass_context
def images(ctx: FramContext, directory: str, output: Optional[str],
           weather: bool, geocode: bool, threshold: Optional[int],
           extensions: Optional[str]):
    """
    Process images: extract GPS/EXIF, fetch weather & location data.

    Scans DIRECTORY for image files, extracts GPS coordinates and timestamps
    from EXIF data, fetches historical weather data from Open-Meteo, and
    optionally geocodes locations using OpenStreetMap Nominatim.

    Output is saved as JSON with all extracted metadata.

    Example:
        fram-cli images ./Final/ --output images.json
    """
    # Import command implementation
    from commands.images_cmd import run_images_command

    # Prepare options
    options = {
        'output_file': output or ctx.config['file_paths']['output_json'],
        'weather_enabled': weather,
        'geocode_enabled': geocode,
        'threshold_meters': threshold or ctx.config['processing']['image']['location_threshold_meters'],
        'extensions': extensions.split(',') if extensions else ctx.config['file_paths']['supported_formats']['image']
    }

    # Run command
    try:
        success = run_images_command(
            directory=directory,
            config=ctx.config,
            options=options,
            dry_run=ctx.dry_run,
            verbose=ctx.verbose
        )

        if success:
            print_success(f"Images processed successfully")
            sys.exit(0)
        else:
            print_error("Image processing failed")
            sys.exit(1)

    except Exception as e:
        print_error(f"Command failed: {e}")
        if ctx.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output', '-o', default=None,
              help='Output JSON file (default: from config)')
@click.option('--duration', type=int, default=None,
              help='Seconds to transcribe (default: from config)')
@click.option('--model', default=None,
              help='Whisper model name (default: from config)')
@click.option('--types', default=None,
              help='Comma-separated audio file types (e.g., wav,mp3)')
@click.option('--timestamps/--no-timestamps', default=True,
              help='Include word timestamps (default: enabled)')
@pass_context
def transcribe(ctx: FramContext, directory: str, output: Optional[str],
               duration: Optional[int], model: Optional[str],
               types: Optional[str], timestamps: bool):
    """
    Transcribe audio files using Whisper.

    Scans DIRECTORY for audio files and transcribes them using OpenAI Whisper.
    Can use either local Whisper model or OpenAI API.

    For long files, extracts first N seconds (specified by --duration).
    Output includes transcription text and optional word-level timestamps.

    Example:
        fram-cli transcribe ./recordings/ --duration 30 --model base.en
    """
    # Import command implementation
    from commands.transcribe_cmd import run_transcribe_command

    # Prepare options
    options = {
        'output_file': output or ctx.config['file_paths']['output_json'],
        'duration': duration or ctx.config['processing']['audio']['default_duration'],
        'model': model or ctx.config['ai_models']['whisper']['model_name'],
        'word_timestamps': timestamps,
        'file_types': types.split(',') if types else ctx.config['file_paths']['supported_formats']['audio']
    }

    # Run command
    try:
        success = run_transcribe_command(
            directory=directory,
            config=ctx.config,
            options=options,
            dry_run=ctx.dry_run,
            verbose=ctx.verbose
        )

        if success:
            print_success("Transcription completed successfully")
            sys.exit(0)
        else:
            print_error("Transcription failed")
            sys.exit(1)

    except Exception as e:
        print_error(f"Command failed: {e}")
        if ctx.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('json_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', '-o', default=None,
              help='Output JSON file (default: overwrite input)')
@click.option('--model', default=None,
              help='GPT model name (default: from config)')
@click.option('--temperature', type=float, default=None,
              help='GPT temperature (default: from config)')
@click.option('--prompt', default=None,
              help='Custom prompt template')
@pass_context
def refine(ctx: FramContext, json_file: str, output: Optional[str],
           model: Optional[str], temperature: Optional[float],
           prompt: Optional[str]):
    """
    Refine transcriptions using GPT-4.

    Takes a JSON file with transcriptions (from transcribe command) and
    refines them using GPT-4 to create concise, professional descriptions.

    Useful for cleaning up Whisper transcriptions and formatting them
    according to audio library standards.

    Example:
        fram-cli refine fram.json --output refined.json
    """
    # Import command implementation
    from commands.refine_cmd import run_refine_command

    # Prepare options
    options = {
        'output_file': output or json_file,  # Overwrite by default
        'model': model or ctx.config['ai_models']['openai']['gpt_model'],
        'temperature': temperature or ctx.config['ai_models']['openai']['temperature'],
        'prompt_template': prompt
    }

    # Run command
    try:
        success = run_refine_command(
            json_file=json_file,
            config=ctx.config,
            options=options,
            dry_run=ctx.dry_run,
            verbose=ctx.verbose
        )

        if success:
            print_success("Refinement completed successfully")
            sys.exit(0)
        else:
            print_error("Refinement failed")
            sys.exit(1)

    except Exception as e:
        print_error(f"Command failed: {e}")
        if ctx.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--json', default=None,
              help='Input JSON with trim points (default: from config)')
@click.option('--fade', type=int, default=None,
              help='Fade duration in seconds (default: from config)')
@click.option('--suffix', default=None,
              help='Output file suffix (default: from config)')
@click.option('--header', type=int, default=0,
              help='Header buffer in seconds (default: 0)')
@click.option('--footer', type=int, default=0,
              help='Footer buffer in seconds (default: 0)')
@pass_context
def postprocess(ctx: FramContext, directory: str, json: Optional[str],
                fade: Optional[int], suffix: Optional[str],
                header: int, footer: int):
    """
    Post-process audio: trim and fade files.

    Reads trim points from JSON file (with 'header' and 'footer' fields)
    and trims audio files accordingly. Applies fade in/out effects.

    Output files are saved with specified suffix (e.g., filename_POST.wav).

    Example:
        fram-cli postprocess ./recordings/ --fade 30 --suffix _FINAL
    """
    # Import command implementation
    from commands.postprocess_cmd import run_postprocess_command

    # Prepare options
    options = {
        'json_file': json or ctx.config['file_paths']['output_json'],
        'fade_duration': fade or ctx.config['processing']['audio']['fade_duration'],
        'output_suffix': suffix or ctx.config['file_paths']['output_suffix'],
        'header_buffer': header,
        'footer_buffer': footer
    }

    # Run command
    try:
        success = run_postprocess_command(
            directory=directory,
            config=ctx.config,
            options=options,
            dry_run=ctx.dry_run,
            verbose=ctx.verbose
        )

        if success:
            print_success("Post-processing completed successfully")
            sys.exit(0)
        else:
            print_error("Post-processing failed")
            sys.exit(1)

    except Exception as e:
        print_error(f"Command failed: {e}")
        if ctx.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@pass_context
def config_info(ctx: FramContext):
    """Show current configuration."""
    print_header("FRAMAI Configuration")
    print_config_summary(ctx.config)


if __name__ == '__main__':
    cli()
