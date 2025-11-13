"""
Postprocess command implementation
Trims and applies fade effects to audio files
Refactored from postprocess.py to use new infrastructure
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from utils.console import (
    create_progress_bar,
    print_dry_run_summary,
    print_error,
    print_header,
    print_info,
    print_section,
    print_success,
    print_warning
)

logger = logging.getLogger(__name__)


def run_postprocess_command(directory: str, config: Dict[str, Any],  # pylint: disable=unused-argument
                           options: Dict[str, Any], dry_run: bool = False,
                           verbose: bool = False) -> bool:  # pylint: disable=unused-argument
    """
    Main entry point for postprocess command.

    Args:
        directory: Directory containing audio files
        config: Configuration dictionary
        options: Command options (json_file, fade_duration, suffix, etc.)
        dry_run: If True, show what would be done without executing
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    print_header("FRAMAI - Audio Post-Processing")

    # Extract options
    json_file = options['json_file']
    fade_duration = options['fade_duration']
    output_suffix = options['output_suffix']
    header_buffer = options['header_buffer']
    footer_buffer = options['footer_buffer']

    # Show configuration
    print_section("Configuration")
    print(f"  Directory: {directory}")
    print(f"  JSON file: {json_file}")
    print(f"  Fade duration: {fade_duration}s")
    print(f"  Output suffix: {output_suffix}")
    print(f"  Header buffer: {header_buffer}s")
    print(f"  Footer buffer: {footer_buffer}s")

    # Resolve JSON path
    json_path = Path(json_file)
    if not json_path.is_absolute() and not str(json_file).startswith(directory):
        json_path = Path(directory) / json_file

    # Load JSON file
    print_section("Loading trim points from JSON")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print_error(f"JSON file not found: {json_path}")
        return False
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON file: {e}")
        return False

    # Get audio events with trim points
    audio_events = data.get('audio_events', [])
    events_to_process = [e for e in audio_events if 'header' in e or 'footer' in e]

    if not events_to_process:
        print_warning("No audio events with trim points found")
        print_info("Run 'fram-cli transcribe' first to generate trim points")
        return False

    print_success(f"Found {len(events_to_process)} audio files to process")

    # Dry-run mode
    if dry_run:
        actions = [
            f"Process {len(events_to_process)} audio files",
            f"Trim based on header/footer timestamps from JSON",
            f"Apply {fade_duration}s fade in/out effects",
            f"Save with suffix: {output_suffix}"
        ]

        for event in events_to_process[:3]:  # Show first 3
            filename = event.get('audio_filename', 'unknown')
            header = event.get('header', 0)
            footer = event.get('footer', 0)
            actions.append(f"  {filename}: trim at {header:.1f}s / {footer:.1f}s")  # noqa: W1309

        print_dry_run_summary("Audio Post-Processing Plan", actions)
        return True

    # Process audio files
    print_section("Processing audio files")
    processed_count = process_audio_files(events_to_process, directory, fade_duration,
                                         output_suffix, header_buffer, footer_buffer)

    if processed_count == 0:
        print_error("Failed to process audio files")
        return False

    print_success(f"Processed {processed_count} audio files")
    return True


def process_audio_files(audio_events: list, directory: str, fade_duration: int,
                       output_suffix: str, header_buffer: int, footer_buffer: int) -> int:
    """
    Process audio files with trimming and fading.

    Args:
        audio_events: List of audio event dictionaries with trim points
        directory: Base directory
        fade_duration: Fade duration in seconds
        output_suffix: Suffix for output files
        header_buffer: Additional buffer for header in seconds
        footer_buffer: Additional buffer for footer in seconds

    Returns:
        Number of successfully processed files
    """
    from pydub import AudioSegment

    processed_count = 0
    dir_path = Path(directory)

    with create_progress_bar("Processing audio", total=len(audio_events)) as progress:
        task = progress.add_task("Processing", total=len(audio_events), status="Starting...")

        for event in audio_events:
            audio_filename = event.get('audio_filename')
            if not audio_filename:
                progress.advance(task)
                continue

            audio_filepath = dir_path / audio_filename

            if not audio_filepath.exists():
                logger.warning(f"Audio file not found: {audio_filepath}")
                progress.update(task, status=f"⚠️  Not found: {audio_filename[:20]}")
                progress.advance(task)
                continue

            try:
                # Load audio
                audio = AudioSegment.from_file(str(audio_filepath))

                # Get trim points (in seconds)
                header = event.get('header', 0) + header_buffer
                footer = event.get('footer', 0) + footer_buffer

                # Convert to milliseconds
                start_time = int(header * 1000) if header else 0
                end_time = len(audio) - int(footer * 1000) if footer else len(audio)

                # Trim audio
                trimmed_audio = audio[start_time:end_time]

                # Apply fade in/out if audio is long enough
                fade_ms = fade_duration * 1000
                min_length = fade_ms * 3  # Need at least 3x fade duration

                if len(trimmed_audio) > min_length:
                    faded_audio = trimmed_audio.fade_in(fade_ms).fade_out(fade_ms)
                else:
                    logger.info(f"Audio too short for {fade_duration}s fades: {audio_filename}")
                    faded_audio = trimmed_audio

                # Create output filename
                file_stem = audio_filepath.stem
                file_suffix = audio_filepath.suffix
                output_filename = f"{file_stem}{output_suffix}{file_suffix}"
                output_path = dir_path / output_filename

                # Export
                faded_audio.export(str(output_path), format=file_suffix[1:])

                processed_count += 1
                progress.update(task, status=f"✓ {audio_filename[:20]}")
                progress.advance(task)

            except Exception as e:
                logger.error(f"Error processing {audio_filename}: {e}")
                progress.update(task, status=f"❌ Error: {audio_filename[:20]}")
                progress.advance(task)

    return processed_count


if __name__ == '__main__':
    # Test postprocess command
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        test_dir = sys.argv[1]

        # Load config
        from config import load_config
        config = load_config()

        # Test options
        options = {
            'json_file': 'fram.json',
            'fade_duration': 30,
            'output_suffix': '_POST',
            'header_buffer': 0,
            'footer_buffer': 0
        }

        # Run command
        success = run_postprocess_command(
            directory=test_dir,
            config=config,
            options=options,
            dry_run=False,
            verbose=True
        )

        sys.exit(0 if success else 1)
    else:
        print("Usage: python postprocess_cmd.py <directory>")
