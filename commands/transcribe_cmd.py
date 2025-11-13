"""
Transcribe command implementation
Transcribes audio files using Whisper (local or API)
Refactored from transcribe.py to use new infrastructure
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import time

# Import utilities
from utils.console import (
    print_header,
    print_section,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_summary_table,
    print_dry_run_summary,
    create_progress_bar
)

logger = logging.getLogger(__name__)


def run_transcribe_command(directory: str, config: Dict[str, Any],
                          options: Dict[str, Any], dry_run: bool = False,
                          verbose: bool = False) -> bool:
    """
    Main entry point for transcribe command.

    Args:
        directory: Directory containing audio files
        config: Configuration dictionary
        options: Command options (output_file, duration, model, etc.)
        dry_run: If True, show what would be done without executing
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    print_header("FRAMAI - Audio Transcription")

    # Extract options
    output_file = options['output_file']
    duration = options['duration']
    model = options['model']
    word_timestamps = options['word_timestamps']
    file_types = options['file_types']

    # Show configuration
    print_section("Configuration")
    print(f"  Directory: {directory}")
    print(f"  Output file: {output_file}")
    print(f"  Duration to transcribe: {duration}s")
    print(f"  Whisper model: {model}")
    print(f"  Word timestamps: {'enabled' if word_timestamps else 'disabled'}")
    print(f"  File types: {', '.join(file_types)}")

    # Find audio files
    print_section("Scanning for audio files")
    audio_files = find_audio_files(directory, file_types)

    if not audio_files:
        print_warning(f"No audio files found in {directory}")
        return False

    print_success(f"Found {len(audio_files)} audio files")

    # Dry-run mode
    if dry_run:
        actions = [
            f"Process {len(audio_files)} audio files",
            f"Transcribe first and last {duration}s of each file using Whisper ({model})",
            "Extract header and footer timestamps for trimming",
            f"Save transcriptions to {output_file}"
        ]

        print_dry_run_summary("Audio Transcription Plan", actions)
        return True

    # Process audio files
    print_section("Transcribing audio files")
    results = process_audio_files(audio_files, directory, config, duration, model,
                                  word_timestamps, verbose)

    if not results:
        print_error("Failed to transcribe audio files")
        return False

    # Display summary
    display_summary(results)

    # Save results
    print_section("Saving results")
    success = save_results(results, output_file, directory)

    if success:
        print_success(f"Results saved to {output_file}")
        return True
    else:
        print_error(f"Failed to save results to {output_file}")
        return False


def find_audio_files(directory: str, file_types: List[str]) -> List[Path]:
    """
    Find all audio files in directory with specified types.

    Args:
        directory: Directory to search
        file_types: List of file extensions (e.g., ['wav', 'mp3'])

    Returns:
        List of Path objects for found audio files, sorted by modification time
    """
    dir_path = Path(directory)
    audio_files = []

    # Normalize extensions
    extensions = []
    for ext in file_types:
        if not ext.startswith('.'):
            ext = '.' + ext
        extensions.append(ext.lower())

    # Search for files
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in extensions:
                audio_files.append(file_path)

    # Sort by modification time (oldest first)
    audio_files.sort(key=lambda f: f.stat().st_mtime)

    return audio_files


def process_audio_files(audio_files: List[Path], directory: str,
                       config: Dict[str, Any], duration: int, model: str,
                       word_timestamps: bool, verbose: bool) -> Optional[Dict[str, Any]]:
    """
    Process all audio files and transcribe them.

    Args:
        audio_files: List of audio file paths
        directory: Base directory
        config: Configuration dictionary
        duration: Duration in seconds to transcribe
        model: Whisper model name
        word_timestamps: Include word-level timestamps
        verbose: Verbose output

    Returns:
        Dictionary with transcription results or None if failed
    """
    from whisper_turbo import MLXWhisperTranscriber
    from pydub import AudioSegment

    # Load Whisper model (using whisper-turbo for Apple Silicon)
    print_info(f"Loading Whisper model: {model} (using MLX for Apple Silicon)")
    transcriber = MLXWhisperTranscriber(model_name=model)
    print_success("Model loaded successfully")

    # Transcribe each file
    audio_events = []

    print_info(f"Transcribing {len(audio_files)} audio files...")

    for audio_path in audio_files:
        try:
            # Load audio file
            audio = AudioSegment.from_file(str(audio_path))
            audio_duration_s = len(audio) / 1000

            # Get creation date
            created_date = time.strftime(
                '%Y-%m-%dT%H:%M:%S',
                time.gmtime(audio_path.stat().st_ctime)
            )

            # Prepare data structure
            data = {
                "audio_filename": audio_path.name,
                "created_date": created_date,
                "duration_seconds": audio_duration_s
            }

            print_info(f"Processing: {audio_path.name} ({audio_duration_s:.1f}s)")

            transcribe_duration = min(duration, int(audio_duration_s))
            full_text = ""

            # Extract and transcribe first N seconds (header)
            print_info(f"  Transcribing header ({transcribe_duration}s)...")
            header_segment = audio[:transcribe_duration * 1000]
            temp_header = "temp_header.wav"
            header_segment.export(temp_header, format="wav")

            header_text, header_segments = transcriber.transcribe_file(temp_header)
            if header_text:
                full_text += header_text
                # Get timestamp where speech ends in header
                if header_segments:
                    data['header'] = max(seg['end'] for seg in header_segments)

            os.remove(temp_header)

            # Extract and transcribe last N seconds (footer) if audio is long enough
            if audio_duration_s > duration * 2:
                print_info(f"  Transcribing footer ({transcribe_duration}s)...")
                footer_segment = audio[-transcribe_duration * 1000:]
                temp_footer = "temp_footer.wav"
                footer_segment.export(temp_footer, format="wav")

                footer_text, footer_segments = transcriber.transcribe_file(temp_footer)
                if footer_text:
                    full_text += " " + footer_text
                    # Get timestamp where speech starts in footer
                    if footer_segments:
                        data['footer'] = min(seg['start'] for seg in footer_segments)

                os.remove(temp_footer)

            # Store transcribed text
            if full_text:
                data['extracted_text'] = full_text.strip()
                print_success(f"✓ Transcribed {audio_path.name}")
            else:
                print_warning(f"⚠️  No speech detected: {audio_path.name}")

            audio_events.append(data)

        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}")
            print_error(f"❌ Error processing {audio_path.name}: {e}")

    results = {
        'audio_events': audio_events,
        'summary': {
            'total_files': len(audio_files),
            'transcribed': len([e for e in audio_events if e.get('extracted_text')]),
            'model': model,
            'duration': duration
        }
    }

    return results


def display_summary(results: Dict[str, Any]) -> None:
    """
    Display a summary table of transcription results.

    Args:
        results: Results dictionary
    """
    print_section("Transcription Summary")

    summary_data = [
        {'Metric': 'Total audio files', 'Value': results['summary']['total_files']},
        {'Metric': 'Successfully transcribed', 'Value': results['summary']['transcribed']},
        {'Metric': 'Model used', 'Value': results['summary']['model']},
        {'Metric': 'Duration per file', 'Value': f"{results['summary']['duration']}s"},
    ]

    print_summary_table("Results", summary_data)

    # Show some transcriptions
    if results['audio_events']:
        print_section("Transcriptions (first 5)")
        for i, event in enumerate(results['audio_events'][:5], 1):
            text = event.get('extracted_text', 'No text extracted')
            print(f"  {i}. {event['audio_filename']}")
            print(f"     {text[:100]}{'...' if len(text) > 100 else ''}")
            print()


def save_results(results: Dict[str, Any], output_file: str, directory: str) -> bool:
    """
    Save transcription results to JSON file.

    Merges with existing data if file exists.

    Args:
        results: Results dictionary
        output_file: Output file path
        directory: Base directory

    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve output path
        output_path = Path(output_file)
        if not output_path.is_absolute() and not str(output_file).startswith(directory):
            output_path = Path(directory) / output_file

        # Load existing data if present
        if output_path.exists():
            with open(output_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Ensure data is a dictionary
        if not isinstance(data, dict):
            data = {}

        # Merge audio events
        if 'audio_events' not in data:
            data['audio_events'] = []

        data['audio_events'].extend(results['audio_events'])

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved results to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


if __name__ == '__main__':
    # Test transcribe command
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        test_dir = sys.argv[1]

        # Load config
        from config import load_config
        config = load_config()

        # Test options
        options = {
            'output_file': 'test_transcribe.json',
            'duration': 30,
            'model': 'base.en',
            'word_timestamps': True,
            'file_types': ['wav', 'mp3']
        }

        # Run command
        success = run_transcribe_command(
            directory=test_dir,
            config=config,
            options=options,
            dry_run=False,
            verbose=True
        )

        sys.exit(0 if success else 1)
    else:
        print("Usage: python transcribe_cmd.py <directory>")
