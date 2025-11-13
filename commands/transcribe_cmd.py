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
    # Fix OpenMP library conflicts on macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'

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

    # Check for Whisper availability
    try:
        import whisper
        whisper_available = True
    except ImportError:
        print_error("Whisper not installed. Install with: pip install openai-whisper")
        return False

    # Check for audio processing library
    try:
        from pydub import AudioSegment
        audio_available = True
    except ImportError:
        print_error("Pydub not available (Python 3.13+ compatibility issue)")
        print_info("Alternative: Use OpenAI Whisper API instead of local model")
        return False

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
    try:
        import whisper
        from pydub import AudioSegment
    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        return None

    # Load Whisper model
    print_info(f"Loading Whisper model: {model}")
    try:
        model_path = config['ai_models']['whisper']['model_path']
        whisper._download(whisper._MODELS[model], model_path, False)
        whisper_model = whisper.load_model(model)
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None

    # Transcribe each file
    audio_events = []

    with create_progress_bar("Transcribing audio", total=len(audio_files)) as progress:
        task = progress.add_task("Processing", total=len(audio_files), status="Starting...")

        for audio_path in audio_files:
            try:
                # Load audio
                audio = AudioSegment.from_file(str(audio_path))
                audio_duration_ms = len(audio)
                audio_duration_s = audio_duration_ms / 1000

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

                # Transcribe first N seconds (header)
                transcribe_duration = min(duration, int(audio_duration_s))
                text = ""
                header_segment = audio[:transcribe_duration * 1000]

                progress.update(task, status=f"Transcribing header: {audio_path.name[:20]}")
                header_result = transcribe_segment(header_segment, whisper_model, word_timestamps)

                if header_result and header_result.get('segments'):
                    text += header_result['text']
                    # Get timestamp of last word (for trimming)
                    first_word_start, last_word_end = get_first_and_last_word_time(header_result)
                    if last_word_end:
                        data['header'] = last_word_end

                # Transcribe last N seconds (footer) if audio is long enough
                if audio_duration_s > duration * 2:
                    footer_segment = audio[-transcribe_duration * 1000:]

                    progress.update(task, status=f"Transcribing footer: {audio_path.name[:20]}")
                    footer_result = transcribe_segment(footer_segment, whisper_model, word_timestamps)

                    if footer_result and footer_result.get('segments'):
                        text += " " + footer_result['text']
                        # Get timestamp of first word (for trimming)
                        first_word_start, last_word_end = get_first_and_last_word_time(footer_result)
                        if first_word_start:
                            data['footer'] = first_word_start

                # Store transcribed text
                if text:
                    data['extracted_text'] = text.strip()
                    progress.update(task, status=f"✓ {audio_path.name[:20]}")
                else:
                    progress.update(task, status=f"⚠️  No speech: {audio_path.name[:20]}")

                audio_events.append(data)
                progress.advance(task)

            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                progress.update(task, status=f"❌ Error: {audio_path.name[:20]}")
                progress.advance(task)

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


def transcribe_segment(audio_segment, whisper_model, word_timestamps: bool) -> Optional[Dict[str, Any]]:
    """
    Transcribe an audio segment using Whisper.

    Args:
        audio_segment: AudioSegment to transcribe
        whisper_model: Loaded Whisper model
        word_timestamps: Include word-level timestamps

    Returns:
        Transcription result dictionary or None
    """
    try:
        # Export to temporary file
        temp_file = "temp.mp3"
        audio_segment.export(temp_file, format="mp3")

        # Transcribe
        result = whisper_model.transcribe(
            temp_file,
            word_timestamps=word_timestamps,
            fp16=False
        )

        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass

        return result

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None


def get_first_and_last_word_time(transcription_result: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Get the start time of the first word and end time of the last word.

    Args:
        transcription_result: Whisper transcription result

    Returns:
        Tuple of (first_word_start, last_word_end) in seconds
    """
    segments = transcription_result.get("segments", [])
    if not segments:
        return None, None

    try:
        first_segment = segments[0]
        last_segment = segments[-1]

        first_word_start = first_segment["words"][0]["start"]
        last_word_end = last_segment["words"][-1]["end"]

        return first_word_start, last_word_end
    except (KeyError, IndexError):
        # If word timestamps not available, use segment timestamps
        try:
            first_word_start = segments[0].get("start")
            last_word_end = segments[-1].get("end")
            return first_word_start, last_word_end
        except:
            return None, None


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
        if not output_path.is_absolute():
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
