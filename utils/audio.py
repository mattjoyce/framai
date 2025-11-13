"""
Audio processing utilities
Handles audio extraction, trimming, fading using pydub
"""

from pydub import AudioSegment
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_audio(file_path: str) -> Optional[AudioSegment]:
    """
    Load an audio file using pydub.

    Args:
        file_path: Path to audio file

    Returns:
        AudioSegment object or None if load fails
    """
    try:
        audio = AudioSegment.from_file(file_path)
        logger.debug(f"Loaded audio: {file_path} ({len(audio)/1000:.1f}s)")
        return audio
    except Exception as e:
        logger.error(f"Failed to load audio {file_path}: {e}")
        return None


def extract_audio_segment(file_path: str, start_ms: int, end_ms: int,
                          output_path: Optional[str] = None) -> Optional[AudioSegment]:
    """
    Extract a segment from an audio file.

    Args:
        file_path: Path to input audio file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        output_path: Optional path to save extracted segment

    Returns:
        AudioSegment of extracted portion or None if extraction fails
    """
    try:
        audio = load_audio(file_path)
        if audio is None:
            return None

        # Extract segment
        segment = audio[start_ms:end_ms]
        logger.debug(f"Extracted segment: {start_ms}ms to {end_ms}ms ({len(segment)/1000:.1f}s)")

        # Save if output path provided
        if output_path:
            save_audio(segment, output_path)

        return segment

    except Exception as e:
        logger.error(f"Failed to extract segment from {file_path}: {e}")
        return None


def apply_fade(audio: AudioSegment, fade_in_ms: int = 0, fade_out_ms: int = 0) -> AudioSegment:
    """
    Apply fade in/out effects to audio.

    Args:
        audio: Input AudioSegment
        fade_in_ms: Fade in duration in milliseconds
        fade_out_ms: Fade out duration in milliseconds

    Returns:
        AudioSegment with fades applied
    """
    try:
        if fade_in_ms > 0:
            audio = audio.fade_in(fade_in_ms)
            logger.debug(f"Applied fade in: {fade_in_ms}ms")

        if fade_out_ms > 0:
            audio = audio.fade_out(fade_out_ms)
            logger.debug(f"Applied fade out: {fade_out_ms}ms")

        return audio

    except Exception as e:
        logger.error(f"Failed to apply fade: {e}")
        return audio


def trim_audio(file_path: str, start_ms: int, end_ms: int,
               fade_duration_ms: int = 0,
               output_path: Optional[str] = None) -> Optional[AudioSegment]:
    """
    Trim audio file to specified range with optional fades.

    Args:
        file_path: Path to input audio file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        fade_duration_ms: Duration of fade in/out in milliseconds
        output_path: Optional path to save trimmed audio

    Returns:
        AudioSegment of trimmed audio or None if trim fails
    """
    try:
        audio = load_audio(file_path)
        if audio is None:
            return None

        # Trim to specified range
        trimmed = audio[start_ms:end_ms]
        logger.debug(f"Trimmed audio: {start_ms}ms to {end_ms}ms")

        # Apply fades if specified
        if fade_duration_ms > 0:
            # Check if audio is long enough for fades
            min_length = fade_duration_ms * 2
            if len(trimmed) >= min_length:
                trimmed = apply_fade(trimmed, fade_duration_ms, fade_duration_ms)
            else:
                logger.warning(f"Audio too short for {fade_duration_ms}ms fades")

        # Save if output path provided
        if output_path:
            save_audio(trimmed, output_path)

        return trimmed

    except Exception as e:
        logger.error(f"Failed to trim audio {file_path}: {e}")
        return None


def save_audio(audio: AudioSegment, output_path: str, format: Optional[str] = None) -> bool:
    """
    Save AudioSegment to file.

    Args:
        audio: AudioSegment to save
        output_path: Output file path
        format: Audio format (auto-detected from extension if None)

    Returns:
        True if save successful, False otherwise
    """
    try:
        # Auto-detect format from extension if not specified
        if format is None:
            ext = Path(output_path).suffix.lower().lstrip('.')
            format = ext if ext else 'wav'

        audio.export(output_path, format=format)
        logger.info(f"Saved audio: {output_path} ({len(audio)/1000:.1f}s, {format})")
        return True

    except Exception as e:
        logger.error(f"Failed to save audio to {output_path}: {e}")
        return False


def get_audio_duration(file_path: str) -> Optional[float]:
    """
    Get duration of audio file in seconds.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds or None if file can't be read
    """
    audio = load_audio(file_path)
    if audio:
        return len(audio) / 1000.0
    return None


def convert_audio_format(input_path: str, output_path: str,
                        format: Optional[str] = None) -> bool:
    """
    Convert audio file to different format.

    Args:
        input_path: Input file path
        output_path: Output file path
        format: Output format (auto-detected from extension if None)

    Returns:
        True if conversion successful, False otherwise
    """
    audio = load_audio(input_path)
    if audio:
        return save_audio(audio, output_path, format)
    return False


def extract_first_n_seconds(file_path: str, duration_seconds: int,
                           output_path: Optional[str] = None) -> Optional[AudioSegment]:
    """
    Extract the first N seconds of an audio file.

    Args:
        file_path: Path to input audio file
        duration_seconds: Duration to extract in seconds
        output_path: Optional path to save extracted audio

    Returns:
        AudioSegment of extracted portion or None if extraction fails
    """
    duration_ms = duration_seconds * 1000
    return extract_audio_segment(file_path, 0, duration_ms, output_path)


def extract_last_n_seconds(file_path: str, duration_seconds: int,
                          output_path: Optional[str] = None) -> Optional[AudioSegment]:
    """
    Extract the last N seconds of an audio file.

    Args:
        file_path: Path to input audio file
        duration_seconds: Duration to extract in seconds
        output_path: Optional path to save extracted audio

    Returns:
        AudioSegment of extracted portion or None if extraction fails
    """
    try:
        audio = load_audio(file_path)
        if audio is None:
            return None

        duration_ms = duration_seconds * 1000
        start_ms = max(0, len(audio) - duration_ms)

        return extract_audio_segment(file_path, start_ms, len(audio), output_path)

    except Exception as e:
        logger.error(f"Failed to extract last {duration_seconds}s from {file_path}: {e}")
        return None


def extract_first_and_last_minutes(file_path: str, output_path: Optional[str] = None) -> Optional[AudioSegment]:
    """
    Extract first and last minute of audio, concatenate them.

    Useful for longer recordings where only beginning and end are needed.

    Args:
        file_path: Path to input audio file
        output_path: Optional path to save concatenated audio

    Returns:
        AudioSegment with first and last minutes or None if extraction fails
    """
    try:
        audio = load_audio(file_path)
        if audio is None:
            return None

        one_minute_ms = 60 * 1000

        # If audio is shorter than 2 minutes, return as-is
        if len(audio) <= one_minute_ms * 2:
            logger.debug(f"Audio shorter than 2 minutes, returning full audio")
            return audio

        # Extract first minute
        first_minute = audio[:one_minute_ms]

        # Extract last minute
        last_minute = audio[-one_minute_ms:]

        # Concatenate
        combined = first_minute + last_minute

        logger.debug(f"Extracted first and last minutes ({len(combined)/1000:.1f}s total)")

        # Save if output path provided
        if output_path:
            save_audio(combined, output_path)

        return combined

    except Exception as e:
        logger.error(f"Failed to extract first/last minutes from {file_path}: {e}")
        return None


def normalize_audio(audio: AudioSegment, target_dBFS: float = -20.0) -> AudioSegment:
    """
    Normalize audio to target dBFS level.

    Args:
        audio: Input AudioSegment
        target_dBFS: Target loudness in dBFS (default: -20.0)

    Returns:
        Normalized AudioSegment
    """
    try:
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized = audio.apply_gain(change_in_dBFS)
        logger.debug(f"Normalized audio: {audio.dBFS:.1f} -> {target_dBFS:.1f} dBFS")
        return normalized
    except Exception as e:
        logger.error(f"Failed to normalize audio: {e}")
        return audio


if __name__ == '__main__':
    # Test audio utilities
    import sys
    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Testing audio utilities on: {audio_file}")

        # Get duration
        duration = get_audio_duration(audio_file)
        print(f"Duration: {duration:.1f}s" if duration else "Could not read audio")

        if duration:
            # Test extract first 10 seconds
            print("\nExtracting first 10 seconds...")
            segment = extract_first_n_seconds(audio_file, 10, "test_first_10s.wav")
            if segment:
                print(f"Extracted: {len(segment)/1000:.1f}s")

            # Test with fade
            print("\nExtracting with fade...")
            trimmed = trim_audio(audio_file, 0, 30000, fade_duration_ms=3000, output_path="test_fade.wav")
            if trimmed:
                print(f"Trimmed with fade: {len(trimmed)/1000:.1f}s")

    else:
        print("Usage: python audio.py <audio_file>")
