"""
Utility modules for FRAMAI
"""

from .exif import extract_gps_from_image, extract_datetime_from_image
from .geocoding import GeocodingClient
from .console import create_progress_bar, print_summary_table

# Audio utilities (optional - may not work on Python 3.13+)
try:
    from .audio import extract_audio_segment, apply_fade
    __all__ = [
        'extract_gps_from_image',
        'extract_datetime_from_image',
        'GeocodingClient',
        'create_progress_bar',
        'print_summary_table',
        'extract_audio_segment',
        'apply_fade'
    ]
except ImportError:
    __all__ = [
        'extract_gps_from_image',
        'extract_datetime_from_image',
        'GeocodingClient',
        'create_progress_bar',
        'print_summary_table'
    ]
