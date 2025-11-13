"""
EXIF metadata extraction utilities
Handles GPS coordinates and datetime extraction from images
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def extract_gps_from_image(image_path: str, precision: int = 6) -> Optional[Tuple[float, float]]:
    """
    Extract GPS coordinates from image EXIF data.

    Args:
        image_path: Path to image file
        precision: Number of decimal places for coordinates

    Returns:
        Tuple of (latitude, longitude) or None if no GPS data
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            logger.debug(f"No EXIF data in {image_path}")
            return None

        # Find GPS Info tag (34853)
        gps_info = None
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                gps_info = value
                break

        if not gps_info:
            logger.debug(f"No GPS data in {image_path}")
            return None

        # Extract GPS coordinates
        gps_data = {}
        for key in gps_info.keys():
            decode = GPSTAGS.get(key, key)
            gps_data[decode] = gps_info[key]

        # Get latitude
        lat = _convert_to_degrees(gps_data.get('GPSLatitude'))
        if lat is None:
            return None

        # Check hemisphere
        if gps_data.get('GPSLatitudeRef') == 'S':
            lat = -lat

        # Get longitude
        lon = _convert_to_degrees(gps_data.get('GPSLongitude'))
        if lon is None:
            return None

        # Check hemisphere
        if gps_data.get('GPSLongitudeRef') == 'W':
            lon = -lon

        # Round to specified precision
        lat = round(lat, precision)
        lon = round(lon, precision)

        logger.debug(f"Extracted GPS: ({lat}, {lon}) from {image_path}")
        return (lat, lon)

    except Exception as e:
        logger.error(f"Error extracting GPS from {image_path}: {e}")
        return None


def _convert_to_degrees(value) -> Optional[float]:
    """
    Convert GPS coordinates from DMS (degrees, minutes, seconds) to decimal degrees.

    Args:
        value: GPS coordinate in DMS format (tuple of 3 values)

    Returns:
        Decimal degrees or None if invalid
    """
    if not value or len(value) != 3:
        return None

    try:
        degrees = float(value[0])
        minutes = float(value[1])
        seconds = float(value[2])

        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def extract_datetime_from_image(image_path: str) -> Optional[datetime]:
    """
    Extract datetime from image EXIF data.

    Args:
        image_path: Path to image file

    Returns:
        datetime object or None if no datetime found
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            return None

        # Look for DateTime tag (306)
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'DateTime':
                # Parse datetime string (format: "YYYY:MM:DD HH:MM:SS")
                try:
                    dt = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                    logger.debug(f"Extracted datetime: {dt} from {image_path}")
                    return dt
                except ValueError as e:
                    logger.warning(f"Could not parse datetime '{value}' from {image_path}: {e}")
                    return None

        logger.debug(f"No datetime in EXIF for {image_path}")
        return None

    except Exception as e:
        logger.error(f"Error extracting datetime from {image_path}: {e}")
        return None


def extract_all_exif(image_path: str) -> Dict[str, Any]:
    """
    Extract all EXIF metadata from an image.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with all EXIF data (human-readable tag names)
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            return {}

        # Convert tag IDs to human-readable names
        exif_dict = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            exif_dict[tag_name] = value

        return exif_dict

    except Exception as e:
        logger.error(f"Error extracting EXIF from {image_path}: {e}")
        return {}


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions (width, height).

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height) or None
    """
    try:
        image = Image.open(image_path)
        return image.size
    except Exception as e:
        logger.error(f"Error getting dimensions for {image_path}: {e}")
        return None


def round_datetime_to_hour(dt: datetime) -> datetime:
    """
    Round datetime to nearest hour for weather API lookups.

    Args:
        dt: datetime object

    Returns:
        datetime rounded to nearest hour
    """
    if dt.minute >= 30:
        # Round up
        return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        # Round down
        return dt.replace(minute=0, second=0, microsecond=0)


if __name__ == '__main__':
    # Test EXIF extraction
    import sys
    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Testing EXIF extraction on: {image_path}")

        # GPS
        gps = extract_gps_from_image(image_path)
        print(f"GPS: {gps}")

        # Datetime
        dt = extract_datetime_from_image(image_path)
        print(f"DateTime: {dt}")

        # All EXIF
        exif = extract_all_exif(image_path)
        print(f"\nAll EXIF tags:")
        for key, value in exif.items():
            print(f"  {key}: {value}")
    else:
        print("Usage: python exif.py <image_path>")
