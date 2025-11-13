"""
Images command implementation
Processes images: extracts GPS/EXIF, fetches weather & location data
Refactored from images.py and fram-image.py to use new infrastructure
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import utilities
from utils.exif import extract_gps_from_image, extract_datetime_from_image
from utils.geocoding import GeocodingClient, cluster_locations, calculate_distance
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
from weather import WeatherClient

logger = logging.getLogger(__name__)


def run_images_command(directory: str, config: Dict[str, Any],
                      options: Dict[str, Any], dry_run: bool = False,
                      verbose: bool = False) -> bool:
    """
    Main entry point for images command.

    Args:
        directory: Directory containing image files
        config: Configuration dictionary
        options: Command options (output_file, weather_enabled, etc.)
        dry_run: If True, show what would be done without executing
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    print_header("FRAMAI - Image Processing")

    # Extract options
    output_file = options['output_file']
    weather_enabled = options['weather_enabled']
    geocode_enabled = options['geocode_enabled']
    threshold_meters = options['threshold_meters']
    extensions = options['extensions']

    # Show configuration
    print_section("Configuration")
    print(f"  Directory: {directory}")
    print(f"  Output file: {output_file}")
    print(f"  Weather data: {'enabled' if weather_enabled else 'disabled'}")
    print(f"  Geocoding: {'enabled' if geocode_enabled else 'disabled'}")
    print(f"  Location threshold: {threshold_meters}m")
    print(f"  Extensions: {', '.join(extensions)}")

    # Find image files
    print_section("Scanning for images")
    image_files = find_image_files(directory, extensions)

    if not image_files:
        print_warning(f"No image files found in {directory}")
        return False

    print_success(f"Found {len(image_files)} image files")

    # Dry-run mode
    if dry_run:
        actions = [
            f"Process {len(image_files)} image files",
            "Extract GPS coordinates and timestamps from EXIF data"
        ]

        if weather_enabled:
            actions.append("Fetch weather data from Open-Meteo API")

        if geocode_enabled:
            actions.append("Fetch location names from OpenStreetMap Nominatim")

        actions.append(f"Save results to {output_file}")

        print_dry_run_summary("Image Processing Plan", actions)
        return True

    # Process images
    print_section("Processing images")
    results = process_images(image_files, config, weather_enabled, geocode_enabled,
                           threshold_meters, verbose)

    if not results:
        print_error("Failed to process images")
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


def find_image_files(directory: str, extensions: List[str]) -> List[Path]:
    """
    Find all image files in directory with specified extensions.

    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.jpg', '.png'])

    Returns:
        List of Path objects for found images
    """
    dir_path = Path(directory)
    image_files = []

    # Normalize extensions (ensure they start with '.')
    normalized_extensions = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_extensions.append(ext.lower())

    # Search for files
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in normalized_extensions:
                image_files.append(file_path)

    return sorted(image_files)


def process_images(image_files: List[Path], config: Dict[str, Any],
                  weather_enabled: bool, geocode_enabled: bool,
                  threshold_meters: int, verbose: bool) -> Optional[Dict[str, Any]]:
    """
    Process all image files and extract metadata.

    Args:
        image_files: List of image file paths
        config: Configuration dictionary
        weather_enabled: Fetch weather data
        geocode_enabled: Fetch location names
        threshold_meters: Location clustering threshold
        verbose: Verbose output

    Returns:
        Dictionary with processed results or None if failed
    """
    # Initialize clients
    weather_client = WeatherClient(config) if weather_enabled else None
    geocoding_client = GeocodingClient(config) if geocode_enabled else None

    # Extract metadata from all images
    image_metadata = []

    with create_progress_bar("Extracting EXIF data", total=len(image_files)) as progress:
        task = progress.add_task("Processing images", total=len(image_files), status="Starting...")

        for i, image_path in enumerate(image_files, 1):
            try:
                # Extract GPS and datetime
                gps = extract_gps_from_image(str(image_path))
                dt = extract_datetime_from_image(str(image_path))

                if gps or dt:
                    metadata = {
                        'filename': image_path.name,
                        'filepath': str(image_path),
                        'gps': {'latitude': gps[0], 'longitude': gps[1]} if gps else None,
                        'datetime': dt.isoformat() if dt else None
                    }
                    image_metadata.append(metadata)
                    progress.update(task, status=f"✓ {image_path.name[:30]}")
                else:
                    progress.update(task, status=f"⚠️  No EXIF: {image_path.name[:30]}")
                    logger.warning(f"No GPS or datetime in {image_path.name}")

                progress.advance(task)

            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                progress.update(task, status=f"❌ Error: {image_path.name[:30]}")
                progress.advance(task)

    if not image_metadata:
        print_warning("No metadata extracted from images")
        return None

    print_info(f"Extracted metadata from {len(image_metadata)} images")

    # Cluster locations
    locations = cluster_and_enrich_locations(
        image_metadata, config, weather_client, geocoding_client,
        threshold_meters, verbose
    )

    # Build results
    results = {
        'images': image_metadata,
        'locations': locations,
        'summary': {
            'total_images': len(image_files),
            'images_with_metadata': len(image_metadata),
            'unique_locations': len(locations),
            'weather_data_fetched': weather_enabled,
            'geocoding_performed': geocode_enabled
        }
    }

    return results


def cluster_and_enrich_locations(image_metadata: List[Dict[str, Any]],
                                 config: Dict[str, Any],
                                 weather_client: Optional[WeatherClient],
                                 geocoding_client: Optional[GeocodingClient],
                                 threshold_meters: int,
                                 verbose: bool) -> List[Dict[str, Any]]:
    """
    Cluster nearby locations and enrich with weather and geocoding data.

    Args:
        image_metadata: List of image metadata dictionaries
        config: Configuration dictionary
        weather_client: WeatherClient instance (or None)
        geocoding_client: GeocodingClient instance (or None)
        threshold_meters: Clustering threshold in meters
        verbose: Verbose output

    Returns:
        List of location dictionaries with enriched data
    """
    # Extract GPS coordinates
    gps_coords = []
    for img in image_metadata:
        if img['gps']:
            gps_coords.append((img['gps']['latitude'], img['gps']['longitude']))

    if not gps_coords:
        return []

    # Cluster locations
    print_info(f"Clustering {len(gps_coords)} GPS points (threshold: {threshold_meters}m)")
    clustered_coords = cluster_locations(gps_coords, threshold_meters)
    print_info(f"Clustered to {len(clustered_coords)} unique locations")

    # Enrich each clustered location
    locations = []

    with create_progress_bar("Enriching locations", total=len(clustered_coords)) as progress:
        task = progress.add_task("Processing", total=len(clustered_coords), status="Starting...")

        for lat, lon in clustered_coords:
            location = {
                'latitude': round(lat, 6),
                'longitude': round(lon, 6)
            }

            # Find representative datetime for this location
            representative_dt = find_representative_datetime(lat, lon, image_metadata)

            if representative_dt:
                location['datetime'] = representative_dt

                # Fetch weather data
                if weather_client and representative_dt:
                    try:
                        dt_obj = datetime.fromisoformat(representative_dt)
                        weather_data = weather_client.fetch_weather_for_datetime(lat, lon, dt_obj)

                        if weather_data:
                            location['weather'] = weather_data
                            progress.update(task, status="✓ Weather fetched")
                    except Exception as e:
                        logger.error(f"Error fetching weather for ({lat}, {lon}): {e}")

            # Fetch location name
            if geocoding_client:
                try:
                    location_name = geocoding_client.get_location_name(lat, lon)
                    if location_name:
                        location['location_name'] = location_name
                        progress.update(task, status=f"✓ {location_name[:30]}")
                except Exception as e:
                    logger.error(f"Error geocoding ({lat}, {lon}): {e}")

            locations.append(location)
            progress.advance(task)

    return locations


def find_representative_datetime(lat: float, lon: float,
                                image_metadata: List[Dict[str, Any]]) -> Optional[str]:
    """
    Find a representative datetime for a location from nearby images.

    Args:
        lat: Latitude
        lon: Longitude
        image_metadata: List of image metadata

    Returns:
        ISO datetime string or None
    """
    # Find images near this location
    nearby_datetimes = []

    for img in image_metadata:
        if img['gps'] and img['datetime']:
            img_lat = img['gps']['latitude']
            img_lon = img['gps']['longitude']

            # Check if within 100m (close enough to be same location)
            distance = calculate_distance(lat, lon, img_lat, img_lon) * 1000  # Convert to meters

            if distance < 100:
                nearby_datetimes.append(img['datetime'])

    if nearby_datetimes:
        # Return the earliest datetime
        return min(nearby_datetimes)

    return None


def display_summary(results: Dict[str, Any]) -> None:
    """
    Display a summary table of processing results.

    Args:
        results: Results dictionary
    """
    print_section("Processing Summary")

    summary_data = [
        {'Metric': 'Total images', 'Value': results['summary']['total_images']},
        {'Metric': 'Images with metadata', 'Value': results['summary']['images_with_metadata']},
        {'Metric': 'Unique locations', 'Value': results['summary']['unique_locations']},
    ]

    print_summary_table("Results", summary_data)

    # Show locations
    if results['locations']:
        print_section("Locations")
        location_data = []

        for loc in results['locations'][:10]:  # Show first 10
            row = {
                'Latitude': f"{loc['latitude']:.4f}",
                'Longitude': f"{loc['longitude']:.4f}",
            }

            if 'location_name' in loc:
                row['Location'] = loc['location_name'][:40]

            if 'weather' in loc:
                temp = loc['weather'].get('temperature_2m', 'N/A')
                row['Temperature'] = f"{temp}°C" if temp != 'N/A' else 'N/A'

            location_data.append(row)

        if location_data:
            print_summary_table("Locations (first 10)", location_data)


def save_results(results: Dict[str, Any], output_file: str, directory: str) -> bool:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        output_file: Output file path
        directory: Base directory (for relative path resolution)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve output path (relative to directory if not absolute)
        output_path = Path(output_file)
        if not output_path.is_absolute() and not str(output_file).startswith(directory):
            output_path = Path(directory) / output_file

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


if __name__ == '__main__':
    # Test images command
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        test_dir = sys.argv[1]

        # Load config
        from config import load_config
        config = load_config()

        # Test options
        options = {
            'output_file': 'test_images.json',
            'weather_enabled': True,
            'geocode_enabled': True,
            'threshold_meters': 1000,
            'extensions': ['.jpg', '.jpeg', '.png']
        }

        # Run command
        success = run_images_command(
            directory=test_dir,
            config=config,
            options=options,
            dry_run=False,
            verbose=True
        )

        sys.exit(0 if success else 1)
    else:
        print("Usage: python images_cmd.py <directory>")
