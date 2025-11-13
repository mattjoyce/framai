"""
Geocoding utilities using OpenStreetMap Nominatim API
Respects OSM usage policy with rate limiting
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class GeocodingClient:
    """
    OpenStreetMap Nominatim geocoding client with rate limiting.

    Features:
    - Reverse geocoding (coordinates -> location name)
    - Rate limiting to respect OSM usage policy
    - Detailed address information
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize geocoding client with configuration.

        Args:
            config: Configuration dictionary with geocoding_api section
        """
        self.config = config
        geocoding_config = config.get('geocoding_api', {})

        self.endpoint = geocoding_config.get('endpoint', 'https://nominatim.openstreetmap.org/reverse')
        self.format = geocoding_config.get('format', 'jsonv2')
        self.zoom_level = geocoding_config.get('zoom_level', 18)
        self.address_details = geocoding_config.get('address_details', True)
        self.rate_limit_delay = geocoding_config.get('rate_limit_delay', 1.1)
        self.user_agent = geocoding_config.get('user_agent', 'framai-cli/1.0')

        self.last_request_time = 0

        logger.info(f"Geocoding client initialized (rate limit: {self.rate_limit_delay}s)")

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Convert coordinates to location name and address.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            Dictionary with location data or None if request fails

        Example return:
            {
                'name': 'Sydney Opera House',
                'display_name': 'Sydney Opera House, Sydney, NSW, Australia',
                'address': {
                    'building': 'Sydney Opera House',
                    'road': 'Bennelong Point',
                    'suburb': 'Sydney',
                    'city': 'Sydney',
                    'state': 'New South Wales',
                    'country': 'Australia',
                    'country_code': 'au',
                    'postcode': '2000'
                },
                'lat': '-33.8568',
                'lon': '151.2153'
            }
        """
        # Rate limiting
        self._wait_for_rate_limit()

        try:
            params = {
                'format': self.format,
                'lat': latitude,
                'lon': longitude,
                'zoom': self.zoom_level,
                'addressdetails': 1 if self.address_details else 0
            }

            headers = {
                'User-Agent': self.user_agent
            }

            logger.debug(f"Reverse geocoding: ({latitude}, {longitude})")

            response = requests.get(self.endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            logger.debug(f"Location: {data.get('display_name', 'Unknown')}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Geocoding request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during reverse geocoding: {e}")
            return None

    def get_location_name(self, latitude: float, longitude: float) -> Optional[str]:
        """
        Get a simple location name for coordinates.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            Location name string or None
        """
        data = self.reverse_geocode(latitude, longitude)

        if not data:
            return None

        # Try to build a meaningful location name
        address = data.get('address', {})

        # Priority order for location name components
        name_parts = []

        # Primary location identifiers
        if 'suburb' in address:
            name_parts.append(address['suburb'])
        elif 'village' in address:
            name_parts.append(address['village'])
        elif 'hamlet' in address:
            name_parts.append(address['hamlet'])
        elif 'town' in address:
            name_parts.append(address['town'])

        # City/region
        if 'city' in address:
            if not name_parts or address['city'] != name_parts[0]:
                name_parts.append(address['city'])
        elif 'county' in address:
            name_parts.append(address['county'])

        # State/Province
        if 'state' in address:
            name_parts.append(address['state'])

        # Country
        if 'country' in address:
            name_parts.append(address['country'])

        if name_parts:
            return ', '.join(name_parts)

        # Fallback to display_name
        return data.get('display_name', None)

    def _wait_for_rate_limit(self):
        """
        Enforce rate limiting by waiting if necessary.

        Respects OSM Nominatim usage policy (max 1 request per second).
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        self.last_request_time = time.time()


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float,
                       earth_radius_km: float = 6371.0) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.

    Args:
        lat1: Latitude of first point (decimal degrees)
        lon1: Longitude of first point (decimal degrees)
        lat2: Latitude of second point (decimal degrees)
        lon2: Longitude of second point (decimal degrees)
        earth_radius_km: Radius of Earth in kilometers (default: 6371.0)

    Returns:
        Distance in kilometers
    """
    import math

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    distance = earth_radius_km * c

    return distance


def cluster_locations(locations: list, threshold_meters: float = 1000) -> list:
    """
    Group nearby GPS locations into clusters.

    Args:
        locations: List of (lat, lon) tuples
        threshold_meters: Maximum distance in meters to group locations

    Returns:
        List of representative locations (cluster centroids)
    """
    if not locations:
        return []

    threshold_km = threshold_meters / 1000.0
    clusters = []

    for lat, lon in locations:
        # Find existing cluster within threshold
        found_cluster = False
        for cluster in clusters:
            cluster_lat, cluster_lon, count = cluster
            distance = calculate_distance(lat, lon, cluster_lat, cluster_lon)

            if distance <= threshold_km:
                # Add to existing cluster (update centroid)
                new_count = count + 1
                new_lat = (cluster_lat * count + lat) / new_count
                new_lon = (cluster_lon * count + lon) / new_count
                cluster[0] = new_lat
                cluster[1] = new_lon
                cluster[2] = new_count
                found_cluster = True
                break

        if not found_cluster:
            # Create new cluster
            clusters.append([lat, lon, 1])

    # Return just the coordinates (without count)
    return [(cluster[0], cluster[1]) for cluster in clusters]


if __name__ == '__main__':
    # Test geocoding client
    import sys
    logging.basicConfig(level=logging.DEBUG)

    # Load config
    from config import load_config
    config = load_config()

    # Create client
    client = GeocodingClient(config)

    # Test reverse geocoding
    if len(sys.argv) > 2:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    else:
        # Default: Sydney Opera House
        lat, lon = -33.8568, 151.2153

    print(f"\nReverse geocoding: ({lat}, {lon})")

    # Get full data
    data = client.reverse_geocode(lat, lon)
    if data:
        print(f"Display name: {data.get('display_name')}")
        print("\nAddress:")
        for key, value in data.get('address', {}).items():
            print(f"  {key}: {value}")

    # Get simple name
    name = client.get_location_name(lat, lon)
    print(f"\nSimple name: {name}")

    # Test distance calculation
    print("\nTesting distance calculation:")
    # Sydney to Melbourne
    sydney = (-33.8688, 151.2093)
    melbourne = (-37.8136, 144.9631)
    distance = calculate_distance(*sydney, *melbourne)
    print(f"Sydney to Melbourne: {distance:.1f} km")

    # Test clustering
    print(f"\nTesting location clustering:")
    test_locations = [
        (-33.8568, 151.2153),  # Sydney Opera House
        (-33.8570, 151.2155),  # Very close to Opera House
        (-37.8136, 144.9631),  # Melbourne (far away)
    ]
    clusters = cluster_locations(test_locations, threshold_meters=1000)
    print(f"Input locations: {len(test_locations)}")
    print(f"Clustered to: {len(clusters)} locations")
    for i, (lat, lon) in enumerate(clusters, 1):
        print(f"  Cluster {i}: ({lat:.4f}, {lon:.4f})")
