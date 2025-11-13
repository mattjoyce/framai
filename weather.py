"""
Weather Data Integration Module for FRAMAI
Uses Open-Meteo API for historical weather data retrieval
Adapted from biophony-ai weather_integration.py patterns
"""

import openmeteo_requests
import requests_cache
from retry_requests import retry
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class WeatherClient:
    """
    Open-Meteo weather API client with caching and retry logic.

    Features:
    - Automatic request caching to reduce API calls
    - Retry logic with exponential backoff
    - Hourly historical weather data
    - Daily sunrise/sunset times
    - No API key required (free service)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize weather client with configuration.

        Args:
            config: Configuration dictionary with weather_api section
        """
        self.config = config
        weather_config = config.get('weather_api', {})

        # Setup cache directory
        cache_dir = Path(weather_config.get('cache_dir', '.weather_cache'))
        cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir

        # Setup cached session with retry logic (from biophony-ai pattern)
        cache_session = requests_cache.CachedSession(
            str(cache_dir / 'weather_cache'),
            expire_after=-1  # Never expire cache (historical data doesn't change)
        )
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

        # API configuration
        self.endpoint = weather_config.get('endpoint', 'https://archive-api.open-meteo.com/v1/archive')
        self.variables = weather_config.get('variables', [
            'temperature_2m',
            'relative_humidity_2m',
            'precipitation',
            'wind_speed_10m',
            'weather_code',
            'cloud_cover',
            'pressure_msl'
        ])
        self.daily_variables = weather_config.get('daily_variables', ['sunrise', 'sunset'])
        self.timezone = weather_config.get('timezone', 'auto')

        logger.info(f"Weather client initialized with cache dir: {cache_dir}")
        logger.debug(f"Weather variables: {', '.join(self.variables)}")

    def fetch_weather(self, latitude: float, longitude: float, date: str) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data for a specific location and date.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            date: Date string in YYYY-MM-DD format

        Returns:
            Dictionary with weather data or None if fetch fails

        Example return:
            {
                'temperature_2m': 15.5,
                'relative_humidity_2m': 75,
                'precipitation': 0.0,
                'wind_speed_10m': 8.5,
                'weather_code': 2,
                'cloud_cover': 50,
                'pressure_msl': 1013.2,
                'sunrise': '2025-06-20T06:30:00',
                'sunset': '2025-06-20T20:15:00',
                'datetime': '2025-06-20T14:00:00'
            }
        """
        try:
            # Parse date to ensure valid format
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y-%m-%d')

            logger.debug(f"Fetching weather for ({latitude:.4f}, {longitude:.4f}) on {date_str}")

            # Prepare API parameters
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": ",".join(self.variables),
                "daily": ",".join(self.daily_variables),
                "timezone": self.timezone
            }

            # Make API request using Open-Meteo SDK
            responses = self.openmeteo.weather_api(self.endpoint, params=params)
            response = responses[0]

            # Also make raw API call to get daily data (sunrise/sunset)
            raw_response = requests.get(self.endpoint, params=params)
            raw_data = raw_response.json() if raw_response.status_code == 200 else None

            # Extract hourly data
            hourly = response.Hourly()
            time_data = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )

            # Build weather records for each hour
            weather_records = []
            for i, dt in enumerate(time_data):
                record = {
                    'datetime': dt.isoformat(),
                    'timestamp': int(dt.timestamp())
                }

                # Extract each weather variable
                for j, var in enumerate(self.variables):
                    value = hourly.Variables(j).ValuesAsNumpy()[i]
                    record[var] = float(value) if not pd.isna(value) else None

                weather_records.append(record)

            # Add daily data (sunrise/sunset) if available
            daily_data = None
            if raw_data and 'daily' in raw_data:
                daily = raw_data['daily']
                if 'sunrise' in daily and 'sunset' in daily:
                    daily_data = {
                        'sunrise': daily['sunrise'][0] if daily['sunrise'] else None,
                        'sunset': daily['sunset'][0] if daily['sunset'] else None
                    }

            logger.info(f"Fetched {len(weather_records)} hourly weather records for {date_str}")

            return {
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'date': date_str,
                'hourly': weather_records,
                'daily': daily_data,
                'source': 'open-meteo',
                'cached': hasattr(raw_response, 'from_cache') and raw_response.from_cache
            }

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def fetch_weather_for_datetime(self, latitude: float, longitude: float,
                                   dt: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data for a specific datetime and return the matching hour.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            dt: Datetime object

        Returns:
            Dictionary with weather data for that specific hour, or None
        """
        date_str = dt.strftime('%Y-%m-%d')
        weather_data = self.fetch_weather(latitude, longitude, date_str)

        if not weather_data or 'hourly' not in weather_data:
            return None

        # Find the weather record for the closest hour
        target_hour = dt.replace(minute=0, second=0, microsecond=0)
        target_timestamp = int(target_hour.timestamp())

        # Find closest hourly record
        closest_record = None
        min_diff = float('inf')

        for record in weather_data['hourly']:
            time_diff = abs(record['timestamp'] - target_timestamp)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_record = record

        if closest_record:
            # Add daily data if available
            if weather_data.get('daily'):
                closest_record['sunrise'] = weather_data['daily'].get('sunrise')
                closest_record['sunset'] = weather_data['daily'].get('sunset')

            closest_record['location'] = weather_data['location']
            closest_record['source'] = weather_data['source']
            closest_record['cached'] = weather_data.get('cached', False)

            logger.debug(f"Matched weather for {dt} (diff: {min_diff}s)")
            return closest_record

        return None

    def fetch_weather_range(self, latitude: float, longitude: float,
                           start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data for a date range.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary with weather data for the entire range
        """
        try:
            logger.info(f"Fetching weather range: {start_date} to {end_date}")

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(self.variables),
                "daily": ",".join(self.daily_variables),
                "timezone": self.timezone
            }

            # Make API request
            responses = self.openmeteo.weather_api(self.endpoint, params=params)
            response = responses[0]

            # Get raw data for daily variables
            raw_response = requests.get(self.endpoint, params=params)
            raw_data = raw_response.json() if raw_response.status_code == 200 else None

            # Extract hourly data
            hourly = response.Hourly()
            time_data = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )

            # Build weather records
            weather_records = []
            for i, dt in enumerate(time_data):
                record = {
                    'datetime': dt.isoformat(),
                    'timestamp': int(dt.timestamp()),
                    'date': dt.strftime('%Y-%m-%d'),
                    'hour': dt.hour
                }

                for j, var in enumerate(self.variables):
                    value = hourly.Variables(j).ValuesAsNumpy()[i]
                    record[var] = float(value) if not pd.isna(value) else None

                weather_records.append(record)

            # Process daily data
            daily_records = []
            if raw_data and 'daily' in raw_data:
                daily = raw_data['daily']
                if 'time' in daily:
                    for i, date_str in enumerate(daily['time']):
                        daily_record = {'date': date_str}
                        for var in self.daily_variables:
                            if var in daily and i < len(daily[var]):
                                daily_record[var] = daily[var][i]
                        daily_records.append(daily_record)

            logger.info(f"Fetched {len(weather_records)} hourly records across {len(daily_records)} days")

            return {
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'start_date': start_date,
                'end_date': end_date,
                'hourly': weather_records,
                'daily': daily_records,
                'source': 'open-meteo'
            }

        except Exception as e:
            logger.error(f"Error fetching weather range: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the weather cache."""
        try:
            cache_file = self.cache_dir / 'weather_cache.sqlite'
            if cache_file.exists():
                cache_file.unlink()
                logger.info("Weather cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the weather cache."""
        cache_file = self.cache_dir / 'weather_cache.sqlite'
        if not cache_file.exists():
            return {'exists': False, 'size_bytes': 0}

        return {
            'exists': True,
            'size_bytes': cache_file.stat().st_size,
            'size_mb': cache_file.stat().st_size / (1024 * 1024),
            'path': str(cache_file)
        }


def format_weather_summary(weather_data: Dict[str, Any]) -> str:
    """
    Format weather data as a human-readable summary string.

    Args:
        weather_data: Weather data dictionary from fetch_weather_for_datetime

    Returns:
        Formatted string describing weather conditions
    """
    if not weather_data:
        return "Weather data unavailable"

    parts = []

    # Temperature
    if 'temperature_2m' in weather_data and weather_data['temperature_2m'] is not None:
        parts.append(f"{weather_data['temperature_2m']:.1f}°C")

    # Humidity
    if 'relative_humidity_2m' in weather_data and weather_data['relative_humidity_2m'] is not None:
        parts.append(f"{weather_data['relative_humidity_2m']:.0f}% humidity")

    # Wind
    if 'wind_speed_10m' in weather_data and weather_data['wind_speed_10m'] is not None:
        parts.append(f"{weather_data['wind_speed_10m']:.1f} km/h wind")

    # Precipitation
    if 'precipitation' in weather_data and weather_data['precipitation'] is not None:
        if weather_data['precipitation'] > 0:
            parts.append(f"{weather_data['precipitation']:.1f}mm rain")

    # Weather code (WMO code)
    if 'weather_code' in weather_data and weather_data['weather_code'] is not None:
        weather_desc = get_weather_description(int(weather_data['weather_code']))
        if weather_desc:
            parts.append(weather_desc)

    return ", ".join(parts) if parts else "No weather data"


def get_weather_description(wmo_code: int) -> str:
    """
    Convert WMO weather code to human-readable description.

    WMO codes: https://open-meteo.com/en/docs
    """
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    return weather_codes.get(wmo_code, f"Weather code {wmo_code}")


if __name__ == '__main__':
    # Test weather client
    import sys
    logging.basicConfig(level=logging.INFO)

    # Load config
    from config import load_config
    config = load_config()

    # Create client
    client = WeatherClient(config)

    # Test fetch
    print("\nTesting weather fetch for Sydney on 2025-06-20:")
    weather = client.fetch_weather(-33.8688, 151.2093, '2025-06-20')

    if weather:
        print(f"Location: {weather['location']}")
        print(f"Date: {weather['date']}")
        print(f"Hourly records: {len(weather['hourly'])}")
        print(f"Daily data: {weather.get('daily', 'N/A')}")
        print(f"Cached: {weather.get('cached', False)}")

        # Show first hourly record
        if weather['hourly']:
            print("\nFirst hourly record:")
            for key, value in weather['hourly'][0].items():
                print(f"  {key}: {value}")

        # Show summary
        if weather['hourly']:
            summary = format_weather_summary(weather['hourly'][0])
            print(f"\nSummary: {summary}")
    else:
        print("Failed to fetch weather data")

    # Show cache stats
    print("\nCache stats:")
    stats = client.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
