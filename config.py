"""
Configuration Management Module for FRAMAI
Handles loading, validation, and environment variable overrides for YAML config
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Default config file search paths (in priority order)
DEFAULT_CONFIG_PATHS = [
    'config.yaml',                        # Current directory
    '~/.config/framai/config.yaml',      # User config
    '/etc/framai/config.yaml'            # System config
]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Explicit path to config file. If None, search default locations.

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If explicit config_path provided but doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    # Load environment variables from ~/.env if it exists
    env_file = Path.home() / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        logger.debug(f"Loaded environment variables from {env_file}")

    config = None
    config_source = None

    # If explicit path provided, use only that
    if config_path:
        config_file = Path(config_path).expanduser()
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        config_source = str(config_file)
        logger.info(f"Loaded config from: {config_source}")

    # Otherwise search default locations
    else:
        for path_str in DEFAULT_CONFIG_PATHS:
            config_file = Path(path_str).expanduser()
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    config_source = str(config_file)
                    logger.info(f"Loaded config from: {config_source}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {config_file}: {e}")
                    continue

    # Fallback to default config if no file found
    if config is None:
        logger.warning("No config file found, using defaults")
        config = get_default_config()
        config_source = "defaults"

    # Apply environment variable overrides
    config = apply_env_overrides(config)

    # Apply API key file fallbacks
    config = apply_api_key_files(config)

    # Validate configuration
    validate_config(config)

    # Store metadata
    config['_meta'] = {
        'source': config_source,
        'loaded_at': None  # Could add timestamp
    }

    return config


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables take precedence over config file values.
    Supported variables:
    - OPENAI_API_KEY: Override ai_models.openai.api_key
    - VISUAL_CROSSING_API_KEY: Override weather_api.visual_crossing.api_key
    - FRAMAI_LOG_LEVEL: Override logging.log_level
    """
    # OpenAI API key
    if 'OPENAI_API_KEY' in os.environ:
        if 'ai_models' not in config:
            config['ai_models'] = {}
        if 'openai' not in config['ai_models']:
            config['ai_models']['openai'] = {}
        config['ai_models']['openai']['api_key'] = os.environ['OPENAI_API_KEY']
        logger.debug("Using OPENAI_API_KEY from environment")

    # Visual Crossing API key (legacy)
    if 'VISUAL_CROSSING_API_KEY' in os.environ:
        if 'weather_api' not in config:
            config['weather_api'] = {}
        if 'visual_crossing' not in config['weather_api']:
            config['weather_api']['visual_crossing'] = {}
        config['weather_api']['visual_crossing']['api_key'] = os.environ['VISUAL_CROSSING_API_KEY']
        logger.debug("Using VISUAL_CROSSING_API_KEY from environment")

    # Log level
    if 'FRAMAI_LOG_LEVEL' in os.environ:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['log_level'] = os.environ['FRAMAI_LOG_LEVEL']
        logger.debug(f"Using log level from environment: {os.environ['FRAMAI_LOG_LEVEL']}")

    return config


def apply_api_key_files(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read API keys from files as fallback if not set via env or config.

    Maintains backward compatibility with existing key_openai.txt pattern.
    """
    # OpenAI API key from file
    if config.get('ai_models', {}).get('openai', {}).get('api_key') in [None, '${OPENAI_API_KEY}', '']:
        key_file = config.get('ai_models', {}).get('openai', {}).get('api_key_file', 'key_openai.txt')
        key_path = Path(key_file)
        if key_path.exists():
            try:
                with open(key_path, 'r') as f:
                    api_key = f.read().strip()
                if api_key:
                    config['ai_models']['openai']['api_key'] = api_key
                    logger.debug(f"Loaded OpenAI API key from {key_file}")
            except Exception as e:
                logger.warning(f"Failed to read OpenAI API key from {key_file}: {e}")

    # Visual Crossing API key from file (legacy)
    if config.get('weather_api', {}).get('visual_crossing', {}).get('enabled'):
        if not config['weather_api']['visual_crossing'].get('api_key'):
            key_file = config['weather_api']['visual_crossing'].get('api_key_file', 'key_visualcrossing.txt')
            key_path = Path(key_file)
            if key_path.exists():
                try:
                    with open(key_path, 'r') as f:
                        api_key = f.read().strip()
                    if api_key:
                        config['weather_api']['visual_crossing']['api_key'] = api_key
                        logger.debug(f"Loaded Visual Crossing API key from {key_file}")
                except Exception as e:
                    logger.warning(f"Failed to read Visual Crossing API key from {key_file}: {e}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_sections = ['ai_models', 'weather_api', 'file_paths', 'processing']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate AI models configuration
    if 'openai' not in config['ai_models']:
        logger.warning("OpenAI configuration missing - GPT features will be unavailable")

    if 'whisper' not in config['ai_models']:
        logger.warning("Whisper configuration missing - transcription may use defaults")

    # Validate weather API
    if config.get('features', {}).get('weather_integration', True):
        if 'weather_api' not in config or 'provider' not in config['weather_api']:
            logger.warning("Weather API not configured - weather features will be disabled")

    # Validate file paths
    if 'output_json' not in config.get('file_paths', {}):
        logger.warning("No output JSON filename specified, using default")
        config['file_paths']['output_json'] = 'fram.json'


def get_default_config() -> Dict[str, Any]:
    """
    Return minimal default configuration.

    Used as fallback when no config file is found.
    """
    return {
        'ai_models': {
            'openai': {
                'api_key': None,
                'api_key_file': 'key_openai.txt',
                'gpt_model': 'gpt-4',
                'temperature': 0.1,
                'whisper_api_model': 'whisper-1'
            },
            'whisper': {
                'model_name': 'base.en',
                'model_path': './models/',
                'word_timestamps': True,
                'fp16': False
            }
        },
        'weather_api': {
            'provider': 'open-meteo',
            'endpoint': 'https://archive-api.open-meteo.com/v1/archive',
            'cache_dir': '.weather_cache',
            'variables': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'wind_speed_10m',
                'weather_code',
                'cloud_cover',
                'pressure_msl'
            ],
            'timezone': 'auto'
        },
        'geocoding_api': {
            'provider': 'openstreetmap',
            'endpoint': 'https://nominatim.openstreetmap.org/reverse',
            'zoom_level': 18,
            'rate_limit_delay': 1.1,
            'user_agent': 'framai-cli/1.0'
        },
        'file_paths': {
            'output_json': 'fram.json',
            'temp_audio_wav': 'temp.wav',
            'temp_audio_mp3': 'temp.mp3',
            'output_suffix': '_POST',
            'supported_formats': {
                'audio': ['wav', 'mp3', 'flac'],
                'image': ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            }
        },
        'processing': {
            'audio': {
                'default_duration': 30,
                'split_threshold_seconds': 60,
                'fade_duration': 30,
                'fade_length_multiplier': 3
            },
            'image': {
                'location_threshold_meters': 1000,
                'gps_precision': 6
            }
        },
        'formatting': {
            'datetime': {
                'time_format': '%H:%M',
                'exif_format': '%Y:%m:%d %H:%M:%S',
                'iso_format': '%Y-%m-%dT%H:%M:%S',
                'weather_date_format': '%Y-%m-%d'
            },
            'coordinates': {
                'decimal_places': 6,
                'format': 'decimal'
            }
        },
        'metadata': {
            'exif_tags': {
                'datetime': 306,
                'gps_info': 34853
            },
            'earth_radius_km': 6371.0
        },
        'logging': {
            'verbose_default': False,
            'log_level': 'INFO'
        },
        'features': {
            'weather_integration': True,
            'geocoding': True,
            'gpt_refinement': True,
            'dry_run_default': False
        }
    }


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'ai_models.openai.gpt_model')
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> config = load_config()
        >>> model = get_config_value(config, 'ai_models.openai.gpt_model', 'gpt-3.5-turbo')
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the loaded configuration.
    Useful for debugging and verification.
    """
    print("=" * 60)
    print("FRAMAI Configuration Summary")
    print("=" * 60)

    if '_meta' in config and 'source' in config['_meta']:
        print(f"Source: {config['_meta']['source']}")

    print(f"\nAI Models:")
    print(f"  OpenAI GPT: {config.get('ai_models', {}).get('openai', {}).get('gpt_model', 'N/A')}")
    print(f"  Whisper: {config.get('ai_models', {}).get('whisper', {}).get('model_name', 'N/A')}")

    print(f"\nWeather API:")
    print(f"  Provider: {config.get('weather_api', {}).get('provider', 'N/A')}")

    print(f"\nGeocoding API:")
    print(f"  Provider: {config.get('geocoding_api', {}).get('provider', 'N/A')}")

    print(f"\nFile Paths:")
    print(f"  Output JSON: {config.get('file_paths', {}).get('output_json', 'N/A')}")

    print(f"\nFeatures:")
    features = config.get('features', {})
    print(f"  Weather Integration: {features.get('weather_integration', 'N/A')}")
    print(f"  Geocoding: {features.get('geocoding', 'N/A')}")
    print(f"  GPT Refinement: {features.get('gpt_refinement', 'N/A')}")

    print("=" * 60)


# Convenience function for backward compatibility
def load_api_key(key_file: str = 'key_openai.txt') -> Optional[str]:
    """
    Load API key from file (legacy function for backward compatibility).

    Args:
        key_file: Path to API key file

    Returns:
        API key string or None if file not found
    """
    key_path = Path(key_file)
    if key_path.exists():
        try:
            with open(key_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read API key from {key_file}: {e}")
    return None


if __name__ == '__main__':
    # Test configuration loading
    logging.basicConfig(level=logging.DEBUG)

    try:
        config = load_config()
        print_config_summary(config)
    except Exception as e:
        print(f"Error loading config: {e}")
