# FRAMAI - Field Recording Audio/Media Analysis & Integration

A comprehensive CLI tool for processing field recordings with GPS, weather data, transcription, and audio post-processing.

## ✨ Features

- **Image Processing**: Extract GPS coordinates and timestamps from EXIF data
- **Weather Integration**: Fetch historical weather data from Open-Meteo (no API key required!)
- **Geocoding**: Convert coordinates to location names using OpenStreetMap Nominatim
- **Audio Transcription**: Transcribe audio using OpenAI Whisper
- **GPT Refinement**: Refine transcriptions with GPT-4
- **Audio Post-Processing**: Trim and fade audio files

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process images
fram-cli images ./recordings/

# Show help
fram-cli --help
```

## 📖 Full Documentation

See [USAGE.md](USAGE.md) for detailed documentation.

## 🎯 Example Workflow

```bash
# 1. Extract GPS & weather from images
fram-cli images ./field_trip/

# 2. Transcribe audio
fram-cli transcribe ./field_trip/

# 3. Refine with GPT-4
fram-cli refine fram.json

# 4. Post-process audio
fram-cli postprocess ./field_trip/
```

## 📊 Sample Output

The tool tested successfully on TestData/:
- **Image**: PXL_20230723_032711335.jpg (Google Pixel 6a)
- **Location**: Croydon Park, Sydney (-33.896°, 151.092°)
- **Weather**: 12°C, 83% humidity, light drizzle
- **Sunrise/Sunset**: 07:55 / 18:09

## 🔧 Configuration

Edit `config.yaml` or use `--config` flag. Set `OPENAI_API_KEY` environment variable for GPT features.

## 📦 Installation

```bash
pip install -e .
```

This installs `fram-cli` command globally.

## 🌟 Key Improvements (Phase 1)

- ✅ Switched from Visual Crossing to Open-Meteo (free, no API key)
- ✅ Unified YAML configuration system
- ✅ Professional Click CLI with Rich output
- ✅ Modular architecture with utilities
- ✅ Comprehensive error handling
- ✅ Dry-run mode for safe testing
- ✅ Location clustering
- ✅ Beautiful progress bars and tables

## 📝 License

MIT License
