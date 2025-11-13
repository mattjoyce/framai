# 🎙️ FRAMAI - Field Recording Audio/Media Analysis & Integration

**A comprehensive CLI tool for field recordists who want to professionally document and archive their recordings with GPS, weather data, AI transcription, and automated audio processing.**

Perfect for sound librarians, nature recordists, acoustic ecologists, and anyone serious about cataloging field recordings.

## 🌟 What Does It Do?

FRAMAI takes your field recording session (audio files + photos from your phone) and automatically:

1. **Extracts location data** from your photos (GPS coordinates, timestamps)
2. **Fetches historical weather** for when/where you recorded (temperature, conditions, sunrise/sunset)
3. **Transcribes your verbal notes** using Whisper AI (local or API)
4. **Refines descriptions** with GPT-4 into professional catalog entries
5. **Trims audio files** based on your narration timestamps
6. **Applies fade effects** for polished final recordings

All packaged into a single JSON file with complete metadata for your audio library.

## 💡 Why Use This?

**Traditional workflow:**
- Manually note GPS coordinates
- Check weather websites days later
- Listen back to transcribe your notes
- Manually trim dead air in a DAW
- Copy/paste metadata into spreadsheets

**With FRAMAI:**
```bash
fram-cli images ./recordings/     # Extract all metadata
fram-cli transcribe ./recordings/  # Transcribe your notes
fram-cli refine fram.json          # Polish descriptions
fram-cli postprocess ./recordings/ # Auto-trim & fade
```

Done. Professional metadata + polished audio files.

## ✨ Key Features

### 📍 Image Processing
- Extract GPS coordinates from photo EXIF
- Get location names from OpenStreetMap (e.g., "Croydon Park, Sydney, NSW")
- Cluster nearby locations automatically
- No API key required for basic features

### 🌤️ Weather Integration
- Fetch historical weather from **Open-Meteo** (completely free!)
- Temperature, humidity, wind speed, precipitation
- Weather codes (clear sky, rain, fog, etc.)
- Sunrise/sunset times
- Cached requests for efficiency

### 🎤 Audio Transcription
- Uses **whisper-turbo** (optimized for Apple Silicon MLX) or OpenAI API
- Transcribes only header/footer (your verbal notes)
- Extracts timestamps for smart trimming
- Word-level timestamps available

### 🤖 GPT-4 Refinement
- Converts "umm, I'm standing in a field..." into professional catalog entries
- Uses prompts designed for audio librarians
- Example: "Suburban Garden Ambience: Light breeze, distant traffic, bird calls, recorded on sunny day"

### 🎚️ Audio Post-Processing
- Auto-trim based on transcription timestamps
- Apply fade in/out effects
- Preserves original files
- Configurable fade duration

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
python3 --version

# For Apple Silicon Macs (recommended)
# whisper-turbo will use MLX for fast transcription

# OpenAI API key (optional, for GPT refinement)
export OPENAI_API_KEY="sk-..."
```

### Installation

```bash
# Clone the repository
git clone https://github.com/mattjoyce/framai.git
cd framai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install as command
pip install -e .
```

### Basic Usage

```bash
# Process images to extract GPS and weather
fram-cli images ./my_recordings/

# Transcribe audio (first and last 30 seconds)
fram-cli transcribe ./my_recordings/ --duration 30

# Refine transcriptions with GPT-4
fram-cli refine fram.json

# Trim and fade audio files
fram-cli postprocess ./my_recordings/ --fade 30
```

## 📖 Detailed Command Guide

### 1. Images Command

Extract GPS, fetch weather, and geocode locations:

```bash
fram-cli images ./recordings/ [OPTIONS]

Options:
  -o, --output FILE        Output JSON file (default: fram.json)
  --weather/--no-weather   Fetch weather data (default: enabled)
  --geocode/--no-geocode   Fetch location names (default: enabled)
  --threshold METERS       Location clustering threshold (default: 1000)
  --extensions TEXT        Comma-separated extensions (default: jpg,jpeg,png)
```

**Example:**
```bash
fram-cli images ./field_trip/ --output metadata.json --threshold 500
```

**Output includes:**
- GPS coordinates from photos
- Location names (via OpenStreetMap)
- Historical weather data (temperature, humidity, wind, conditions)
- Sunrise/sunset times
- Clustered locations (groups nearby GPS points)

### 2. Transcribe Command

Transcribe verbal notes from audio files:

```bash
fram-cli transcribe ./recordings/ [OPTIONS]

Options:
  -o, --output FILE        Output JSON file (default: fram.json)
  --duration SECONDS       Seconds to transcribe (default: 30)
  --model TEXT            Whisper model (default: base.en)
  --types TEXT            File types (default: wav,mp3,flac)
  --timestamps/--no-timestamps  Word timestamps (default: enabled)
```

**Example:**
```bash
fram-cli transcribe ./field_trip/ --duration 45 --model turbo-v3
```

**How it works:**
- Extracts first N seconds (your opening narration)
- Extracts last N seconds (your closing notes)
- Uses Whisper to transcribe both segments
- Saves timestamps for trimming in post-processing

### 3. Refine Command

Polish transcriptions into professional descriptions:

```bash
fram-cli refine fram.json [OPTIONS]

Options:
  -o, --output FILE        Output JSON (default: overwrites input)
  --model TEXT            GPT model (default: gpt-4)
  --temperature FLOAT     Temperature (default: 0.1)
  --prompt TEXT           Custom prompt template
```

**Example:**
```bash
fram-cli refine fram.json --model gpt-4 --temperature 0.1
```

**Before:**
> "Recording is taken standing in a suburban back garden, uh, it's a sunny day. I heard helicopters, dog barking, traffic..."

**After:**
> "Suburban Garden Ambience: Recorded on a sunny day using a DIY microphone, capturing ambient noises including helicopter, dog barking, traffic, door closing, and person coughing."

### 4. Postprocess Command

Trim and fade audio based on transcription timestamps:

```bash
fram-cli postprocess ./recordings/ [OPTIONS]

Options:
  --json FILE              Input JSON (default: fram.json)
  --fade SECONDS          Fade duration (default: 30)
  --suffix TEXT           Output suffix (default: _POST)
  --header SECONDS        Header buffer (default: 0)
  --footer SECONDS        Footer buffer (default: 0)
```

**Example:**
```bash
fram-cli postprocess ./field_trip/ --fade 5 --suffix _FINAL
```

**What it does:**
- Reads header/footer timestamps from JSON
- Trims audio to remove dead air
- Applies smooth fade in/out
- Saves as `filename_POST.wav` (or custom suffix)

## 🎯 Complete Workflow Example

Here's a real-world example from a field recording session:

```bash
# 1. Process images taken with phone
fram-cli images ./my_recordings/ --output metadata.json

# Output:
# ✓ Found 1 image files
# ✓ Extracted metadata from 1 images
# ✓ Clustered to 1 unique locations
# Location: [Your Location]
# Weather: 12°C, 83% humidity, light drizzle
# Sunrise: 07:55, Sunset: 18:09

# 2. Transcribe verbal notes from audio files
fram-cli transcribe ./my_recordings/ --duration 30 --output metadata.json

# Output:
# ✓ Found 2 audio files
# ✓ Model loaded successfully
# ✓ Transcribed recording_001.wav
# "Recording is taken in a garden, sunny day, light breeze..."

# 3. Refine with GPT-4 (requires API key)
fram-cli refine metadata.json

# Output:
# ✓ Refined 2 transcriptions
# "Suburban Garden Ambience: Light breeze, distant aircraft,
#  traffic noise, bird calls..."

# 4. Trim and fade audio
fram-cli postprocess ./my_recordings/ --fade 30 --json metadata.json

# Output:
# ✓ Processed 2 audio files
# Created: recording_001_POST.wav (trimmed & faded)
```

**Final result:** Complete JSON metadata + polished audio files ready for your library.

## ⚙️ Configuration

### YAML Configuration

Edit `config.yaml` to customize:

```yaml
ai_models:
  openai:
    api_key: ${OPENAI_API_KEY}  # Use environment variable
    gpt_model: gpt-4
    temperature: 0.1
  whisper:
    model_name: turbo-v3  # or base.en, small.en, etc.

weather_api:
  provider: open-meteo
  cache_dir: .weather_cache

processing:
  audio:
    default_duration: 30  # Seconds to transcribe
    fade_duration: 30     # Fade effect length
  image:
    location_threshold_meters: 1000  # GPS clustering
```

### Environment Variables

```bash
# Add to ~/.env or export
export OPENAI_API_KEY="sk-your-key-here"
```

FRAMAI will automatically load from:
1. `~/.env` file
2. Environment variables
3. `config.yaml`

## 📊 Sample Output

From the test recording in `TestData/`:

```json
{
  "locations": [{
    "latitude": XX.XXXXX,
    "longitude": XX.XXXXX,
    "location_name": "[Your Location]",
    "datetime": "2023-07-23T13:27:11",
    "weather": {
      "temperature_2m": 11.9,
      "relative_humidity_2m": 83.3,
      "precipitation": 0.1,
      "weather_code": 51,
      "sunrise": "2023-07-23T07:55",
      "sunset": "2023-07-23T18:09"
    }
  }],
  "audio_events": [{
    "audio_filename": "recording_001.wav",
    "duration_seconds": 198.45,
    "header": 29.12,
    "footer": 11.62,
    "extracted_text": "Recording taken in suburban garden, sunny day...",
    "gpt_refined_text": "Suburban Garden Ambience: Sunny day with light breeze, capturing distant aircraft, traffic, bird calls..."
  }]
}
```

## 🛠️ Advanced Features

### Dry Run Mode

Test commands without making changes:

```bash
fram-cli images ./recordings/ --dry-run
fram-cli transcribe ./recordings/ --dry-run
fram-cli postprocess ./recordings/ --dry-run
```

### Custom Configuration

Use a different config file:

```bash
fram-cli --config my-config.yaml images ./recordings/
```

### Verbose Output

See detailed logging:

```bash
fram-cli --verbose transcribe ./recordings/
```

### Location Clustering

Automatically groups GPS points within threshold distance:

```bash
# Cluster locations within 500 meters
fram-cli images ./recordings/ --threshold 500
```

Useful when you move around a small area during recording.

## 🔧 Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS (Apple Silicon optimized), Linux, Windows
- **Optional**:
  - OpenAI API key (for GPT-4 refinement)
  - ffmpeg (for audio processing)

### Python Packages

All listed in `requirements.txt`:
- `click` - CLI framework
- `rich` - Beautiful terminal output
- `pyyaml` - Configuration
- `pillow` - Image EXIF
- `pydub` - Audio processing
- `openai` - GPT integration
- `whisper-turbo` - Fast transcription (Apple Silicon)
- `openmeteo-requests` - Weather data
- `requests-cache` - API caching

## 🏗️ Architecture

```
framai/
├── fram_cli.py           # Main CLI entry point
├── config.yaml           # Default configuration
├── commands/             # Command implementations
│   ├── images_cmd.py
│   ├── transcribe_cmd.py
│   ├── refine_cmd.py
│   └── postprocess_cmd.py
├── utils/               # Utility modules
│   ├── exif.py         # GPS/EXIF extraction
│   ├── geocoding.py    # Location lookup
│   ├── console.py      # Rich output helpers
│   └── audio.py        # Audio processing
├── weather.py          # Weather API client
└── config.py           # Configuration loader
```

## 🤝 Contributing

This is a personal project but suggestions welcome! Open an issue or PR.

## 🙏 Credits

Built with:
- [Open-Meteo](https://open-meteo.com/) - Free weather API
- [OpenStreetMap Nominatim](https://nominatim.org/) - Geocoding
- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription
- [whisper-turbo](https://github.com/JohannesGaessler/whisper-turbo) - MLX optimization
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Rich](https://github.com/Textualize/rich) - Terminal output

Inspired by the field recording community and the need for better archival tools.

## 📄 License

MIT License - See LICENSE file for details.

## 🐛 Known Issues

- whisper-turbo requires macOS (Apple Silicon) for MLX acceleration
- Large audio files may take time to process
- GPT-4 refinement requires paid OpenAI API access

## 💬 Questions?

Open an issue on GitHub or reach out to the community.

---

**Made with ❤️ for field recordists everywhere**
