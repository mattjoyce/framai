"""
Refine command implementation
Refines transcriptions using GPT-4
Refactored from fram-audio.py to use new infrastructure
"""

import json
import logging
from typing import Any, Dict, Optional

from utils.console import (
    create_progress_bar,
    print_dry_run_summary,
    print_error,
    print_header,
    print_section,
    print_success,
    print_warning
)

logger = logging.getLogger(__name__)


def run_refine_command(json_file: str, config: Dict[str, Any],
                      options: Dict[str, Any], dry_run: bool = False,
                      verbose: bool = False) -> bool:  # pylint: disable=unused-argument
    """
    Main entry point for refine command.

    Args:
        json_file: Path to JSON file with transcriptions
        config: Configuration dictionary
        options: Command options (output_file, model, temperature, etc.)
        dry_run: If True, show what would be done without executing
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    print_header("FRAMAI - GPT Refinement")

    # Extract options
    output_file = options['output_file']
    model = options['model']
    temperature = options['temperature']
    prompt_template = options.get('prompt_template')

    # Show configuration
    print_section("Configuration")
    print(f"  Input file: {json_file}")
    print(f"  Output file: {output_file}")
    print(f"  GPT model: {model}")
    print(f"  Temperature: {temperature}")

    # Load JSON file
    print_section("Loading transcriptions")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print_error(f"JSON file not found: {json_file}")
        return False
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON file: {e}")
        return False

    # Count audio events with transcriptions
    audio_events = data.get('audio_events', [])
    events_to_refine = [e for e in audio_events if e.get('extracted_text')]

    if not events_to_refine:
        print_warning("No transcriptions found to refine")
        return False

    print_success(f"Found {len(events_to_refine)} transcriptions to refine")

    # Dry-run mode
    if dry_run:
        actions = [
            f"Refine {len(events_to_refine)} transcriptions using GPT-4",
            f"Model: {model}, Temperature: {temperature}",
            "Save refined text back to JSON",
            f"Output to: {output_file}"
        ]

        print_dry_run_summary("GPT Refinement Plan", actions)
        return True

    # Get API key
    api_key = config['ai_models']['openai'].get('api_key')
    if not api_key or api_key == '${OPENAI_API_KEY}':
        print_error("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        return False

    # Refine transcriptions
    print_section("Refining transcriptions with GPT-4")
    refined_count = refine_transcriptions(events_to_refine, data, config, model,
                                         temperature, prompt_template, api_key)

    if refined_count == 0:
        print_error("Failed to refine transcriptions")
        return False

    print_success(f"Refined {refined_count} transcriptions")

    # Save results
    print_section("Saving results")
    data['audio_events'] = audio_events  # Update with refined texts
    success = save_results(data, output_file)

    if success:
        print_success(f"Results saved to {output_file}")
        return True
    else:
        print_error(f"Failed to save results to {output_file}")
        return False


def refine_transcriptions(audio_events: list, full_data: Dict[str, Any],
                         config: Dict[str, Any],  # pylint: disable=unused-argument
                         model: str, temperature: float,
                         prompt_template: Optional[str],
                         api_key: str) -> int:
    """
    Refine transcriptions using GPT-4 with full context.

    Args:
        audio_events: List of audio event dictionaries
        full_data: Complete JSON data including locations, weather, etc.
        config: Configuration dictionary
        model: GPT model name
        temperature: Temperature setting
        prompt_template: Optional custom prompt template
        api_key: OpenAI API key

    Returns:
        Number of successfully refined transcriptions
    """
    from pathlib import Path

    # Load prompt template from file
    prompt_file = Path(__file__).parent.parent / 'prompts' / 'refine_prompt.txt'

    if prompt_template:
        # Use custom prompt if provided
        user_prompt = prompt_template
    elif prompt_file.exists():
        # Load from external file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            user_prompt = f.read()
    else:
        # Fallback to simple prompt
        user_prompt = "Write a concise recording description from these notes:\n\n{text}"

    refined_count = 0

    with create_progress_bar("Refining with GPT-4", total=len(audio_events)) as progress:
        task = progress.add_task("Processing", total=len(audio_events), status="Starting...")

        for event in audio_events:
            text = event.get('extracted_text')
            if not text:
                progress.advance(task)
                continue

            try:
                # Build context from full data
                context_parts = []

                # Add location information
                locations = full_data.get('locations', [])
                if locations:
                    loc = locations[0]  # Use first location
                    context_parts.append(f"Location: {loc.get('location_name', 'Unknown')}")
                    context_parts.append(f"Coordinates: {loc.get('latitude', 'N/A')}, {loc.get('longitude', 'N/A')}")

                    # Add weather data
                    weather = loc.get('weather', {})
                    if weather:
                        temp = weather.get('temperature_2m', 'N/A')
                        humidity = weather.get('relative_humidity_2m', 'N/A')
                        weather_code = weather.get('weather_code', 'N/A')
                        sunrise = weather.get('sunrise', 'N/A')
                        sunset = weather.get('sunset', 'N/A')

                        context_parts.append(f"Temperature: {temp}°C")
                        context_parts.append(f"Humidity: {humidity}%")
                        context_parts.append(f"Weather code: {weather_code}")
                        context_parts.append(f"Sunrise: {sunrise}")
                        context_parts.append(f"Sunset: {sunset}")

                # Add recording metadata
                context_parts.append(f"Filename: {event.get('audio_filename', 'Unknown')}")
                context_parts.append(f"Duration: {event.get('duration_seconds', 'N/A')}s")
                context_parts.append(f"Recorded: {event.get('created_date', 'N/A')}")

                context = "\n".join(context_parts)

                # Call GPT-4 (using new OpenAI API >=1.0.0)
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                # Format prompt with context and text
                formatted_prompt = user_prompt.format(context=context, text=text)

                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ]
                )

                refined_text = response.choices[0].message.content
                event['gpt_refined_text'] = refined_text
                refined_count += 1

                progress.update(task, status=f"✓ {event['audio_filename'][:20]}")
                progress.advance(task)

            except Exception as e:
                logger.error(f"Error refining {event.get('audio_filename', 'unknown')}: {e}")
                progress.update(task, status=f"❌ Error: {event['audio_filename'][:20]}")
                progress.advance(task)

    return refined_count


def save_results(data: Dict[str, Any], output_file: str) -> bool:
    """
    Save refined data to JSON file.

    Args:
        data: Data dictionary
        output_file: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved results to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


if __name__ == '__main__':
    # Test refine command
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        test_json = sys.argv[1]

        # Load config
        from config import load_config
        config = load_config()

        # Test options
        options = {
            'output_file': test_json,  # Overwrite input
            'model': 'gpt-4',
            'temperature': 0.1,
            'prompt_template': None
        }

        # Run command
        success = run_refine_command(
            json_file=test_json,
            config=config,
            options=options,
            dry_run=False,
            verbose=True
        )

        sys.exit(0 if success else 1)
    else:
        print("Usage: python refine_cmd.py <json_file>")
