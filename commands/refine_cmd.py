"""
Refine command implementation
Refines transcriptions using GPT-4
Refactored from fram-audio.py to use new infrastructure
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import utilities
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

logger = logging.getLogger(__name__)


def run_refine_command(json_file: str, config: Dict[str, Any],
                      options: Dict[str, Any], dry_run: bool = False,
                      verbose: bool = False) -> bool:
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
        with open(json_file, 'r') as f:
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
            f"Save refined text back to JSON",
            f"Output to: {output_file}"
        ]

        print_dry_run_summary("GPT Refinement Plan", actions)
        return True

    # Check for OpenAI API
    try:
        import openai
    except ImportError:
        print_error("OpenAI library not installed. Install with: pip install openai")
        return False

    # Get API key
    api_key = config['ai_models']['openai'].get('api_key')
    if not api_key or api_key == '${OPENAI_API_KEY}':
        print_error("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        return False

    # Refine transcriptions
    print_section("Refining transcriptions with GPT-4")
    refined_count = refine_transcriptions(events_to_refine, config, model,
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


def refine_transcriptions(audio_events: list, config: Dict[str, Any],
                         model: str, temperature: float,
                         prompt_template: Optional[str],
                         api_key: str) -> int:
    """
    Refine transcriptions using GPT-4.

    Args:
        audio_events: List of audio event dictionaries
        config: Configuration dictionary
        model: GPT model name
        temperature: Temperature setting
        prompt_template: Optional custom prompt template
        api_key: OpenAI API key

    Returns:
        Number of successfully refined transcriptions
    """
    try:
        import openai
        openai.api_key = api_key
    except ImportError:
        return 0

    # Get prompts from config
    system_prompt = config['ai_models']['openai']['prompts'].get(
        'audio_librarian',
        "You are an audio librarian with expertise in cataloging field recordings."
    )

    if not prompt_template:
        prompt_template = config['ai_models']['openai']['prompts'].get(
            'recording_description',
            "Write a concise recording description from these notes: {text}"
        )

    refined_count = 0

    with create_progress_bar("Refining with GPT-4", total=len(audio_events)) as progress:
        task = progress.add_task("Processing", total=len(audio_events), status="Starting...")

        for event in audio_events:
            text = event.get('extracted_text')
            if not text:
                progress.advance(task)
                continue

            try:
                # Call GPT-4
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_template.format(text=text)}
                    ]
                )

                refined_text = response['choices'][0]['message']['content']
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
        with open(output_file, 'w') as f:
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
