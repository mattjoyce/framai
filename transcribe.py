# transcribe.py
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple
import time

import openai
import whisper
from pydub import AudioSegment
from rich.logging import RichHandler

from datetime import datetime

# Function to find the closest location ID based on datetime
def find_closest_location_id(created_date, locations):
    audio_date = datetime.strptime(created_date, "%Y-%m-%dT%H:%M:%S")
    min_diff = float('inf')
    closest_id = None
    
    for location in locations:
        location_date = datetime.strptime(location["DateTime"], "%Y-%m-%dT%H:%M:%S")
        diff = abs((audio_date - location_date).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_id = location["ID"]
            
    return closest_id

def link_audio_to_closest_location(folder, jsonout):
    json_fsp = os.path.join(folder, jsonout)
    logger.debug(json_fsp)
    
    try:
        with open(json_fsp, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    if not data.get('locations'):
        logger.warning("No location data found.")
        return None

    if not data.get('audio_events'):
        logger.warning("No audio events found.")
        return None

    for audio_info in data['audio_events']:
        created_date = audio_info['created_date']
        loc_id = find_closest_location_id(created_date, data['locations'])
        audio_info['location_id'] = loc_id

    # Write back the updated data to the JSON file
    with open(json_fsp, "w") as file:
        json.dump(data, file, indent=4)

        
            
        


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process audio files and transcribe words."
    )
    parser.add_argument(
        "folder", type=str, help="Path to the folder containing audio files."
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="If True, refine the extracted text using GPT. Default is False.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds for the beginning and end of audio to transcribe. Default is 30.",
    )
    parser.add_argument(
        "--audio_type",
        type=str,
        nargs='+',
        default=["wav"],
        help="Type of audio files to process. Default is 'wav'.",
    )
    parser.add_argument(
        "--narrative",
        type=str,
        default=None,
        help="Additional narrative words to include in the transcription. Optional.",
    )
    parser.add_argument(
        "--jsonout",
        type=str,
        default="fram.json",
        help="File to save file metadata and extracted text. Default is 'fram.json'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default="False",
        help="File to save file metadata and extracted text. Default is 'fram.json'.",
    )
    return parser.parse_args()


def get_openai_api_key():
    """
    Get the OpenAI API key from the environment variable 'OPENAI_API_KEY'.

    Returns:
    str: OpenAI API key.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable."
        )
    return api_key


def get_first_and_last_word_duration(json_data):
    """
    Get the start time of the first word and the end time of the last word in the entire audio.

    Parameters:
    json_data (dict): JSON data containing the transcription information.

    Returns:
    tuple: A tuple containing the start time of the first word and the end time of the last word in seconds.
    """
    segments = json_data.get("segments", [])
    if not segments:
        raise ValueError("No segments found in the JSON data.")

    first_segment = segments[0]
    last_segment = segments[-1]

    # Get the start time of the first word in the entire audio
    first_word_start_time = first_segment["words"][0]["start"]

    # Get the end time of the last word in the entire audio
    last_word_end_time = last_segment["words"][-1]["end"]

    return first_word_start_time, last_word_end_time


def refine_extracted_text(text: str) -> str:
    """
    Process the extracted text from transcription: clean it and add an entry to a JSON file.

    Parameters:
    text (str): The extracted text from the transcription.
    api_key (str): OpenAI API key.
    json_file (str): Path to the JSON file.
    audio_filename (str): Name of the audio file.
    """
    # Use GPT to clean the text.
    system_prompt = "Adopt the role of an audio librarian"

    print("Checking extracted text")
    response = openai.ChatCompletion.create(
        api_key=get_openai_api_key(),
        model="gpt-4",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Stick to the facts, turn these notes into a sentence. : {text}",
            },
        ],
    )

    refined_text = response["choices"][0]["message"]["content"]
    return refined_text


def transcribe(
    folder,
    refine=False,
    duration=30,
    audio_type="wav",
    narrative="",
    jsonout="fram.json",
):
    """
    Process all the files in the specified folder and transcribe words.

    Arguments:
    folder: Path to the folder containing audio files.

    Options:
    --refine: If True, refine the extracted text using GPT. Default is False.
    --duration: Duration in seconds for the beginning and end of audio to transcribe. Default is 30.
    --audio_type: Type of audio files to process. Default is 'wav'.
    --narrative: Additional narrative words to include in the transcription. Optional.
    --jsonout: File to save file metadata and extracted text. Default is "fram.json".
    """
    if duration is not None:
        if duration <= 0:
            raise ValueError("Duration must be a positive integer value.")

    # get a list of files to process
    sorted_audio_files = filter_and_sort_audio_files(folder, audio_type)
    logger.debug(f'Found {len(sorted_audio_files)} files to process')
    # process each file
    
    audio_event_count=0
    for audio_file, mtime in sorted_audio_files:
        logger.info(f"Processing audio file: {audio_file}")
        file_path = os.path.join(folder, audio_file)
        audio = AudioSegment.from_file(file_path)
        text = ""
        # Fetch the creation date
        created_date = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(os.path.getctime(file_path)))
        
        data_to_append = {
            "audio_filename": audio_file,
            "created_date": created_date  # Add the creation date here
        }
        # If audio is longer than twice the duration, transcribe the first and last duration
        if duration > len(audio):
            duration = len(audio)

        # header
        header = audio[: duration * 1000]
        response = transcribe_audio_segment(header)
        logger.debug(response)
        if response['segments']:
            text += response["text"]
            first, last = get_first_and_last_word_duration(response)
            data_to_append["header"] = last
        else:
            logger.info("no segments to transcribe")

        #footer
        if len(audio) > duration * 2 * 1000:
            footer = audio[-duration * 1000 :]
            response = transcribe_audio_segment(footer)
            logger.debug(response)
            if response['segments']:
                text += response["text"]
                first, last = get_first_and_last_word_duration(response)
                data_to_append["footer"] = first
            else:
                logger.info("no segments to transcribe")

        if text:
            logger.info(f'Final transcripted text : {text}')
            data_to_append["extracted_text"] = text
        else:
            logger.info("No text extracted")
            
        refined_text = None
        if refine:
            if text:
                logger.info("refining text")
                refined_text = refine_extracted_text(text)
                data_to_append["gpt_refined_text"] = refined_text
            else :
                logger.info("no text to refine")
            
        # save data
        # Add the cleaned text to the JSON file.

        json_fsp=os.path.join(folder,jsonout)
        logger.debug(json_fsp)

        try:
            with open(json_fsp, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        # Check if data is a list, and if so, make it a dictionary
        if isinstance(data, list):
            data = {}
            
        if "audio_events" not in data:
            data["audio_events"] = []
            
        if narrative:
            data["provided text"]=narrative

        # Associating the "audio_file" key with the "data_to_append" value
        data["audio_events"].append(data_to_append)
        
        with open(json_fsp, "w") as file:
            json.dump(data, file, indent=4)


def filter_and_sort_audio_files(folder: Path, audio_types: List[str]) -> List[Tuple[str, float]]:
    """
    Get all files in the folder with the specified audio type and sort by modification time.

    Parameters:
    folder (str): Path to the directory containing audio files.
    audio_type (str): Extension of the audio files.

    Returns:
    List[Tuple[str, float]]: Sorted list of audio files and their modification times.
    """
    # Get all files in the directory
    all_files = os.listdir(folder)
    audio_extensions = [ext.lower() for ext in audio_types] + [ext.upper() for ext in audio_types]
    # Filter for audio files
    audio_files = [file for file in all_files if any(file.endswith(ext) for ext in audio_extensions)]
    
    # Associate files with their modification times
    audio_files_times = [
        (file, os.path.getmtime(os.path.join(folder, file))) for file in audio_files
    ]

    # Sort files based on modification time, oldest first
    sorted_audio_files = sorted(audio_files_times, key=lambda x: x[1], reverse=False)

    return sorted_audio_files


def transcribe_audio_segment(audio_segment: AudioSegment) -> str:
    """
    Transcribe an audio segment using the OpenAI Whisper ASR API.

    Parameters:
    audio_segment (AudioSegment): PyDub AudioSegment to transcribe.
    api_key (str): API key for OpenAI.

    Returns:
    str: Transcribed text.
    """
    # Export audio segment to wav format for transcription
    audio_file = audio_segment.export("temp.mp3", format="mp3")

    # Transcribe the audio file
    logger.info("Transcribing audio")
    whisper._download(whisper._MODELS["base.en"], "./models/", False)
    model = whisper.load_model("base.en")
    response = model.transcribe("temp.mp3", word_timestamps=True,fp16=False)
    # response = openai.Audio.transcribe(
    #     model="whisper-1", api_key=get_openai_api_key(), file=audio_file
    # )
    return response


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format="%(message)s",
        handlers=[RichHandler(markup=True)]    
    )       
    logger = logging.getLogger('rich')

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    logger.info("Starting")
    
    logger.debug(args)
    
    transcribe(
        args.folder,
        args.refine,
        args.duration,
        args.audio_type,
        args.narrative,
        args.jsonout,
    )

    #link audio files to location
    link_audio_to_closest_location(args.folder,args.jsonout)
    
    