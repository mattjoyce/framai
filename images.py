import argparse
import json
import os
from datetime import datetime, timedelta
from fractions import Fraction
from math import atan2, cos, radians, sin, sqrt

import requests
from PIL import Image
from PIL.ExifTags import TAGS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process image files and extract metadata for audio files."
    )
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing image\audio files."
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="If True, link audio records to most likely image based on time of day.",
    )
    parser.add_argument(
        "--weather",
        action="store_true",
        help="If True, get weather data for the locations.",
    )
    parser.add_argument(
        "--address",
        action="store_true",
        default="false",
        help="If True, get weather data for the locations.",
    )
    parser.add_argument(
        "--jsonout",
        type=str,
        default="fram.json",
        help="File in the folder to save metadata to. Default is 'fram.json'.",
    )
    return parser.parse_args()


def round_time_to_nearest_hour(time_str):
    # Convert the time string to a datetime object
    time_obj = datetime.strptime(time_str, '%H:%M')

    # Add 30 minutes to round to the nearest hour
    time_obj += timedelta(minutes=30)

    # Reset the minutes and seconds to 0
    time_obj = time_obj.replace(minute=0, second=0)

    # Convert the datetime object back to a string and return it
    return time_obj.strftime('%H:%M')

def dms_to_decimal(dms):
    """Convert degrees, minutes, seconds to decimal"""
    degrees, minutes, seconds = dms
    return degrees + minutes/60 + seconds/3600

def get_weather_data(lat, lon, datetime_str, api_key):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat}%2C{lon}/{datetime_str}?key={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get weather data: {response.content}")
        return None

def update_json_file(locations, folder, filename):
    full_path = os.path.join(folder, filename)
    print(locations)
    # Check if the file already exists
    if os.path.exists(full_path):
        # Load existing data
        with open(full_path, 'r') as file:
            existing_data = json.load(file)
        # Modify the data as needed, e.g., append or update
        existing_data['locations']=locations 
    else:
        existing_data = locations

    # Save the updated data back to the file
    with open(full_path, 'w') as file:
        json.dump(existing_data, file)

    print(f'Saved updated data to {full_path}')







def extract_relevant_weather_data(weather_data, image_timestamp):
    relevant_data = {}

    # Extract data from the top level of the weather data
    relevant_data['timezone'] = weather_data.get('timezone')

    # Get the date from the image's timestamp
    image_date = image_timestamp.split('T')[0]

    # Get the data for the date matching the image's timestamp
    for day_data in weather_data.get('days', []):
        if day_data['datetime'] == image_date:
            # Extract data from the day level
            relevant_data.update({
                'day_conditions': day_data.get('conditions'),
                'day_description': day_data.get('description')
            })

            # Get the time from the image's timestamp
            image_time = image_timestamp.split('T')[1][:5]  # 'HH:MM'
            image_time = round_time_to_nearest_hour(image_time)
            print('Time from image ' + image_time)
            # Get the hour that matches the image's timestamp
            for hour_data in day_data.get('hours', []):
                hour_str = hour_data['datetime'][:5]  # 'HH:MM:SS'
                if hour_str == image_time:
                    # Extract data from the hour level
                    relevant_data.update({
                        'hour_temp': hour_data.get('temp'),
                        'hour_feels_like': hour_data.get('feelslike'),
                        'hour_humidity': hour_data.get('humidity'),
                        'hour_precip': hour_data.get('precip'),
                        'hour_precipprob': hour_data.get('precipprob'),
                        'hour_snow': hour_data.get('snow'),
                        'hour_snowdepth': hour_data.get('snowdepth'),
                        'hour_preciptype': hour_data.get('preciptype'),
                        'hour_windgust': hour_data.get('windgust'),
                        'hour_windspeed': hour_data.get('windspeed'),
                        'hour_winddir': hour_data.get('winddir'),
                        'hour_pressure': hour_data.get('pressure'),
                        'hour_visibility': hour_data.get('visibility'),
                        'hour_cloudcover': hour_data.get('cloudcover'),
                        'hour_conditions': hour_data.get('conditions')
                    })
                    break

    return relevant_data

def get_image_metadata(image_path):
    image = Image.open(image_path)
    
    # Fetch raw EXIF data
    exif_data = image._getexif()
    if exif_data is not None:
        # Fetch DateTime, if available
        datetime_str = exif_data.get(306, None)
        
        # Convert the DateTime string into a datetime object
        datetime_obj = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S") if datetime_str is not None else None
        
        # Fetch GPSInfo, if available
        gpsinfo_raw = exif_data.get(34853, None)
        
        # Process GPSInfo data
        if gpsinfo_raw is not None:
            # Get latitude DMS and hemisphere
            latitude_dms = gpsinfo_raw.get(2, None)
            latitude_hemisphere = gpsinfo_raw.get(1, None)
            
            # Convert latitude DMS to decimal
            latitude = dms_to_decimal(latitude_dms) if latitude_dms is not None else None
            if latitude_hemisphere == 'S':  # If in the southern hemisphere, make latitude negative
                latitude *= -1
            
            # Get longitude DMS and hemisphere
            longitude_dms = gpsinfo_raw.get(4, None)
            longitude_hemisphere = gpsinfo_raw.get(3, None)
            
            # Convert longitude DMS to decimal
            longitude = dms_to_decimal(longitude_dms) if longitude_dms is not None else None
            if longitude_hemisphere == 'W':  # If in the western hemisphere, make longitude negative
                longitude *= -1
            
            gpsinfo = {
                'Latitude': "{:.6f}".format(float(latitude)),
                'Longitude': "{:.6f}".format(float(longitude))
            }
        else:
            gpsinfo = None

        # Store the data into a dictionary
        data = {
            "DateTime": datetime_obj.isoformat() if datetime_obj is not None else None,
            "GPSInfo": gpsinfo
        }
        
        return data  # Return the data as a JSON string

    # Return None if there's no EXIF data
    return None





def calculate_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    # Convert to meters
    return distance * 1000

def consolidate_images(folder, threshold_meters=1000):
    locations = []
    location_id = 0

    for file_name in os.listdir(folder):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder, file_name)
            metadata = get_image_metadata(image_path)
            if metadata and metadata["GPSInfo"]:
                lat = float(metadata["GPSInfo"]["Latitude"])
                lon = float(metadata["GPSInfo"]["Longitude"])
                date = metadata['DateTime']

                # Check if the current location is close to any existing locations
                matched = False
                for location in locations:
                    existing_lat = location["Latitude"]
                    existing_lon = location["Longitude"]
                    distance = calculate_distance(lat, lon, existing_lat, existing_lon)
                    if distance <= threshold_meters:
                        # Add this image to the existing location
                        if "Images" not in location:
                            location["Images"] = [] # Initialize if not present
                        location["Images"].append({
                            "FileName": file_name
                        })
                        matched = True
                        break

                if not matched:
                    # Create a new location record
                    locations.append({
                        "ID": location_id,  # Assigning the current location ID
                        "Latitude": lat,
                        "Longitude": lon,
                        "DateTime": date,  # Moved DateTime up to location level
                        "Images": [{"FileName": file_name}]
                    })
                    location_id += 1  # Increment the location ID

    return locations

import requests
import time

def get_location_details(lat, lon):
    time.sleep(1.1)  # Delay to comply with rate limiting
    url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
    headers = {"User-Agent": "myApplication"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        display_name = data['display_name']
        road = data['address']['road']
        # You can extract other details as needed
        return {"display_name": display_name, "road": road}
    else:
        print(f"Error fetching location details: {response.status_code}")
        return None


def process_images(folder, align, weather, address, jsonout):
    locations = consolidate_images(folder)
    print(locations)

    if weather:
        # Check for the presence of the Visual Crossing API key in the environment variable
        api_key = os.environ.get('VISUAL_CROSSING_API_KEY')
        if api_key:
            # Loop through the locations to get weather data
            for location in locations:
                lat = location["Latitude"]
                lon = location["Longitude"]
                datetime_str = location["DateTime"]
                # Get the weather data for the given location and time
                weather_data = get_weather_data(lat, lon, datetime_str, api_key)
                # Extract relevant weather information
                relevant_weather_data = extract_relevant_weather_data(weather_data, datetime_str)
                # Append the relevant weather data to the location's information
                location["Weather"] = relevant_weather_data
        else:
            print("Weather data requested, but no Visual Crossing API key found in environment variables.")

    if address:
        for location in locations:
            lat = location['Latitude']
            lon = location['Longitude']
            address_details = get_location_details(lat, lon)
            location['Address'] = address_details



    update_json_file(locations, folder, jsonout)

    print(locations)


if __name__ == "__main__":
    args = parse_args()
    process_images(
        args.folder,
        args.align,
        args.weather,
        args.address,
        args.jsonout,
    )