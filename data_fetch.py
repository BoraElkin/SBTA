import requests
import pandas as pd
import logging
import boto3
import botocore
import json
import numpy as np
from data_cache import DataCache

def load_api_keys(filepath="keys.txt"):
    """
    Load API keys from a file.
    Expected format (one per line):
        KEY_NAME=VALUE
    Lines starting with '#' or empty ones are ignored.
    """
    keys = {}
    try:
        with open(filepath, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        keys[key.strip()] = value.strip()
        logging.info("Loaded API keys from %s.", filepath)
    except Exception as e:
        logging.error("Error loading API keys: %s", e)
    return keys

def fetch_nasa_data():
    """Fetch solar activity data from NASA's DONKI API."""
    try:
        keys = load_api_keys()
        nasa_api_key = keys.get("NASA_API_KEY", "DEMO_KEY")
        
        # Set up parameters
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "api_key": nasa_api_key
        }
        
        # Try to get from cache first
        cache = DataCache()
        cached_data = cache.get('nasa', params)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, fetch from API
        base_url = "https://api.nasa.gov/DONKI/FLR"
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        nasa_df = process_nasa_response(data)
        
        # Cache the results
        cache.save(nasa_df, 'nasa', params)
        
        return nasa_df
    except Exception as e:
        logging.error("Error fetching NASA data: %s", e)
        return pd.DataFrame()

def parse_time(time_str, current_date):
    """
    Helper function to parse time strings in different formats.
    
    Args:
        time_str (str): Time string in either 'HH:MM' or 'HH:MM:SS' format
        current_date (datetime): The base date to combine with the time
    
    Returns:
        datetime: Combined date and time, or None if parsing fails
    """
    formats = ['%H:%M:%S', '%H:%M']  # Try both formats
    
    # Input validation
    if not isinstance(time_str, str) or not isinstance(current_date, pd.Timestamp):
        logging.warning(f"Invalid input types: time_str={type(time_str)}, current_date={type(current_date)}")
        return None
        
    # Clean input
    time_str = time_str.strip()
    
    # Validate time format before parsing
    if not ':' in time_str:
        logging.warning(f"Invalid time format '{time_str}': missing colon separator")
        return None
        
    for fmt in formats:
        try:
            # Parse just the time part
            parsed_time = pd.to_datetime(time_str, format=fmt).time()
            
            # Validate hours and minutes
            if parsed_time.hour >= 24 or parsed_time.minute >= 60:
                continue
                
            # Combine with the current date
            return pd.Timestamp.combine(current_date.date(), parsed_time)
        except ValueError:
            continue
    
    logging.warning(f"Could not parse time string '{time_str}' in any known format")
    return None

def fetch_noaa_data():
    """Fetch solar activity data from NOAA S3 bucket."""
    try:
        s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
        bucket_name = "noaa-swpc-pds"
        
        # Get the latest SWPC reports
        prefix = "products/reports/primary-products/Report_of_Solar-Geophysical_Activity"
        
        # List objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            logging.error("No files found in NOAA S3 bucket")
            return pd.DataFrame()
            
        # Get the most recent file
        latest_file = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]
        key = latest_file['Key']
        
        # Get the file content
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        content = obj['Body'].read().decode('utf-8')
        
        # Parse the text content into structured data
        data_list = []
        current_date = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                # Check if line starts with a date (YYYY MM DD)
                if line[:10].replace(' ', '').isdigit():
                    current_date = pd.to_datetime(line[:10], format='%Y %m %d')
                    continue
                    
                # Parse data lines
                if current_date and ':' in line:  # Time entries
                    parts = line.split()
                    if len(parts) >= 3:  # Need at least time and two values
                        timestamp = parse_time(parts[0], current_date)
                        if timestamp is not None:
                            try:
                                data_list.append({
                                    'timestamp': timestamp,
                                    'intensity': float(parts[1]),
                                    'kp_index': float(parts[2])
                                })
                            except (ValueError, IndexError) as e:
                                logging.warning(f"Error parsing data values: {e}")
                                continue
                            
            except Exception as e:
                logging.warning(f"Skipping line due to parsing error: {e}")
                continue
        
        noaa_df = pd.DataFrame(data_list)
        logging.info("NOAA data fetched from S3 with %d records.", len(noaa_df))
        return noaa_df
    except Exception as e:
        logging.error("Error fetching NOAA data from S3: %s", e)
        return pd.DataFrame()

def generate_sample_data(n_samples=100):
    """Generate sample data for testing when API/S3 access is not available."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='h')  # Changed 'H' to 'h'
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'intensity': np.random.exponential(scale=2, size=n_samples),
        'kp_index': np.random.uniform(0, 9, size=n_samples),
        'location': np.random.choice(['N00W00', 'N30E45', 'S15W30'], size=n_samples)
    })
    
    # Add some synthetic events (ensure integer type)
    sample_data['event'] = (sample_data['intensity'] > 5).astype(np.int32)
    
    return sample_data

def fetch_solar_data(use_sample_data=False):
    """
    Fetch and combine solar activity data from multiple sources.
    
    Parameters:
    use_sample_data (bool): If True, uses generated sample data instead of real API calls
    
    Returns:
    pandas.DataFrame: Combined solar activity data.
    """
    if use_sample_data:
        logging.info("Using sample data for development/testing.")
        return generate_sample_data()
        
    nasa_df = fetch_nasa_data()
    noaa_df = fetch_noaa_data()
    
    # Combine data from both sources
    if nasa_df.empty and noaa_df.empty:
        logging.error("Both data sources returned empty data.")
        logging.info("Falling back to sample data.")
        return generate_sample_data()
    
    # Combine and align the data
    combined_df = pd.concat([nasa_df, noaa_df], ignore_index=True, sort=False)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    logging.info("Combined data has %d records.", len(combined_df))
    return combined_df

def process_nasa_response(data):
    """Process NASA API response into a DataFrame"""
    flares_data = []
    for flare in data:
        flare_info = {
            'timestamp': pd.to_datetime(flare['beginTime']),
            'intensity': flare['classType'],  # Example: 'M5.2', 'X1.1'
            'duration': pd.to_datetime(flare['endTime']) - pd.to_datetime(flare['beginTime']),
            'location': flare.get('sourceLocation', 'Unknown')
        }
        flares_data.append(flare_info)
        
    nasa_df = pd.DataFrame(flares_data)
    
    # Convert intensity to numeric (e.g., 'M5.2' -> 5.2, 'X1.1' -> 11.0)
    if not nasa_df.empty:
        nasa_df['intensity'] = nasa_df['intensity'].apply(lambda x: 
            float(x[1:]) * (10.0 if x.startswith('X') else 1.0))
    
    logging.info("NASA data processed with %d records.", len(nasa_df))
    return nasa_df 