import streamlit as st
import os
import requests
from azure.core.credentials import AzureKeyCredential
from azure.maps.search import MapsSearchClient
from azure.core.exceptions import HttpResponseError
import math
import numpy as np
from PIL import Image
import pandas as pd
import io

subscription_key = st.secrets["SUBSCRIPTION_KEY"]
maps_search_client = MapsSearchClient(
   credential=AzureKeyCredential(subscription_key)
)

def tile_to_lat_lon(x, y, zoom):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

#lat, lon = tile_to_lat_lon(2044, 1360, 12)
#st.write(f"Tile center: {lat:.6f}, {lon:.6f}")

def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile coordinates with validation"""
    import math
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    # Validate coordinates
    max_tiles = 2 ** zoom
    if x < 0 or x >= max_tiles or y < 0 or y >= max_tiles:
        st.warning(f"Warning: Tile coordinates ({x}, {y}) may be outside valid range for zoom {zoom}")
        st.warning(f"Valid range: x=[0, {max_tiles-1}], y=[0, {max_tiles-1}]")
    
    return x, y

def analyze_traffic_tile(image_data):
    """Analyze traffic tile image to extract flow information using PIL only"""
    
    # Load image from bytes and convert to RGB if needed
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Convert RGB to HSV manually (since we can't use cv2)
    def rgb_to_hsv(rgb):
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Calculate Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        # Calculate Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Calculate Value
        v = max_val
        
        return h, s, v
    
    # Convert image to HSV
    hsv_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=float)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            hsv_array[i, j] = rgb_to_hsv(img_array[i, j])
    
    # Define color ranges for different traffic levels in HSV
    # Red: Hue 0-10 or 350-360, Saturation > 0.2, Value > 0.2
    # Yellow: Hue 20-40, Saturation > 0.2, Value > 0.2  
    # Green: Hue 40-80, Saturation > 0.2, Value > 0.2
    
    red_pixels = 0
    yellow_pixels = 0
    green_pixels = 0
    
    for i in range(hsv_array.shape[0]):
        for j in range(hsv_array.shape[1]):
            h, s, v = hsv_array[i, j]
            
            # Check if pixel meets minimum saturation and value thresholds
            if s > 0.2 and v > 0.2:
                # Red traffic (Hue 0-10 or 350-360)
                if (h >= 0 and h <= 10) or (h >= 350 and h <= 360):
                    red_pixels += 1
                # Yellow traffic (Hue 20-40)
                elif h >= 20 and h <= 40:
                    yellow_pixels += 1
                # Green traffic (Hue 40-80)
                elif h >= 40 and h <= 80:
                    green_pixels += 1
    
    total_pixels = red_pixels + yellow_pixels + green_pixels
    
    # Calculate percentages
    red_percentage = (red_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    yellow_percentage = (yellow_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    green_percentage = (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    return {
        'heavy_traffic': red_percentage,
        'moderate_traffic': yellow_percentage,
        'light_traffic': green_percentage,
        'total_pixels': total_pixels
    }

def get_traffic_flow_tile(longitude, latitude, zoom):
    # Convert lat/lon to tile coordinates
    x, y = deg2num(latitude, longitude, zoom)
    
    st.write(f"Requesting tile at zoom {zoom}, coords ({x}, {y})")
    
    url = "https://atlas.microsoft.com/traffic/flow/tile/png"
    params = {
        'api-version': '1.0',
        'style': 'absolute',
        'zoom': zoom,
        'x': x,
        'y': y
    }
    headers = {
        'subscription-key': subscription_key
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        
        # Debug information
        st.write(f"Status Code: {response.status_code}")
        st.write(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
        st.write(f"Content Length: {len(response.content)} bytes")
        
        if response.status_code == 204:
            st.warning("No traffic data available for this location and zoom level")
            st.info("Try a different zoom level or location where traffic data is available")
            return False
        elif response.status_code != 200:
            st.error(f"API Error: {response.text}")
            return False
        
        # Check if response is actually an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            st.error(f"Expected image but got: {content_type}")
            st.error(f"Response content: {response.text[:200]}...")
            return False
        
        # Check if content is not empty
        if len(response.content) == 0:
            st.error("Received empty response")
            return None
        
        # Validate the image data before processing
        try:
            img = Image.open(io.BytesIO(response.content))
            img.verify()  # Verify it's a valid image
            st.write(f"Image size: {img.size}")
        except Exception as e:
            st.error(f"Downloaded data is not a valid image: {e}")
            return None
        
        st.success(f"Traffic tile downloaded successfully! Tile coords: ({x}, {y})")
        st.image(response.content)
        
        return response.content
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching traffic data: {e}")
        return False


def geocode(write_address):
    from azure.core.credentials import AzureKeyCredential
    from azure.maps.search import MapsSearchClient
    import pandas as pd

    maps_search_client = MapsSearchClient(credential=AzureKeyCredential(subscription_key))
    try:
        result = maps_search_client.get_geocoding(query=write_address)
        if result.get('features', False):
            coordinates = result['features'][0]['geometry']['coordinates']
            longitude = coordinates[0]
            latitude = coordinates[1]
            
            # Create DataFrame for st.map()
            df = pd.DataFrame({
                'lat': [latitude],
                'lon': [longitude]
            })
            st.title("Map, show the location")
            st.map(df, zoom=18)
            st.write(f"Coordinates: {longitude}, {latitude}")
            
            # Return coordinates for use in traffic function
            return longitude, latitude
        else:
            st.error("No results")
            return None, None

    except HttpResponseError as exception:
        if exception.error is not None:
            st.error(f"Error Code: {exception.error.code}")
            st.error(f"Message: {exception.error.message}")
        return None, None

if __name__ == "__main__":
    st.header("Traffic Flow Analyzer")
    st.write("This is a realtime traffic flow analyzer, you can enter an address and get the traffic flow data for that location.")
    st.caption("Azure maps request that returns real-time information about traffic conditions in 256 x 256 pixel tiles that show traffic flow, for more information see https://learn.microsoft.com/en-us/rest/api/maps/traffic/get-traffic-flow-tile?view=rest-maps-1.0&tabs=HTTP")
    st.caption("The data is analyzed by counting the number of pixels in the red, yellow, and green areas of the tile.")
    st.caption("The red area is the most congested, the yellow area is the moderate traffic, and the green area is the least congested.")
    write_address = st.text_input("Enter an address", placeholder="Format like 'Mannerheimintie 9, Helsinki, Finland'")
    st.caption("Zoom level is the level of detail of the tile returned from the azure maps api, the higher the zoom level, the more detailed the tile is.")
    st.caption("Not all tile zoom levels are available for all locations, for example some areas in the world may not have a zoom level 12 or higher.")
    st.caption("Safest zoom level to use is 12, because it is the most detailed zoom level and it is available for most locations.")
    zoom = st.selectbox("Select zoom level for the tile returned from the azure maps api", 
                   options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                   index=12) 
    if write_address and zoom:
       # with st.expander("Validate location on map"):
        st.write("Validating location on map")
        x, y = geocode(write_address)  # Only call once
        if x is not None and y is not None:
            result = get_traffic_flow_tile(x, y, zoom)

            if result:
                analysis = analyze_traffic_tile(result)
                st.dataframe(pd.DataFrame([analysis]))
    else:
        st.error("Please enter an address")