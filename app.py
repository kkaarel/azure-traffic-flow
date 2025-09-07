import streamlit as st
import os
import requests
from azure.core.credentials import AzureKeyCredential
from azure.maps.search import MapsSearchClient
from azure.core.exceptions import HttpResponseError
import math
import cv2
import numpy as np
from PIL import Image
import pandas as pd

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

def analyze_traffic_tile(image_path):
    """Analyze traffic tile image to extract flow information"""
    
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for different traffic levels
    # (These would need to be calibrated based on Azure's color scheme)
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])
    
    yellow_lower = np.array([20, 50, 50])
    yellow_upper = np.array([30, 255, 255])
    
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    
    # Create masks for each traffic level
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Count pixels for each traffic level
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
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
            return False
        
        # Save the image
        with open('traffic_tile.png', 'wb') as f:
            f.write(response.content)
        
        # Validate the saved image before displaying
        try:
            from PIL import Image
            with Image.open('traffic_tile.png') as img:
                img.verify()  # Verify it's a valid image
                st.write(f"Image size: {img.size}")
        except Exception as e:
            st.error(f"Downloaded file is not a valid image: {e}")
            return False
        
        st.success(f"Traffic tile downloaded successfully! Tile coords: ({x}, {y})")
        st.image('traffic_tile.png')
        
        return True
        
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
                st.dataframe(analyze_traffic_tile('traffic_tile.png'))
    else:
        st.error("Please enter an address")