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


st.set_page_config(page_title="Traffic Flow Analyzer", page_icon="🚗", layout="wide")

subscription_key = st.secrets["SUBSCRIPTION_KEY"]
maps_search_client = MapsSearchClient(
   credential=AzureKeyCredential(subscription_key)
)



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
    
    # Store pixel-level data for hotspot detection
    pixel_data = []
    
    for i in range(hsv_array.shape[0]):
        for j in range(hsv_array.shape[1]):
            h, s, v = hsv_array[i, j]
            
            # Check if pixel meets minimum saturation and value thresholds
            if s > 0.2 and v > 0.2:
                # Red traffic (Hue 0-10 or 350-360)
                if (h >= 0 and h <= 10) or (h >= 350 and h <= 360):
                    red_pixels += 1
                    pixel_data.append({'x': j, 'y': i, 'type': 'heavy', 'intensity': v})
                # Yellow traffic (Hue 20-40)
                elif h >= 20 and h <= 40:
                    yellow_pixels += 1
                    pixel_data.append({'x': j, 'y': i, 'type': 'moderate', 'intensity': v})
                # Green traffic (Hue 40-80)
                elif h >= 40 and h <= 80:
                    green_pixels += 1
                    pixel_data.append({'x': j, 'y': i, 'type': 'light', 'intensity': v})
    
    total_pixels = red_pixels + yellow_pixels + green_pixels
    
    # Calculate percentages
    red_percentage = (red_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    yellow_percentage = (yellow_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    green_percentage = (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    return {
        'heavy_traffic': red_percentage,
        'moderate_traffic': yellow_percentage,
        'light_traffic': green_percentage,
        'total_pixels': total_pixels,
        'pixel_data': pixel_data,
        'image_size': img_array.shape
    }

def detect_traffic_hotspots(pixel_data, image_size, grid_size=8):
    """Detect traffic hotspots by dividing image into grid and finding high-density areas"""
    height, width = image_size[0], image_size[1]
    grid_height = height // grid_size
    grid_width = width // grid_size
    
    hotspots = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Define grid cell boundaries
            start_y = i * grid_height
            end_y = min((i + 1) * grid_height, height)
            start_x = j * grid_width
            end_x = min((j + 1) * grid_width, width)
            
            # Count traffic pixels in this grid cell
            cell_pixels = [p for p in pixel_data 
                          if start_x <= p['x'] < end_x and start_y <= p['y'] < end_y]
            
            if cell_pixels:
                # Calculate density metrics
                heavy_count = len([p for p in cell_pixels if p['type'] == 'heavy'])
                moderate_count = len([p for p in cell_pixels if p['type'] == 'moderate'])
                light_count = len([p for p in cell_pixels if p['type'] == 'light'])
                total_count = len(cell_pixels)
                
                # Calculate center coordinates of grid cell
                center_x = (start_x + end_x) // 2
                center_y = (start_y + end_y) // 2
                
                # Calculate traffic intensity score
                intensity_score = (heavy_count * 3 + moderate_count * 2 + light_count * 1) / total_count if total_count > 0 else 0
                
                # Only include hotspots with significant traffic
                if total_count > 5 and intensity_score > 1.5:  # Threshold for hotspot detection
                    hotspots.append({
                        'grid_x': j,
                        'grid_y': i,
                        'center_x': center_x,
                        'center_y': center_y,
                        'heavy_count': heavy_count,
                        'moderate_count': moderate_count,
                        'light_count': light_count,
                        'total_count': total_count,
                        'intensity_score': intensity_score,
                        'bounds': {
                            'start_x': start_x,
                            'end_x': end_x,
                            'start_y': start_y,
                            'end_y': end_y
                        }
                    })
    
    # Sort by intensity score (highest first)
    hotspots.sort(key=lambda x: x['intensity_score'], reverse=True)
    
    return hotspots

def pixel_to_lat_lon(pixel_x, pixel_y, tile_x, tile_y, zoom, image_size=(256, 256)):
    """Convert pixel coordinates within a tile to lat/lon coordinates"""
    # Convert tile coordinates back to lat/lon bounds
    n = 2.0 ** zoom
    
    # Get tile bounds
    lon_min = (tile_x / n) * 360.0 - 180.0
    lon_max = ((tile_x + 1) / n) * 360.0 - 180.0
    
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n))))
    
    # Convert pixel coordinates to lat/lon within tile
    # Handle both 2D (width, height) and 3D (height, width, channels) image_size formats
    if len(image_size) == 3:
        # For numpy array shape: (height, width, channels)
        tile_height, tile_width = image_size[0], image_size[1]
    else:
        # For tuple: (width, height)
        tile_width, tile_height = image_size
    
    lon = lon_min + (pixel_x / tile_width) * (lon_max - lon_min)
    lat = lat_max - (pixel_y / tile_height) * (lat_max - lat_min)
    
    return lat, lon

def reverse_geocode_coordinates(lat, lon):
    """Reverse geocode coordinates to get street name and address"""
    try:
        # Use Azure Maps Search API for reverse geocoding
        result = maps_search_client.get_reverse_geocoding(
            coordinates=[lon, lat]  # Note: Azure Maps expects [longitude, latitude]
        )
        
        # Check if result has 'features' structure (like the existing geocoding function)
        if result and 'features' in result and len(result['features']) > 0:
            feature = result['features'][0]
            properties = feature.get('properties', {})
            address = properties.get('address', {})
            
            # Extract street name from addressLine (e.g., "Eerikinkatu 20" -> "Eerikinkatu")
            address_line = address.get('addressLine', '')
            if address_line:
                # Split by space and take the first part (street name)
                street_parts = address_line.split()
                street_name = street_parts[0] if street_parts else 'Unknown Street'
                street_number = ' '.join(street_parts[1:]) if len(street_parts) > 1 else ''
            else:
                # Fallback to intersection baseStreet if available
                intersection = address.get('intersection', {})
                street_name = intersection.get('baseStreet', 'Unknown Street')
                street_number = ''
            
            full_address = address.get('formattedAddress', f"{street_name} {street_number}".strip())
            
            return {
                'street_name': street_name,
                'street_number': street_number,
                'full_address': full_address,
                'confidence': properties.get('confidence', 'Unknown')
            }
        # Check if result has 'addresses' structure (alternative format)
        elif result and 'addresses' in result and len(result['addresses']) > 0:
            address = result['addresses'][0]['address']
            street_name = address.get('streetName', 'Unknown Street')
            street_number = address.get('streetNumber', '')
            full_address = f"{street_number} {street_name}".strip()
            
            return {
                'street_name': street_name,
                'street_number': street_number,
                'full_address': full_address,
                'confidence': result['addresses'][0].get('confidence', 'Unknown')
            }
        else:
            st.write(f"Debug - No valid address data found in response")
            return {
                'street_name': 'Unknown Street',
                'street_number': '',
                'full_address': 'Unknown Address',
                'confidence': 'Low'
            }
    except Exception as e:
        st.warning(f"Reverse geocoding failed for coordinates ({lat}, {lon}): {e}")
        return {
            'street_name': 'Unknown Street',
            'street_number': '',
            'full_address': 'Unknown Address',
            'confidence': 'Error'
        }

def analyze_street_density(hotspots, tile_x, tile_y, zoom, image_size=(256, 256)):
    """Analyze traffic density per street by reverse geocoding hotspots"""
    street_analysis = []
    
    # Limit to top 10 hotspots for debugging and to avoid API limits
    limited_hotspots = hotspots[:10]
    st.write(f"Processing top {len(limited_hotspots)} hotspots for street analysis...")
    
    for i, hotspot in enumerate(limited_hotspots):
        # Convert pixel coordinates to lat/lon
        lat, lon = pixel_to_lat_lon(
            hotspot['center_x'], 
            hotspot['center_y'], 
            tile_x, 
            tile_y, 
            zoom,
            image_size
        )
        
        # Debug: Show coordinate conversion for first few hotspots
        if i < 3:
            st.write(f"Hotspot {i+1}: pixel({hotspot['center_x']}, {hotspot['center_y']}) -> lat/lon({lat:.6f}, {lon:.6f})")
        
        # Reverse geocode to get street name
        geocode_result = reverse_geocode_coordinates(lat, lon)
        
        # Create street analysis entry
        street_entry = {
            'street_name': geocode_result['street_name'],
            'full_address': geocode_result['full_address'],
            'confidence': geocode_result['confidence'],
            'coordinates': {'lat': lat, 'lon': lon},
            'heavy_traffic_cars': hotspot['heavy_count'],
            'moderate_traffic_cars': hotspot['moderate_count'],
            'light_traffic_cars': hotspot['light_count'],
            'total_traffic_pixels': hotspot['total_count'],
            'intensity_score': round(hotspot['intensity_score'], 2),
            'grid_position': f"({hotspot['grid_x']}, {hotspot['grid_y']})"
        }
        
        street_analysis.append(street_entry)
    
    return street_analysis

def estimate_cars_from_traffic_analysis(image_data, zoom_level):
    """Estimate number of cars based on traffic analysis and zoom level"""
    
    # Get the traffic analysis
    analysis = analyze_traffic_tile(image_data)
    
    # Load image to get actual dimensions
    img = Image.open(io.BytesIO(image_data))
    width, height = img.size
    
    # Calibration factors based on zoom level
    # Each zoom level doubles the detail, so pixels_per_car should halve
    # These are rough estimates and would need real-world calibration
    zoom_factors = {
        12: {'pixels_per_car': 100, 'tile_size_km': 2.4},  # ~2.4km per tile
        13: {'pixels_per_car': 80, 'tile_size_km': 1.2},   # ~1.2km per tile
        14: {'pixels_per_car': 60, 'tile_size_km': 0.6},   # ~0.6km per tile
        15: {'pixels_per_car': 40, 'tile_size_km': 0.3},   # ~0.3km per tile
        16: {'pixels_per_car': 30, 'tile_size_km': 0.15},  # ~0.15km per tile
        17: {'pixels_per_car': 20, 'tile_size_km': 0.075}, # ~0.075km per tile
        18: {'pixels_per_car': 15, 'tile_size_km': 0.0375},# ~0.0375km per tile
        19: {'pixels_per_car': 10, 'tile_size_km': 0.01875},# ~0.01875km per tile
        20: {'pixels_per_car': 8, 'tile_size_km': 0.009375},# ~0.009375km per tile
        21: {'pixels_per_car': 6, 'tile_size_km': 0.0046875},# ~0.0046875km per tile
        22: {'pixels_per_car': 4, 'tile_size_km': 0.00234375} # ~0.00234375km per tile
    }
    
    factor = zoom_factors.get(zoom_level, zoom_factors[18])  # Default to zoom 18 if zoom_level not found
    
    # Calculate traffic pixels for each category
    total_pixels = width * height
    heavy_traffic_pixels = (analysis['heavy_traffic'] / 100) * total_pixels
    moderate_traffic_pixels = (analysis['moderate_traffic'] / 100) * total_pixels
    light_traffic_pixels = (analysis['light_traffic'] / 100) * total_pixels
    
    # Estimate cars based on traffic density and zoom level
    # Heavy traffic: more cars per pixel (congested, slower moving)
    # Light traffic: fewer cars per pixel (free flowing, faster moving)
    
    # Adjust pixels_per_car based on traffic type
    # Heavy traffic: cars are closer together (more cars per pixel)
    # Light traffic: cars are more spread out (fewer cars per pixel)
    heavy_pixels_per_car = factor['pixels_per_car'] * 0.8  # More cars in heavy traffic
    moderate_pixels_per_car = factor['pixels_per_car']     # Standard density
    light_pixels_per_car = factor['pixels_per_car'] * 1.2  # Fewer cars in light traffic
    
    heavy_cars = int(heavy_traffic_pixels / heavy_pixels_per_car) if heavy_traffic_pixels > 0 else 0
    moderate_cars = int(moderate_traffic_pixels / moderate_pixels_per_car) if moderate_traffic_pixels > 0 else 0
    light_cars = int(light_traffic_pixels / light_pixels_per_car) if light_traffic_pixels > 0 else 0
    
    total_estimated_cars = heavy_cars + moderate_cars + light_cars
    
    # Calculate confidence based on total traffic coverage
    traffic_coverage = (heavy_traffic_pixels + moderate_traffic_pixels + light_traffic_pixels) / total_pixels
    confidence = min(100, max(20, traffic_coverage * 200))  # 20-100% confidence
    
    return {
        'estimated_total_cars': total_estimated_cars,
        'heavy_traffic_cars': heavy_cars,
        'moderate_traffic_cars': moderate_cars,
        'light_traffic_cars': light_cars,
        'confidence_percentage': round(confidence, 1),
        'traffic_coverage': round(traffic_coverage * 100, 1),
        'zoom_level': zoom_level,
        'image_size': f"{width}x{height}",
        'tile_size_km': factor['tile_size_km'],
        'pixels_per_car': factor['pixels_per_car'],
        'method': 'pixel_density_estimation'
    }

def get_traffic_flow_tile(longitude, latitude, zoom):
    # Try multiple zoom levels to find the best available traffic data
    # Start with higher zoom for precision, fall back to lower zoom if no data
    zoom_levels = [min(zoom + 2, 16), zoom, max(zoom - 2, 8)]
    
    url = "https://atlas.microsoft.com/traffic/flow/tile/png"
    headers = {'subscription-key': subscription_key}
    
    for precision_zoom in zoom_levels:
        # Convert lat/lon to tile coordinates at this zoom level
        x, y = deg2num(latitude, longitude, precision_zoom)
        
        st.write(f"Trying zoom {precision_zoom}, coords ({x}, {y})")
        
        params = {
            'api-version': '1.0',
            'style': 'absolute',
            'zoom': precision_zoom,
            'x': x,
            'y': y,
            'thickness': 15  # Increase line thickness for better visibility
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                # Success! We found traffic data at this zoom level
                from PIL import Image
                import io
                
                image = Image.open(io.BytesIO(response.content))
                
                # For better centering, create a 2x2 grid if we're at a lower zoom
                if precision_zoom <= zoom:
                    # Get 2x2 grid for better centering
                    tiles = []
                    for dy in [0, 1]:
                        for dx in [0, 1]:
                            tiles.append((x + dx, y + dy))
                    
                    # Fetch all 4 tiles
                    tile_images = [image]  # Start with the one we already have
                    for tile_x, tile_y in tiles[1:]:  # Skip the first one we already have
                        tile_params = params.copy()
                        tile_params['x'] = tile_x
                        tile_params['y'] = tile_y
                        tile_params['thickness'] = 15  # Ensure thickness is set for all tiles
                        
                        try:
                            tile_response = requests.get(url, params=tile_params, headers=headers)
                            if tile_response.status_code == 200 and 'image' in tile_response.headers.get('content-type', ''):
                                tile_img = Image.open(io.BytesIO(tile_response.content))
                                tile_images.append(tile_img)
                            else:
                                tile_images.append(Image.new('RGB', (256, 256), (0, 0, 0)))
                        except:
                            tile_images.append(Image.new('RGB', (256, 256), (0, 0, 0)))
                    
                    # Create composite if we have multiple tiles
                    if len(tile_images) > 1:
                        composite = Image.new('RGB', (512, 512), (0, 0, 0))
                        composite.paste(tile_images[0], (0, 0))      # Top-left
                        composite.paste(tile_images[1], (256, 0))    # Top-right  
                        composite.paste(tile_images[2], (0, 256))    # Bottom-left
                        composite.paste(tile_images[3], (256, 256))  # Bottom-right
                        
                        # Calculate precise position within the 2x2 grid for better centering
                        import math
                        n = 2.0 ** precision_zoom
                        lat_rad = math.radians(latitude)
                        
                        # Calculate exact fractional position within the original tile
                        exact_x = (longitude + 180.0) / 360.0 * n
                        exact_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
                        frac_x = exact_x - x
                        frac_y = exact_y - y
                        
                        st.write(f"Precise position within tile: {frac_x:.3f} from left, {frac_y:.3f} from top")
                        
                        # Calculate crop position to center the point
                        # In the 2x2 grid, our point is at position (frac_x, frac_y) in the first tile
                        # We want to center this point in the final 256x256 image
                        crop_size = 256
                        target_center_x = 128  # Center of final image
                        target_center_y = 128
                        
                        # Calculate where to crop from the 512x512 composite
                        left = int(target_center_x + (frac_x - 0.5) * 256)
                        top = int(target_center_y + (frac_y - 0.5) * 256)
                        
                        # Ensure crop area is within bounds
                        left = max(0, min(left, 512 - crop_size))
                        top = max(0, min(top, 512 - crop_size))
                        
                        cropped = composite.crop((left, top, left + crop_size, top + crop_size))
                        st.write(f"Cropping from ({left}, {top}) to center point at ({frac_x:.3f}, {frac_y:.3f})")
                        
                        st.image(cropped, caption=f"Traffic Flow - 2x2 Grid at Zoom {precision_zoom}")
                        st.write(f"Image size: {cropped.size}")
                        
                        # Convert back to bytes
                        img_bytes = io.BytesIO()
                        cropped.save(img_bytes, format='PNG')
                        
                        return {
                            'image_data': img_bytes.getvalue(),
                            'zoom': precision_zoom,
                            'tile_x': x,
                            'tile_y': y
                        }
                
                # Single tile display
                st.image(image, caption=f"Traffic Flow Tile - Zoom {precision_zoom}")
                st.write(f"Image size: {image.size}")
                
                return {
                    'image_data': response.content,
                    'zoom': precision_zoom,
                    'tile_x': x,
                    'tile_y': y
                }
                
            elif response.status_code == 204:
                st.write(f"No traffic data at zoom {precision_zoom}, trying next level...")
                continue
            else:
                st.write(f"Error at zoom {precision_zoom}: {response.status_code}, trying next level...")
                continue
                
        except Exception as e:
            st.write(f"Exception at zoom {precision_zoom}: {str(e)}, trying next level...")
            continue
    
    # If we get here, no zoom level worked
    st.error("No traffic data available at any zoom level")
    st.info("Traffic data may not be available for this location")
    return None


def geocode(write_address, zoom):
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
            st.map(df, zoom=zoom)
            #st.write(f"Coordinates: {longitude}, {latitude}")
            
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
        longitude, latitude = geocode(write_address, zoom)  # Only call once
        if longitude is not None and latitude is not None:
            traffic_result = get_traffic_flow_tile(longitude, latitude, zoom)

            if traffic_result:
                # Extract image data and coordinates
                result = traffic_result['image_data']
                actual_zoom = traffic_result['zoom']
                actual_tile_x = traffic_result['tile_x']
                actual_tile_y = traffic_result['tile_y']
                
                st.write(f"Traffic tile coordinates: zoom={actual_zoom}, tile=({actual_tile_x}, {actual_tile_y})")
                
                # Traffic analysis
                analysis = analyze_traffic_tile(result)
                st.subheader("Traffic Flow Analysis")
                st.dataframe(pd.DataFrame([analysis]))
                
                # Car estimation
                car_estimation = estimate_cars_from_traffic_analysis(result, zoom)
                st.subheader("Car Estimation")
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Cars", car_estimation['estimated_total_cars'])
                with col2:
                    st.metric("Heavy Traffic Cars", car_estimation['heavy_traffic_cars'])
                with col3:
                    st.metric("Moderate Traffic Cars", car_estimation['moderate_traffic_cars'])
                with col4:
                    st.metric("Light Traffic Cars", car_estimation['light_traffic_cars'])
                
                # Display detailed analysis
                st.write(f"**Confidence:** {car_estimation['confidence_percentage']}%")
                st.write(f"**Traffic Coverage:** {car_estimation['traffic_coverage']}% of image")
                st.write(f"**Image Size:** {car_estimation['image_size']} pixels")
                st.write(f"**Zoom Level:** {car_estimation['zoom_level']}")
                
                # Show detailed breakdown
                st.subheader("Detailed Analysis")
                st.dataframe(pd.DataFrame([car_estimation]))
                
                # NEW: Street-level analysis
                st.subheader("🚗 Street-Level Traffic Analysis")
                st.write("Analyzing traffic density per street...")
                
                # Detect traffic hotspots
                hotspots = detect_traffic_hotspots(analysis['pixel_data'], analysis['image_size'])
                st.write(f"Found {len(hotspots)} traffic hotspots")
                
                if hotspots:
                    # Use the actual coordinates from the traffic flow tile
                    # These should match the traffic visualization
                    st.write(f"Using coordinates from traffic tile: zoom={actual_zoom}, tile=({actual_tile_x}, {actual_tile_y})")
                    st.write(f"Image size for coordinate conversion: {analysis['image_size']}")
                    
                    # Debug: Show first hotspot coordinates
                    if hotspots:
                        first_hotspot = hotspots[0]
                        st.write(f"First hotspot pixel coordinates: ({first_hotspot['center_x']}, {first_hotspot['center_y']})")
                    
                    with st.spinner("Reverse geocoding hotspots to get street names..."):
                        street_analysis = analyze_street_density(hotspots, actual_tile_x, actual_tile_y, actual_zoom, analysis['image_size'])
                    
                    if street_analysis:
                        st.success(f"Successfully analyzed {len(street_analysis)} street segments")
                        
                        # Display street analysis results
                        st.subheader("📊 Street Traffic Density")
                        
                        # Create a more detailed dataframe for street analysis
                        street_df = pd.DataFrame(street_analysis)
                        
                        # Display summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Streets Analyzed", len(street_df))
                        with col2:
                            st.metric("Highest Intensity", f"{street_df['intensity_score'].max():.2f}")
                        with col3:
                            st.metric("Avg Intensity", f"{street_df['intensity_score'].mean():.2f}")
                        
                        # Display the street analysis table
                        st.dataframe(street_df, width='stretch')
                        
                        # Show top congested streets
                        st.subheader("🔥 Most Congested Streets")
                        top_streets = street_df.nlargest(5, 'intensity_score')
                        for idx, street in top_streets.iterrows():
                            st.write(f"**{street['street_name']}** - Intensity: {street['intensity_score']} | "
                                   f"Heavy: {street['heavy_traffic_cars']} | "
                                   f"Moderate: {street['moderate_traffic_cars']} | "
                                   f"Light: {street['light_traffic_cars']}")
                        
                        # Optional: Show on map with hover tooltips
                        if st.checkbox("Show street locations on map"):
                            # Use existing street_analysis data
                            map_data = pd.DataFrame([{
                                'lat': street['coordinates']['lat'],
                                'lon': street['coordinates']['lon'],
                                'street': street['street_name'],
                                'intensity': round(street['intensity_score'], 2),
                                'heavy': street['heavy_traffic_cars'],
                                'moderate': street['moderate_traffic_cars'],
                                'light': street['light_traffic_cars']
                            } for street in street_analysis])
                            
                            # Create interactive map with hover tooltips using pydeck
                            import pydeck as pdk
                            
                            # Define the layer with tooltips
                            layer = pdk.Layer(
                                'ScatterplotLayer',
                                map_data,
                                get_position='[lon, lat]',
                                get_color='[255, 68, 68, 200]',  # Red color with transparency
                                get_radius=50,  # Size of markers
                                pickable=True
                            )
                            
                            # Define tooltip content
                            tooltip = {
                                "html": """
                                <b>Street:</b> {street}<br/>
                                <b>Traffic Intensity:</b> {intensity}<br/>
                                <b>Heavy Traffic:</b> {heavy} cars<br/>
                                <b>Moderate Traffic:</b> {moderate} cars<br/>
                                <b>Light Traffic:</b> {light} cars
                                """,
                                "style": {
                                    "backgroundColor": "steelblue",
                                    "color": "white",
                                    "border": "1px solid white",
                                    "borderRadius": "5px",
                                    "padding": "10px"
                                }
                            }
                            
                            # Set view state
                            view_state = pdk.ViewState(
                                latitude=map_data['lat'].mean(),
                                longitude=map_data['lon'].mean(),
                                zoom=zoom,
                                pitch=0
                            )
                            
                            # Create the deck
                            r = pdk.Deck(
                                layers=[layer],
                                initial_view_state=view_state,
                                tooltip=tooltip
                            )
                            
                            # Display the interactive map
                            st.pydeck_chart(r)
                    else:
                        st.warning("No street analysis data available")
                else:
                    st.info("No significant traffic hotspots detected in this area")
    else:
        st.error("Please enter an address")