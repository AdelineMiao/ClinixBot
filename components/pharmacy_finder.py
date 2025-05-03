# pharmacy_finder.py
import math
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import requests
import json
import random
from math import sin, cos, sqrt, atan2, radians
from translations import translations

class PharmacyFinder:
    def __init__(self, lang="zh"):
        self.api_key = "PHARMACY_API_KEY"  # 这里需要替换为真实的API密钥
        self.lang = lang
        self.t = translations
        
        # Default location
        default_lat = 41.306023
        default_lng = -72.925615
        
        # Initial pharmacy data
        self.base_pharmacies = [
            {"name": "CVS Pharmacy", "name_zh": "CVS药房", "address": "123 Main St", "distance": 0.7, "lat": 40.7128, "lng": -74.0060},
            {"name": "Walgreens", "name_zh": "沃尔格林药房", "address": "456 Broadway", "distance": 1.2, "lat": 40.7168, "lng": -74.0030},
            {"name": "Rite Aid", "name_zh": "莱德药房", "address": "789 Park Ave", "distance": 1.8, "lat": 40.7148, "lng": -74.0090},
            {"name": "Duane Reade", "name_zh": "杜安里德药房", "address": "101 Fifth Ave", "distance": 2.1, "lat": 40.7108, "lng": -74.0040},
            {"name": "Target Pharmacy", "name_zh": "塔吉特药房", "address": "202 Madison Ave", "distance": 2.5, "lat": 40.7188, "lng": -74.0070},
        ]
        
        # Generate more pharmacies around the default location
        self.pharmacies = self.base_pharmacies + self._generate_nearby_pharmacies(default_lat, default_lng, 15)
    
    def _get_user_location(self):
        """获取用户位置"""
        t = self.t.pharmacy[self.lang]
        ui = self.t.ui[self.lang]
        
        # Add new entries to translation dictionaries if they don't exist
        if "your_location" not in t:
            t["your_location"] = "您当前的位置:" if self.lang == "zh" else "Your current location:"
        if "requesting_location" not in ui:
            ui["requesting_location"] = "正在请求您的位置..." if self.lang == "zh" else "Requesting your location..."
        if "location_acquired" not in ui:
            ui["location_acquired"] = "已成功获取您的位置!" if self.lang == "zh" else "Location acquired successfully!"
        if "location_default" not in ui:
            ui["location_default"] = "使用默认位置。" if self.lang == "zh" else "Using default location."
        if "manual_location" not in t:
            t["manual_location"] = "手动输入位置" if self.lang == "zh" else "Enter location manually"
        if "latitude" not in t:
            t["latitude"] = "纬度" if self.lang == "zh" else "Latitude"
        if "longitude" not in t:
            t["longitude"] = "经度" if self.lang == "zh" else "Longitude"
        if "use_location" not in t:
            t["use_location"] = "使用此位置" if self.lang == "zh" else "Use this location"
            
        # Check if location is in URL parameters
        try:
            query_params = st.experimental_get_query_params()
            
            if 'lat' in query_params and 'lng' in query_params:
                try:
                    lat = float(query_params['lat'][0])
                    lng = float(query_params['lng'][0])
                    st.session_state.user_location = {"lat": lat, "lng": lng}
                    st.success(ui["location_acquired"])
                except:
                    # Invalid parameters, use default
                    if 'user_location' not in st.session_state:

                        st.session_state.user_location = {"lat": 41.306023, "lng": -72.925615}
                        st.warning(ui["location_default"])
            elif 'user_location' not in st.session_state:
                # No location in URL or session state, use default
                st.session_state.user_location = {"lat": 41.306023, "lng": -72.925615}
                st.warning(ui["location_default"])
                
                # Add location sharing button
                location_html = """
                <button onclick="getLocation()" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 10px 0;">
                    %s
                </button>
                
                <script>
                function getLocation() {
                    if (navigator.geolocation) {
                        navigator.geolocation.getCurrentPosition(showPosition, showError);
                    } else {
                        alert("Geolocation is not supported by this browser.");
                    }
                }
                
                function showPosition(position) {
                    // Create URL with location parameters
                    var baseUrl = window.location.href.split('?')[0];
                    var url = baseUrl + "?lat=" + position.coords.latitude + "&lng=" + position.coords.longitude;
                    
                    // Redirect to the same page with location parameters
                    window.location.href = url;
                }
                
                function showError(error) {
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            alert("User denied the request for Geolocation.");
                            break;
                        case error.POSITION_UNAVAILABLE:
                            alert("Location information is unavailable.");
                            break;
                        case error.TIMEOUT:
                            alert("The request to get user location timed out.");
                            break;
                        case error.UNKNOWN_ERROR:
                            alert("An unknown error occurred.");
                            break;
                    }
                }
                </script>
                """ % ("分享我的位置" if self.lang == "zh" else "Share My Location")
                
                from streamlit.components.v1 import html
                html(location_html, height=60)
        except:
            # If experimental_get_query_params fails, use default location
            if 'user_location' not in st.session_state:
                st.session_state.user_location = {"lat": 41.306023, "lng": -72.925615}
        
        return st.session_state.user_location
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the distance between two coordinates using the Haversine formula"""
        # Approximate radius of earth in km
        R = 6371.0
        
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c
        return round(distance, 1)  # Round to 1 decimal place
    
    def _search_nearby_pharmacies(self, user_location, medication=None, radius=5):
        """Search for nearby pharmacies using OpenStreetMap's Overpass API"""
        
        # Convert radius from km to meters
        radius_meters = radius * 1000
        
        # Construct Overpass query to find pharmacies within radius
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        node["amenity"="pharmacy"]
        (around:{radius_meters},{user_location['lat']},{user_location['lng']});
        out body;
        """
        
        try:
            # Make the request to Overpass API
            response = requests.get(overpass_url, params={"data": overpass_query})
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Process the results
                pharmacies = []
                
                for element in data.get('elements', []):
                    # Extract pharmacy data
                    tags = element.get('tags', {})
                    
                    pharmacy = {
                        "name": tags.get('name', 'Unnamed Pharmacy'),
                        "name_zh": tags.get('name:zh', tags.get('name', '未命名药房')),
                        "address": self._format_address(tags),
                        "lat": element.get('lat'),
                        "lng": element.get('lon'),
                    }
                    
                    # Calculate distance from user
                    pharmacy["distance"] = self._calculate_distance(
                        user_location["lat"], user_location["lng"],
                        pharmacy["lat"], pharmacy["lng"]
                    )
                    
                    pharmacies.append(pharmacy)
                
                # Sort by distance
                pharmacies = sorted(pharmacies, key=lambda x: x["distance"])
                
                # Filter by medication if provided (note: real medication data would require another API)
                if medication and pharmacies:
                    # In a real application, you would need to use a pharmaceutical database API
                    # For now, we'll assume all pharmacies might have the medication
                    return pharmacies
                
                return pharmacies
            else:
                st.error(f"Error from Overpass API: {response.status_code}")
                # Fall back to simulated data
                return self._simulate_places_api_results(user_location, radius)
        except Exception as e:
            st.error(f"Error accessing Overpass API: {e}")
            # Fall back to simulated data
            return self._simulate_places_api_results(user_location, radius)
        
    def _format_address(self, tags):
        """Format the address from OSM tags"""
        address_parts = []
        
        # Try to build an address from available tags
        if 'addr:housenumber' in tags:
            address_parts.append(tags['addr:housenumber'])
        
        if 'addr:street' in tags:
            address_parts.append(tags['addr:street'])
        
        if 'addr:city' in tags:
            address_parts.append(tags['addr:city'])
        
        # If we have a formatted address, use it
        if address_parts:
            return " ".join(address_parts)
        
        # Otherwise, use the description if available
        if 'description' in tags:
            return tags['description']
        
        # Last resort
        return "Address unavailable"

    def _simulate_places_api_results(self, user_location, radius):
            """Simulate Google Places API results for nearby pharmacies"""
            import random
            
            # Number of pharmacies to generate based on radius
            # Larger radius = more pharmacies, but with diminishing returns
            count = min(int(radius * 2), 20)  # Cap at 20 pharmacies
            
            # Common pharmacy chains in the US
            chains = [
                {"name": "CVS Pharmacy", "name_zh": "CVS药房"},
                {"name": "Walgreens", "name_zh": "沃尔格林药房"},
                {"name": "Rite Aid", "name_zh": "莱德药房"},
                {"name": "Duane Reade", "name_zh": "杜安里德药房"},
                {"name": "Target Pharmacy", "name_zh": "塔吉特药房"},
                {"name": "Walmart Pharmacy", "name_zh": "沃尔玛药房"},
                {"name": "Costco Pharmacy", "name_zh": "好市多药房"},
                {"name": "Kroger Pharmacy", "name_zh": "克罗格药房"},
                {"name": "Medicine Shoppe", "name_zh": "药物商店"},
                {"name": "Health Mart", "name_zh": "健康药房"}
            ]
            
            pharmacies = []
            
            for i in range(count):
                # Generate a random distance within the radius
                # More pharmacies closer to the user, fewer at the edges
                distance = random.uniform(0, 1) * radius
                
                # Convert distance and random angle to lat/lng
                # This creates a rough circle of points around the user location
                angle = random.uniform(0, 360)
                lat_offset = distance * 0.009 * math.cos(math.radians(angle))
                lng_offset = distance * 0.011 * math.sin(math.radians(angle))
                
                lat = user_location["lat"] + lat_offset
                lng = user_location["lng"] + lng_offset
                
                # Pick a random pharmacy chain
                chain = random.choice(chains)
                
                # Generate a realistic address
                street_num = random.randint(100, 999)
                streets = ["Main St", "Park Ave", "Oak St", "Elm St", "Washington Ave", 
                        "Broadway", "Market St", "Church St", "High St", "Center St"]
                street = random.choice(streets)
                
                pharmacy = {
                    "name": chain["name"],
                    "name_zh": chain["name_zh"],
                    "address": f"{street_num} {street}",
                    "distance": round(distance, 1),
                    "lat": lat,
                    "lng": lng
                }
                
                pharmacies.append(pharmacy)
            
            # Sort by distance
            pharmacies = sorted(pharmacies, key=lambda x: x["distance"])
            
            return pharmacies
    def _get_address_from_coordinates(self, lat, lng):
        """Get address from coordinates using Nominatim reverse geocoding"""
        
        nominatim_url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lng,
            "format": "json",
            "zoom": 18,  # Building level zoom
            "addressdetails": 1
        }
        
        try:
            # Add a small delay to respect Nominatim usage policy
            import time
            time.sleep(1)  # 1 second delay between requests
            
            response = requests.get(nominatim_url, params=params, headers={"User-Agent": "PharmacyFinder/1.0"})
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract the formatted address
                if "display_name" in data:
                    return data["display_name"]
                
            return "Address unavailable"
        except Exception as e:
            print(f"Error retrieving address: {e}")
            return "Address unavailable"
    def _get_pharmacy_details(self, pharmacy_id):
        """Get detailed information about a pharmacy using OSM API"""
        
        osm_api_url = f"https://api.openstreetmap.org/api/0.6/node/{pharmacy_id}.json"
        
        try:
            response = requests.get(osm_api_url)
            if response.status_code == 200:
                data = response.json()
                # Process and return the detailed data
                return data
            else:
                return None
        except:
            return None


    def _generate_nearby_pharmacies(self, center_lat, center_lng, count=10):
        """Generate random pharmacy data points near the given center coordinates"""
        import random
        
        # Pharmacy name templates
        pharmacy_names_en = ["Community Pharmacy", "Health Plus", "MediCare", "QuickRx", "Family Pharmacy", 
                            "City Drugs", "Wellness Pharmacy", "Express Meds", "Care Pharmacy", "ProHealth"]
        
        pharmacy_names_zh = ["社区药房", "健康加", "医保药房", "快速药房", "家庭药房", 
                            "城市药房", "健康药房", "快捷药房", "关爱药房", "专业健康"]
        
        pharmacies = []
        
        # Generate random pharmacies within approximately 5km
        for i in range(count):
            # Random offset (roughly within 5km)
            lat_offset = (random.random() - 0.5) * 0.09  # Roughly 5km in latitude
            lng_offset = (random.random() - 0.5) * 0.11  # Roughly 5km in longitude
            
            lat = center_lat + lat_offset
            lng = center_lng + lng_offset
            
            # Calculate initial distance (will be recalculated later)
            distance = self._calculate_distance(center_lat, center_lng, lat, lng)
            
            # Create pharmacy data
            name_index = i % len(pharmacy_names_en)
            pharmacy = {
                "name": f"{pharmacy_names_en[name_index]}",
                "name_zh": f"{pharmacy_names_zh[name_index]}",
                "address": f"{random.randint(100, 999)} {['Main', 'Park', 'Oak', 'Cedar', 'Pine'][i % 5]} St",
                "distance": distance,
                "lat": lat,
                "lng": lng
            }
            
            pharmacies.append(pharmacy)
        
        return pharmacies
    
    def _create_pharmacy_map(self, user_location, pharmacies):
        """创建药房地图"""
        # 创建地图对象
        m = folium.Map(location=[user_location["lat"], user_location["lng"]], zoom_start=14)
        
        # 添加用户位置标记
        your_location = "您的位置" if self.lang == "zh" else "Your Location"
        folium.Marker(
            [user_location["lat"], user_location["lng"]],
            popup=your_location,
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)
        
        # 添加药房标记
        for pharmacy in pharmacies:
            # 根据语言选择显示的药房名称
            pharmacy_name = pharmacy["name_zh"] if self.lang == "zh" else pharmacy["name"]
            distance_text = f"距离: {pharmacy['distance']}公里" if self.lang == "zh" else f"Distance: {pharmacy['distance']} km"
            
            folium.Marker(
                [pharmacy["lat"], pharmacy["lng"]],
                popup=f"{pharmacy_name}<br>{pharmacy['address']}<br>{distance_text}",
                icon=folium.Icon(color="blue", icon="plus-sign")
            ).add_to(m)
        
        return m
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the distance between two coordinates using the Haversine formula"""
        from math import sin, cos, sqrt, atan2, radians
        
        # Approximate radius of earth in km
        R = 6371.0
        
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c
        return round(distance, 1)  # Round to 1 decimal place
    def _create_pharmacy_map_html(self, user_location, pharmacies):
        """Create HTML for a Leaflet map with pharmacies"""
        
        # Create HTML for a Leaflet map
        map_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <style>
                #map {{height: 500px;}}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([{user_location['lat']}, {user_location['lng']}], 13);
                
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }}).addTo(map);
                
                // Add user location marker
                L.marker([{user_location['lat']}, {user_location['lng']}], {{
                    icon: L.icon({{
                        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
                        iconSize: [25, 41],
                        iconAnchor: [12, 41]
                    }})
                }}).addTo(map).bindPopup('Your Location');
                
                // Add pharmacy markers
        """
        
        # Add a marker for each pharmacy
        for pharmacy in pharmacies:
            pharmacy_name = pharmacy["name_zh"] if self.lang == "zh" else pharmacy["name"]
            map_html += f"""
                L.marker([{pharmacy['lat']}, {pharmacy['lng']}]).addTo(map)
                    .bindPopup('{pharmacy_name}<br>{pharmacy['address']}<br>Distance: {pharmacy['distance']} km');
            """
        
        map_html += """
            </script>
        </body>
        </html>
        """
        
        return map_html
    def render(self):
        """渲染药房查找界面"""
        t = self.t.pharmacy[self.lang]
        ui = self.t.ui[self.lang]
        
        st.header(t["header"])
        
        # 用户输入
        st.write(t["description"])
        
        # Add translations for map-related text if needed
        if "click_map" not in t:
            t["click_map"] = "点击地图设置您的位置" if self.lang == "zh" else "Click on the map to set your location"
        if "location_set" not in t:
            t["location_set"] = "位置已设置为" if self.lang == "zh" else "Location set to"
        if "your_location" not in t:
            t["your_location"] = "您当前的位置" if self.lang == "zh" else "Your current location"
        
        # Initialize user location from session state or use default
        if 'user_location' not in st.session_state:
            st.session_state.user_location = {"lat": 41.306023, "lng": -72.925615}
        
        user_location = st.session_state.user_location
        
        # Create a map for location selection
        st.write(t["click_map"])
        
        # Create the selection map centered on the current user location
        from streamlit_folium import st_folium
        
        # Create a map for location selection
        selection_map = folium.Map(
            location=[user_location["lat"], user_location["lng"]], 
            zoom_start=13
        )
        
        # Add a marker for the current location
        folium.Marker(
            [user_location["lat"], user_location["lng"]],
            popup=t["your_location"],
            icon=folium.Icon(color="red", icon="home")
        ).add_to(selection_map)
        
        # Display the map and get click data
        map_data = st_folium(selection_map, width=700, height=400)
        
        # Check if a location was clicked - FIXED: Added proper None and key checking
        if map_data is not None and isinstance(map_data, dict) and 'last_clicked' in map_data and map_data['last_clicked'] is not None:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']
            
            # Update the user location
            st.session_state.user_location = {"lat": clicked_lat, "lng": clicked_lng}
            user_location = st.session_state.user_location
            
            # Show a success message
            st.success(f"{t['location_set']}: {clicked_lat:.6f}, {clicked_lng:.6f}")
        
        # Display current coordinates
        st.caption(f"Lat: {user_location['lat']:.6f}, Lng: {user_location['lng']:.6f}")
        
        # Get recommended medications from session state
        medications = []
        if 'recommended_medications' in st.session_state and st.session_state.recommended_medications:
            # Extract medication names from recommendations
            import re
            text = st.session_state.recommended_medications
            
            # Choose regex pattern based on language
            pattern = r'推荐药物名称[:：]\s*([^\n]+)' if self.lang == "zh" else r'Recommended medication[:：]\s*([^\n]+)'
            med_matches = re.findall(pattern, text)
            if med_matches:
                medications = [med.strip() for med in med_matches[0].split(',')]
        
        # Medication input/selection
        st.markdown("---")
        if medications:
            empty_option = "" if self.lang == "zh" else ""
            medication = st.selectbox(t["select_medication"], [empty_option] + medications)
        else:
            medication = st.text_input(t["enter_medication"])
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider(t["search_radius"], 1, 20, 5)
        with col2:
            sort_by = st.selectbox(t["sort_by"], t["sort_options"])
        
        # Search button
        if st.button(ui["search_button"]):
            with st.spinner(t["searching"]):
                # Search pharmacies
                pharmacies = self._search_nearby_pharmacies(user_location, medication, radius)
                
                if pharmacies:
                    st.success(t["found_pharmacies"].format(len(pharmacies)))
                    
                    # Display map
                    st.subheader(t["pharmacy_map"])
                    m = self._create_pharmacy_map(user_location, pharmacies)
                    folium_static(m)
                    
                    # Display pharmacy list
                    st.subheader(t["pharmacy_list"])
                    
                    # Sort based on selected option
                    if sort_by == t["sort_options"][0]:  # Distance
                        pharmacies = sorted(pharmacies, key=lambda x: x["distance"])
                    # Add more sorting options if needed
                    
                    # Show pharmacy information
                    for i, pharmacy in enumerate(pharmacies):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Choose display name based on language
                                pharmacy_name = pharmacy["name_zh"] if self.lang == "zh" else pharmacy["name"]
                                
                                st.write(f"#### {i+1}. {pharmacy_name}")
                                st.write(f"{t['address']} {pharmacy['address']}")
                                st.write(t["distance_km"].format(pharmacy['distance']))
                            with col2:
                                if st.button(ui["navigate_button"], key=f"nav_{i}"):
                                    # In a real app, this would launch navigation
                                    st.info(t["navigation_started"].format(pharmacy_name))
                                    
                                    # Uncomment for production to open maps in new tab
                                    # import webbrowser
                                    # maps_url = f"https://www.google.com/maps/dir/?api=1&destination={pharmacy['lat']},{pharmacy['lng']}"
                                    # webbrowser.open_new_tab(maps_url)
                                
                                if st.button(ui["order_button"], key=f"order_{i}"):
                                    if medication:
                                        st.success(t["added_to_cart"].format(medication, pharmacy_name))
                                    else:
                                        st.info(t["select_medication_first"])
                            
                            st.divider()
                else:
                    st.error(t["no_pharmacies"].format(radius, medication if medication else ""))
                    st.info(t["suggestion"])
    def render_map(self, user_location, pharmacies):
        """Render the map in Streamlit"""
        
        map_html = self._create_pharmacy_map_html(user_location, pharmacies)
        
        from streamlit.components.v1 import html
        html(map_html, height=500)
