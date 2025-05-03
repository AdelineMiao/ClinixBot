# hospital_finder.py
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

class HospitalFinder:
    def __init__(self, lang="zh"):
        self.api_key = "HOSPITAL_API_KEY"  # 需要替换为实际API密钥
        self.lang = lang
        self.t = translations
        
        # Default location
        default_lat = 41.306023
        default_lng = -72.925615
        
        # Initial hospital data
        self.base_hospitals = [
            {"name": "General Hospital", "name_zh": "综合医院", "address": "123 Health St", "distance": 1.2, "lat": 40.7120, "lng": -74.0050, "beds_available": 5, "specialty": "综合医院", "specialty_en": "General Hospital", "wait_time": "30分钟", "wait_time_en": "30 minutes"},
            {"name": "City Medical Center", "name_zh": "城市医疗中心", "address": "456 Care Ave", "distance": 2.3, "lat": 40.7150, "lng": -74.0080, "beds_available": 2, "specialty": "急诊中心", "specialty_en": "Emergency Center", "wait_time": "45分钟", "wait_time_en": "45 minutes"},
            {"name": "University Hospital", "name_zh": "大学医院", "address": "789 Research Blvd", "distance": 3.1, "lat": 40.7180, "lng": -74.0020, "beds_available": 8, "specialty": "教学医院", "specialty_en": "Teaching Hospital", "wait_time": "15分钟", "wait_time_en": "15 minutes"},
            {"name": "Children's Hospital", "name_zh": "儿童医院", "address": "101 Pediatric Way", "distance": 3.5, "lat": 40.7100, "lng": -74.0100, "beds_available": 3, "specialty": "儿科医院", "specialty_en": "Pediatric Hospital", "wait_time": "20分钟", "wait_time_en": "20 minutes"},
            {"name": "Community Health Center", "name_zh": "社区健康中心", "address": "202 Wellness Dr", "distance": 1.8, "lat": 40.7140, "lng": -74.0070, "beds_available": 0, "specialty": "社区医疗", "specialty_en": "Community Health", "wait_time": "60分钟", "wait_time_en": "60 minutes"},
        ]
        
        # Generate more hospitals around the default location
        self.hospitals = self.base_hospitals + self._generate_nearby_hospitals(default_lat, default_lng, 15)
    
    def _get_user_location(self):
        """Gets the user's location"""
        t = self.t.hospital[self.lang]
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
            query_params = st.query_params
            
            if 'lat' in query_params and 'lng' in query_params:
                try:
                    lat = float(query_params['lat'])
                    lng = float(query_params['lng'])
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
            # If query_params fails, use default location
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
    
    def _search_nearby_hospitals(self, user_location, specialty=None, radius=10):
        """Search for nearby hospitals"""
        # Try to use API if available
        if hasattr(self, 'api_key') and self.api_key and self.api_key != "HOSPITAL_API_KEY":
            # Here you would implement the API call using self.api_key
            # For now, we'll use the simulated implementation
            pass
        
        # Update distances based on current user location
        for hospital in self.hospitals:
            hospital["distance"] = self._calculate_distance(
                user_location["lat"], user_location["lng"],
                hospital["lat"], hospital["lng"]
            )
        
        # Filter hospitals by distance
        within_radius = [h for h in self.hospitals if h["distance"] <= radius]
        
        # Filter by specialty if provided
        departments = self.t.hospital[self.lang]["departments"]
        if specialty and specialty != departments[0]:  # Not "All Departments"
            # Filter by specialty
            if self.lang == "zh":
                filtered_hospitals = [h for h in within_radius if specialty.lower() in h["specialty"].lower()]
            else:
                filtered_hospitals = [h for h in within_radius if specialty.lower() in h["specialty_en"].lower()]
            return filtered_hospitals if filtered_hospitals else within_radius
        
        return within_radius
    
    def _generate_nearby_hospitals(self, center_lat, center_lng, count=10):
        """Generate random hospital data points near the given center coordinates"""
        import random
        
        # Hospital name templates
        hospital_names_en = ["Central Hospital", "Regional Medical", "Memorial Hospital", "Urgent Care", "Family Hospital", 
                           "Metro Hospital", "Health Center", "Care Center", "Medical Center", "Community Hospital"]
        
        hospital_names_zh = ["中心医院", "区域医疗中心", "纪念医院", "紧急护理中心", "家庭医院", 
                           "都市医院", "健康中心", "护理中心", "医疗中心", "社区医院"]
        
        # Specialty types
        specialties_en = ["General Hospital", "Emergency Center", "Cardiology", "Orthopedics", 
                       "Pediatrics", "Neurology", "Oncology", "Obstetrics", "Internal Medicine", "Surgery"]
        
        specialties_zh = ["综合医院", "急诊中心", "心脏科", "骨科", 
                       "儿科", "神经科", "肿瘤科", "产科", "内科", "外科"]
        
        hospitals = []
        
        # Generate random hospitals within approximately 10km
        for i in range(count):
            # Random offset (roughly within 10km)
            lat_offset = (random.random() - 0.5) * 0.18  # Roughly 10km in latitude
            lng_offset = (random.random() - 0.5) * 0.22  # Roughly 10km in longitude
            
            lat = center_lat + lat_offset
            lng = center_lng + lng_offset
            
            # Calculate initial distance
            distance = self._calculate_distance(center_lat, center_lng, lat, lng)
            
            # Random specialty index
            specialty_index = random.randint(0, len(specialties_en) - 1)
            
            # Random wait time (10-90 minutes)
            wait_time_minutes = random.randint(1, 9) * 10
            
            # Random bed availability (0-15)
            beds_available = max(0, int(random.gauss(5, 3)))
            
            # Create hospital data
            name_index = i % len(hospital_names_en)
            hospital = {
                "name": f"{hospital_names_en[name_index]}",
                "name_zh": f"{hospital_names_zh[name_index]}",
                "address": f"{random.randint(100, 999)} {['Medical', 'Health', 'Hospital', 'Care', 'Center'][i % 5]} St",
                "distance": distance,
                "lat": lat,
                "lng": lng,
                "beds_available": beds_available,
                "specialty": specialties_zh[specialty_index],
                "specialty_en": specialties_en[specialty_index],
                "wait_time": f"{wait_time_minutes}分钟",
                "wait_time_en": f"{wait_time_minutes} minutes"
            }
            
            hospitals.append(hospital)
        
        return hospitals
    
    def _create_hospital_map(self, user_location, hospitals):
        """Create a map with hospitals"""
        # Create map object
        m = folium.Map(location=[user_location["lat"], user_location["lng"]], zoom_start=13)
        
        # Add user location marker
        your_location = "您的位置" if self.lang == "zh" else "Your Location"
        folium.Marker(
            [user_location["lat"], user_location["lng"]],
            popup=your_location,
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)
        
        # Add hospital markers
        for hospital in hospitals:
            # Choose color based on bed availability
            color = "green" if hospital["beds_available"] > 0 else "orange"
            
            # Choose display based on language
            hospital_name = hospital["name_zh"] if self.lang == "zh" else hospital["name"]
            specialty = hospital["specialty"] if self.lang == "zh" else hospital["specialty_en"]
            wait_time = hospital["wait_time"] if self.lang == "zh" else hospital["wait_time_en"]
            
            beds_text = f"可用床位: {hospital['beds_available']}" if self.lang == "zh" else f"Available Beds: {hospital['beds_available']}"
            wait_text = f"预计等待时间: {wait_time}" if self.lang == "zh" else f"Estimated Wait Time: {wait_time}"
            distance_text = f"距离: {hospital['distance']}公里" if self.lang == "zh" else f"Distance: {hospital['distance']} km"
            
            folium.Marker(
                [hospital["lat"], hospital["lng"]],
                popup=f"{hospital_name}<br>{specialty}<br>{beds_text}<br>{wait_text}<br>{distance_text}",
                icon=folium.Icon(color=color, icon="plus-sign")
            ).add_to(m)
        
        return m
    
    def _create_hospital_map_html(self, user_location, hospitals):
        """Create HTML for a Leaflet map with hospitals"""
        
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
                
                // Add hospital markers
        """
        
        # Add a marker for each hospital
        for hospital in hospitals:
            hospital_name = hospital["name_zh"] if self.lang == "zh" else hospital["name"]
            specialty = hospital["specialty"] if self.lang == "zh" else hospital["specialty_en"]
            wait_time = hospital["wait_time"] if self.lang == "zh" else hospital["wait_time_en"]
            
            # Choose marker color based on bed availability
            marker_color = "green" if hospital["beds_available"] > 0 else "orange"
            icon_url = f"https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-{marker_color}.png"
            
            popup_content = f"{hospital_name}<br>{specialty}<br>Distance: {hospital['distance']} km<br>Wait: {wait_time}<br>Beds: {hospital['beds_available']}"
            # Escape single quotes in the popup content
            popup_content = popup_content.replace("'", "\\'")
            
            map_html += f"""
                L.marker([{hospital['lat']}, {hospital['lng']}], {{
                    icon: L.icon({{
                        iconUrl: '{icon_url}',
                        iconSize: [25, 41],
                        iconAnchor: [12, 41]
                    }})
                }}).addTo(map).bindPopup('{popup_content}');
            """
        
        map_html += """
            </script>
        </body>
        </html>
        """
        
        return map_html
    
    def render_map(self, user_location, hospitals):
        """Render the map in Streamlit"""
        
        map_html = self._create_hospital_map_html(user_location, hospitals)
        
        from streamlit.components.v1 import html
        html(map_html, height=500)
    
    def render(self):
        """Render the hospital finder interface"""
        t = self.t.hospital[self.lang]
        ui = self.t.ui[self.lang]
        
        st.header(t["header"])
        
        # User input
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
        
        # Check if a location was clicked - with proper None and key checking
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
        
        # Get current medical condition from diagnosis if available
        medical_condition = None
        if 'current_diagnosis' in st.session_state and st.session_state.current_diagnosis:
            # Extract medical condition from diagnosis
            import re
            text = st.session_state.current_diagnosis
            
            # Choose regex pattern based on language
            pattern = r'初步诊断[:：]\s*([^\n]+)' if self.lang == "zh" else r'Preliminary diagnosis[:：]\s*([^\n]+)'
            condition_match = re.search(pattern, text)
            if condition_match:
                medical_condition = condition_match.group(1).strip()
        
        # Department selection
        specialties = t["departments"]
        
        # If there's a medical condition, recommend appropriate department
        recommended_specialty = None
        if medical_condition:
            # Simple condition-to-specialty mapping
            condition_to_specialty_zh = {
                "感冒": "内科",
                "流感": "内科",
                "骨折": "骨科",
                "心脏病": "心脏科",
                "头痛": "神经科",
                "皮疹": "皮肤科",
                "眼睛": "眼科",
                "儿童": "儿科"
            }
            
            condition_to_specialty_en = {
                "cold": "Internal Medicine",
                "flu": "Internal Medicine",
                "fracture": "Orthopedics",
                "heart": "Cardiology",
                "headache": "Neurology",
                "rash": "Dermatology",
                "eye": "Ophthalmology",
                "children": "Pediatrics"
            }
            
            condition_to_specialty = condition_to_specialty_zh if self.lang == "zh" else condition_to_specialty_en
            specialties_list = t["departments"]
            
            for condition, specialty in condition_to_specialty.items():
                if condition.lower() in medical_condition.lower() and specialty in specialties_list:
                    recommended_specialty = specialty
                    break
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            selected_specialty = st.selectbox(
                t["select_department"],
                specialties,
                index=specialties.index(recommended_specialty) if recommended_specialty else 0
            )
        with col2:
            radius = st.slider(t["search_radius"], 1, 30, 10)
        
        col3, col4 = st.columns(2)
        with col3:
            beds_filter = st.checkbox(t["beds_only"], value=True)
        with col4:
            sort_by = st.selectbox(t["sort_by"], t["sort_options"])
        
        # Search button
        if st.button(ui["search_button"]):
            with st.spinner(t["searching"]):
                # Get user location
                user_location = self._get_user_location()
                
                # Search hospitals
                hospitals = self._search_nearby_hospitals(user_location, selected_specialty, radius)
                
                # Apply filters
                if beds_filter:
                    hospitals = [h for h in hospitals if h["beds_available"] > 0]
                
                # Apply sorting
                sort_options = t["sort_options"]
                if sort_by == sort_options[0]:  # Distance
                    hospitals = sorted(hospitals, key=lambda x: x["distance"])
                elif sort_by == sort_options[1]:  # Wait Time
                    if self.lang == "zh":
                        hospitals = sorted(hospitals, key=lambda x: int(x["wait_time"].replace("分钟", "")))
                    else:
                        hospitals = sorted(hospitals, key=lambda x: int(x["wait_time_en"].replace(" minutes", "")))
                elif sort_by == sort_options[2]:  # Available Beds
                    hospitals = sorted(hospitals, key=lambda x: x["beds_available"], reverse=True)
                
                if hospitals:
                    st.success(t["found_hospitals"].format(len(hospitals)))
                    
                    # Display map
                    st.subheader(t["hospital_map"])
                    m = self._create_hospital_map(user_location, hospitals)
                    folium_static(m)
                    
                    # Display hospital list
                    st.subheader(t["hospital_list"])
                    
                    # Show hospital information
                    for i, hospital in enumerate(hospitals):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Choose display name based on language
                                hospital_name = hospital["name_zh"] if self.lang == "zh" else hospital["name"]
                                specialty = hospital["specialty"] if self.lang == "zh" else hospital["specialty_en"]
                                wait_time = hospital["wait_time"] if self.lang == "zh" else hospital["wait_time_en"]
                                
                                st.write(f"#### {i+1}. {hospital_name} ({specialty})")
                                st.write(f"{t['address']} {hospital['address']}")
                                st.write(f"{t['distance_km'].format(hospital['distance'])} | {t['wait_time_min'].format(wait_time)}")
                                st.write(f"{t['available_beds'].format(hospital['beds_available'])}")
                            with col2:
                                if st.button(ui["navigate_button"], key=f"hosp_nav_{i}"):
                                    st.info(t["navigation_started"].format(hospital_name))
                                    
                                    # Uncomment for production to open maps in new tab
                                    # import webbrowser
                                    # maps_url = f"https://www.google.com/maps/dir/?api=1&destination={hospital['lat']},{hospital['lng']}"
                                    # webbrowser.open_new_tab(maps_url)
                                
                                if st.button(ui["book_button"], key=f"hosp_book_{i}"):
                                    st.success(t["appointment_success"].format(hospital_name))
                            
                            st.divider()
                else:
                    st.error(t["no_hospitals"].format(radius))
                    st.info(t["suggestion"])
