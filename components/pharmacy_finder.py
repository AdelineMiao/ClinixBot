# pharmacy_finder.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import requests
import json
import random
from translations import translations

class PharmacyFinder:
    def __init__(self, lang="zh"):
        self.api_key = "PHARMACY_API_KEY"  # 这里需要替换为真实的API密钥
        self.lang = lang
        self.t = translations
        
        self.pharmacies = [
            {"name": "CVS Pharmacy", "name_zh": "CVS药房", "address": "123 Main St", "distance": 0.7, "lat": 40.7128, "lng": -74.0060},
            {"name": "Walgreens", "name_zh": "沃尔格林药房", "address": "456 Broadway", "distance": 1.2, "lat": 40.7168, "lng": -74.0030},
            {"name": "Rite Aid", "name_zh": "莱德药房", "address": "789 Park Ave", "distance": 1.8, "lat": 40.7148, "lng": -74.0090},
            {"name": "Duane Reade", "name_zh": "杜安里德药房", "address": "101 Fifth Ave", "distance": 2.1, "lat": 40.7108, "lng": -74.0040},
            {"name": "Target Pharmacy", "name_zh": "塔吉特药房", "address": "202 Madison Ave", "distance": 2.5, "lat": 40.7188, "lng": -74.0070},
        ]
    
    def _get_user_location(self):
        """获取用户位置"""
        # 真实情况下应使用地理定位API
        # 这里使用模拟数据
        return {"lat": 40.7128, "lng": -74.0060}
    
    def _search_nearby_pharmacies(self, location, medication=None, radius=5):
        """搜索附近的药房"""
        # 真实情况下应调用外部API查询附近药房
        # 例如Google Places API或CVS/药房的API
        # 这里使用模拟数据
        
        # 模拟搜索逻辑
        if medication:
            # 过滤有特定药物的药房
            return [p for p in self.pharmacies if random.random() > 0.3]
        
        return self.pharmacies
    
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
    
    def render(self):
        """渲染药房查找界面"""
        t = self.t.pharmacy[self.lang]
        ui = self.t.ui[self.lang]
        
        st.header(t["header"])
        
        # 用户输入
        st.write(t["description"])
        
        # 获取当前诊断中推荐的药物
        medications = []
        if 'recommended_medications' in st.session_state and st.session_state.recommended_medications:
            # 从推荐中提取药物名称
            import re
            text = st.session_state.recommended_medications
            
            # 根据语言选择正则表达式模式
            pattern = r'推荐药物名称[:：]\s*([^\n]+)' if self.lang == "zh" else r'Recommended medication[:：]\s*([^\n]+)'
            med_matches = re.findall(pattern, text)
            if med_matches:
                medications = [med.strip() for med in med_matches[0].split(',')]
        
        # 药物输入/选择
        if medications:
            empty_option = "" if self.lang == "zh" else ""
            medication = st.selectbox(t["select_medication"], [empty_option] + medications)
        else:
            medication = st.text_input(t["enter_medication"])
        
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider(t["search_radius"], 1, 20, 5)
        with col2:
            sort_by = st.selectbox(t["sort_by"], t["sort_options"])
        
        # 搜索按钮
        if st.button(ui["search_button"]):
            with st.spinner(t["searching"]):
                # 获取用户位置
                user_location = self._get_user_location()
                
                # 搜索药房
                pharmacies = self._search_nearby_pharmacies(user_location, medication, radius)
                
                if pharmacies:
                    st.success(t["found_pharmacies"].format(len(pharmacies)))
                    
                    # 显示地图
                    st.subheader(t["pharmacy_map"])
                    m = self._create_pharmacy_map(user_location, pharmacies)
                    folium_static(m)
                    
                    # 显示药房列表
                    st.subheader(t["pharmacy_list"])
                    
                    # 根据选择的排序方式排序
                    if sort_by == t["sort_options"][0]:  # 距离/Distance
                        pharmacies = sorted(pharmacies, key=lambda x: x["distance"])
                    
                    # 显示药房信息
                    for i, pharmacy in enumerate(pharmacies):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # 根据语言选择显示的药房名称
                                pharmacy_name = pharmacy["name_zh"] if self.lang == "zh" else pharmacy["name"]
                                
                                st.write(f"#### {i+1}. {pharmacy_name}")
                                st.write(f"{t['address']} {pharmacy['address']}")
                                st.write(t["distance_km"].format(pharmacy['distance']))
                            with col2:
                                if st.button(ui["navigate_button"], key=f"nav_{i}"):
                                    st.info(t["navigation_started"].format(pharmacy_name))
                                
                                if st.button(ui["order_button"], key=f"order_{i}"):
                                    if medication:
                                        st.success(t["added_to_cart"].format(medication, pharmacy_name))
                                    else:
                                        st.info(t["select_medication_first"])
                            
                            st.divider()
                else:
                    st.error(t["no_pharmacies"].format(radius, medication if medication else ""))
                    st.info(t["suggestion"])