# hospital_finder.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import requests
import json
import random
from translations import translations

class HospitalFinder:
    def __init__(self, lang="zh"):
        self.api_key = "HOSPITAL_API_KEY"  # 需要替换为实际API密钥
        self.lang = lang
        self.t = translations
        
        self.hospitals = [
            {"name": "General Hospital", "name_zh": "综合医院", "address": "123 Health St", "distance": 1.2, "lat": 40.7120, "lng": -74.0050, "beds_available": 5, "specialty": "综合医院", "specialty_en": "General Hospital", "wait_time": "30分钟", "wait_time_en": "30 minutes"},
            {"name": "City Medical Center", "name_zh": "城市医疗中心", "address": "456 Care Ave", "distance": 2.3, "lat": 40.7150, "lng": -74.0080, "beds_available": 2, "specialty": "急诊中心", "specialty_en": "Emergency Center", "wait_time": "45分钟", "wait_time_en": "45 minutes"},
            {"name": "University Hospital", "name_zh": "大学医院", "address": "789 Research Blvd", "distance": 3.1, "lat": 40.7180, "lng": -74.0020, "beds_available": 8, "specialty": "教学医院", "specialty_en": "Teaching Hospital", "wait_time": "15分钟", "wait_time_en": "15 minutes"},
            {"name": "Children's Hospital", "name_zh": "儿童医院", "address": "101 Pediatric Way", "distance": 3.5, "lat": 40.7100, "lng": -74.0100, "beds_available": 3, "specialty": "儿科医院", "specialty_en": "Pediatric Hospital", "wait_time": "20分钟", "wait_time_en": "20 minutes"},
            {"name": "Community Health Center", "name_zh": "社区健康中心", "address": "202 Wellness Dr", "distance": 1.8, "lat": 40.7140, "lng": -74.0070, "beds_available": 0, "specialty": "社区医疗", "specialty_en": "Community Health", "wait_time": "60分钟", "wait_time_en": "60 minutes"},
        ]
    
    def _get_user_location(self):
        """获取用户位置"""
        # 真实情况下应使用地理定位API
        # 这里使用模拟数据
        return {"lat": 40.7128, "lng": -74.0060}
    
    def _search_nearby_hospitals(self, location, specialty=None, radius=10):
        """搜索附近的医院"""
        # 真实情况下应调用外部API查询附近医院
        # 这里使用模拟数据
        
        # 模拟搜索逻辑
        departments = self.t.hospital[self.lang]["departments"]
        if specialty and specialty != departments[0]:  # 不是"所有科室"/"All Departments"
            # 过滤指定科室的医院
            if self.lang == "zh":
                filtered_hospitals = [h for h in self.hospitals if specialty.lower() in h["specialty"].lower()]
            else:
                filtered_hospitals = [h for h in self.hospitals if specialty.lower() in h["specialty_en"].lower()]
            return filtered_hospitals if filtered_hospitals else self.hospitals
        
        return self.hospitals
    
    def _create_hospital_map(self, user_location, hospitals):
        """创建医院地图"""
        # 创建地图对象
        m = folium.Map(location=[user_location["lat"], user_location["lng"]], zoom_start=13)
        
        # 添加用户位置标记
        your_location = "您的位置" if self.lang == "zh" else "Your Location"
        folium.Marker(
            [user_location["lat"], user_location["lng"]],
            popup=your_location,
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)
        
        # 添加医院标记
        for hospital in hospitals:
            # 根据床位可用性选择颜色
            color = "green" if hospital["beds_available"] > 0 else "orange"
            
            # 根据语言选择显示内容
            hospital_name = hospital["name_zh"] if self.lang == "zh" else hospital["name"]
            specialty = hospital["specialty"] if self.lang == "zh" else hospital["specialty_en"]
            wait_time = hospital["wait_time"] if self.lang == "zh" else hospital["wait_time_en"]
            
            beds_text = f"可用床位: {hospital['beds_available']}" if self.lang == "zh" else f"Available Beds: {hospital['beds_available']}"
            wait_text = f"预计等待时间: {wait_time}" if self.lang == "zh" else f"Estimated Wait Time: {wait_time}"
            
            folium.Marker(
                [hospital["lat"], hospital["lng"]],
                popup=f"{hospital_name}<br>{specialty}<br>{beds_text}<br>{wait_text}",
                icon=folium.Icon(color=color, icon="plus-sign")
            ).add_to(m)
        
        return m
    
    def render(self):
        """渲染医院查找界面"""
        t = self.t.hospital[self.lang]
        ui = self.t.ui[self.lang]
        
        st.header(t["header"])
        
        # 用户输入
        st.write(t["description"])
        
        # 获取当前诊断的医疗条件
        medical_condition = None
        if 'current_diagnosis' in st.session_state and st.session_state.current_diagnosis:
            # 从诊断中提取医疗条件
            import re
            text = st.session_state.current_diagnosis
            
            # 根据语言选择正则表达式模式
            pattern = r'初步诊断[:：]\s*([^\n]+)' if self.lang == "zh" else r'Preliminary diagnosis[:：]\s*([^\n]+)'
            condition_match = re.search(pattern, text)
            if condition_match:
                medical_condition = condition_match.group(1).strip()
        
        # 科室选择
        specialties = t["departments"]
        
        # 如果有医疗条件，推荐相应科室
        recommended_specialty = None
        if medical_condition:
            # 简单的条件-科室映射逻辑
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
        
        # 搜索按钮
        if st.button(ui["search_button"]):
            with st.spinner(t["searching"]):
                # 获取用户位置
                user_location = self._get_user_location()
                
                # 搜索医院
                hospitals = self._search_nearby_hospitals(user_location, selected_specialty, radius)
                
                # 应用筛选器
                if beds_filter:
                    hospitals = [h for h in hospitals if h["beds_available"] > 0]
                
                # 应用排序
                sort_options = t["sort_options"]
                if sort_by == sort_options[0]:  # 距离/Distance
                    hospitals = sorted(hospitals, key=lambda x: x["distance"])
                elif sort_by == sort_options[1]:  # 等待时间/Wait Time
                    if self.lang == "zh":
                        hospitals = sorted(hospitals, key=lambda x: int(x["wait_time"].replace("分钟", "")))
                    else:
                        hospitals = sorted(hospitals, key=lambda x: int(x["wait_time_en"].replace(" minutes", "")))
                elif sort_by == sort_options[2]:  # 可用床位/Available Beds
                    hospitals = sorted(hospitals, key=lambda x: x["beds_available"], reverse=True)
                
                if hospitals:
                    st.success(t["found_hospitals"].format(len(hospitals)))
                    
                    # 显示地图
                    st.subheader(t["hospital_map"])
                    m = self._create_hospital_map(user_location, hospitals)
                    folium_static(m)
                    
                    # 显示医院列表
                    st.subheader(t["hospital_list"])
                    
                    # 显示医院信息
                    for i, hospital in enumerate(hospitals):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # 根据语言选择显示名称
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
                                
                                if st.button(ui["book_button"], key=f"hosp_book_{i}"):
                                    st.success(t["appointment_success"].format(hospital_name))
                            
                            st.divider()
                else:
                    st.error(t["no_hospitals"].format(radius))
                    st.info(t["suggestion"])