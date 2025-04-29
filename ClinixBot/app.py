#app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from components.chat_interface import ChatInterface
from components.visualization_dashboard import VisualizationDashboard
from components.pharmacy_finder import PharmacyFinder
from components.hospital_finder import HospitalFinder
from utils.data_processor import DataProcessor
from models.rag_model import MedicalRAGModel
from translations import translations

# 加载环境变量
load_dotenv()

# 配置页面
st.set_page_config(
    page_title="ClinixBot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主色调：蓝白配色方案 */
    :root {
        --primary-color: #4B89DC;
        --secondary-color: #EBF5FF;
        --text-color: #333333;
        --light-text: #6B7C93;
    }
    
    /* 全局样式 */
    .main {
        background-color: white;
        color: var(--text-color);
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 700;
    }
    
    /* 部件样式 */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #3A70C3;
    }
    
    /* 聊天容器 */
    .chat-container {
        background-color: var(--secondary-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* 卡片样式 */
    .css-1r6slb0 {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(75, 137, 220, 0.1);
    }
    
    /* 边栏样式 */
    .css-1d391kg {
        background-color: var(--secondary-color);
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_diagnosis' not in st.session_state:
    st.session_state.current_diagnosis = None

if 'recommended_medications' not in st.session_state:
    st.session_state.recommended_medications = []

if 'active_view' not in st.session_state:
    st.session_state.active_view = "chat"

# 初始化语言选择（默认为中文）
if 'language' not in st.session_state:
    st.session_state.language = "zh"

# 加载模型和数据
@st.cache_resource
def load_rag_model():
    return MedicalRAGModel()

@st.cache_data
def load_medical_data():
    data = pd.read_csv("data/hospital_records_2021_2024_with_bills.csv")
    processor = DataProcessor()
    return processor.preprocess_data(data)

# 加载数据和模型
rag_model = load_rag_model()
medical_data = load_medical_data()

# 获取当前语言的翻译
lang = st.session_state.language
t = translations

# 侧边栏菜单
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=ClinixBot", width=150)
    st.title(t.app[lang]["nav_menu"])
    
    # 语言选择器
    selected_language = st.selectbox(
        t.ui[lang]["language_selector"],
        options=["中文", "English"],
        index=0 if lang == "zh" else 1
    )
    
    # 根据选择更新语言
    if (selected_language == "中文" and lang != "zh") or (selected_language == "English" and lang != "en"):
        # 更新语言
        st.session_state.language = "zh" if selected_language == "中文" else "en"
        
        # 当语言变更时，清除聊天历史，这样会重新生成对应语言的欢迎语
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []
            
        st.rerun()  # 重新运行应用以应用新的语言设置
    
    # 导航菜单
    selected_view = st.radio(
        t.app[lang]["select_feature"],
        [
            t.app[lang]["chat_diagnosis"], 
            t.app[lang]["medical_data"], 
            t.app[lang]["find_pharmacy"], 
            t.app[lang]["find_hospital"]
        ]
    )
    
    if selected_view == t.app[lang]["chat_diagnosis"]:
        st.session_state.active_view = "chat"
    elif selected_view == t.app[lang]["medical_data"]:
        st.session_state.active_view = "data"
    elif selected_view == t.app[lang]["find_pharmacy"]:
        st.session_state.active_view = "pharmacy"
    elif selected_view == t.app[lang]["find_hospital"]:
        st.session_state.active_view = "hospital"
    
    st.divider()
    st.caption(t.app[lang]["copyright"])

# 应用标题
st.title(t.app[lang]["app_title"])
st.subheader(t.app[lang]["app_subtitle"])

# 主内容区域
if st.session_state.active_view == "chat":
    chat_interface = ChatInterface(rag_model, lang)
    chat_interface.render()
    
elif st.session_state.active_view == "data":
    visualization_dashboard = VisualizationDashboard(medical_data, lang)
    visualization_dashboard.render()
    
elif st.session_state.active_view == "pharmacy":
    pharmacy_finder = PharmacyFinder(lang)
    pharmacy_finder.render()
    
elif st.session_state.active_view == "hospital":
    hospital_finder = HospitalFinder(lang)
    hospital_finder.render()

# 添加页脚
st.markdown("---")
st.caption(t.ui[lang]["footer_disclaimer"])