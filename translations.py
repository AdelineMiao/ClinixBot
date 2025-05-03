# translations.py
# 包含应用程序的所有翻译文本

class TranslationDict:
    def __init__(self):
        # 通用UI元素
        self.ui = {
            "zh": {
                "language_selector": "选择语言:",
                "search_button": "搜索",
                "navigate_button": "导航",
                "order_button": "下单",
                "book_button": "预约",
                "yes": "是",
                "no": "否",
                "distance": "距离",
                "rating": "评分",
                "price": "价格",
                "available_beds": "可用床位",
                "wait_time": "等待时间",
                "generate_report_button": "📝 生成分析报告",
                "continue_button": "继续补充信息",
                "footer_disclaimer": "免责声明：本系统仅提供初步诊断参考，不能替代专业医生的诊断和治疗建议。如有严重症状，请立即就医。"
            },
            "en": {
                "language_selector": "Select Language:",
                "search_button": "Search",
                "navigate_button": "Navigate",
                "order_button": "Order",
                "book_button": "Book",
                "yes": "Yes",
                "no": "No",
                "distance": "Distance",
                "rating": "Rating",
                "price": "Price",
                "available_beds": "Available Beds",
                "wait_time": "Wait Time",
                "generate_report_button": "📝 Generate Report",
                "continue_button": "Continue Describing",
                "footer_disclaimer": "Disclaimer: This system only provides preliminary diagnostic reference and cannot replace the diagnosis and treatment advice of a professional doctor. If you have serious symptoms, please seek medical attention immediately."
            }
        }
        
        # 主应用程序和导航菜单
        self.app = {
            "zh": {
                "app_title": "ClinixBot - 智能医疗诊断助手",
                "app_subtitle": "描述您的症状，获取诊断和用药建议",
                "nav_menu": "导航菜单",
                "select_feature": "选择功能:",
                "chat_diagnosis": "💬 聊天诊断",
                "medical_data": "📊 医疗数据分析",
                "find_pharmacy": "💊 查找药房",
                "find_hospital": "🏥 查找医院",
                "copyright": "© 2025 ClinixBot - 智能医疗诊断系统"
            },
            "en": {
                "app_title": "ClinixBot - Intelligent Medical Diagnostic Assistant",
                "app_subtitle": "Describe your symptoms, get diagnosis and medication recommendations",
                "nav_menu": "Navigation Menu",
                "select_feature": "Select Feature:",
                "chat_diagnosis": "💬 Chat Diagnosis",
                "medical_data": "📊 Medical Data Analysis",
                "find_pharmacy": "💊 Find Pharmacy",
                "find_hospital": "🏥 Find Hospital",
                "copyright": "© 2025 ClinixBot - Intelligent Medical Diagnostic System"
            }
        }
        
        # 聊天界面
        self.chat = {
            "zh": {
                "header": "💬 智能诊断助手",
                "input_placeholder": "请描述您的症状:",
                "initial_greeting1": "👋 您好！我是ClinixBot，您的智能医疗助手。请告诉我您的症状，我将为您提供初步诊断。",
                "initial_greeting2": "欢迎使用ClinixBot！我可以帮助您了解可能的健康问题。请描述您的症状。",
                "initial_greeting3": "您好！我是ClinixBot医疗助手。请详细描述您的症状，我会尽力提供帮助。",
                "analyzing": "ClinixBot正在分析您的症状...",
                "generating_recommendations": "正在生成用药建议...",
                "view_recommendations": "查看药物推荐",
                "find_nearby_pharmacy": "查找附近药房",
                "find_nearby_hospital": "查找附近医院"
            },
            "en": {
                "header": "💬 Intelligent Diagnostic Assistant",
                "input_placeholder": "Please describe your symptoms:",
                "initial_greeting1": "👋 Hello! I'm ClinixBot, your intelligent medical assistant. Please tell me your symptoms, and I will provide you with a preliminary diagnosis.",
                "initial_greeting2": "Welcome to ClinixBot! I can help you understand potential health issues. Please describe your symptoms.",
                "initial_greeting3": "Hello! I'm ClinixBot medical assistant. Please describe your symptoms in detail, and I will do my best to help.",
                "analyzing": "ClinixBot is analyzing your symptoms...",
                "generating_recommendations": "Generating medication recommendations...",
                "view_recommendations": "View Medication Recommendations",
                "find_nearby_pharmacy": "Find Nearby Pharmacies",
                "find_nearby_hospital": "Find Nearby Hospitals"
            }
        }
        
        # 医院查找功能
        self.hospital = {
            "zh": {
                "header": "🏥 查找附近医院",
                "description": "查找附近医院和紧急护理中心",
                "select_department": "选择医院科室:",
                "search_radius": "搜索半径(公里):",
                "beds_only": "仅显示有可用床位的医院",
                "sort_by": "排序方式:",
                "searching": "正在查找附近医院...",
                "found_hospitals": "找到 {} 家附近医院",
                "hospital_map": "附近医院地图",
                "hospital_list": "医院列表",
                "address": "地址:",
                "distance_km": "距离: {} 公里",
                "wait_time_min": "等待时间: {}",
                "available_beds": "可用床位: {}",
                "navigation_started": "已开始导航至 {}",
                "appointment_success": "已为您在 {} 预约挂号",
                "no_hospitals": "在半径 {}公里内未找到符合条件的医院",
                "suggestion": "请尝试增加搜索半径或更改科室选择",
                "departments": ["所有科室", "急诊科", "内科", "外科", "儿科", "妇产科", "神经科", "心脏科", "骨科", "眼科", "皮肤科"],
                "sort_options": ["距离", "等待时间", "可用床位"]
            },
            "en": {
                "header": "🏥 Find Nearby Hospitals",
                "description": "Find nearby hospitals and urgent care centers",
                "select_department": "Select Hospital Department:",
                "search_radius": "Search Radius (km):",
                "beds_only": "Show only hospitals with available beds",
                "sort_by": "Sort by:",
                "searching": "Searching for nearby hospitals...",
                "found_hospitals": "Found {} nearby hospitals",
                "hospital_map": "Nearby Hospitals Map",
                "hospital_list": "Hospital List",
                "address": "Address:",
                "distance_km": "Distance: {} km",
                "wait_time_min": "Wait Time: {}",
                "available_beds": "Available Beds: {}",
                "navigation_started": "Navigation started to {}",
                "appointment_success": "Successfully booked an appointment at {}",
                "no_hospitals": "No hospitals found within {} km radius that meet the criteria",
                "suggestion": "Please try increasing the search radius or changing the department selection",
                "departments": ["All Departments", "Emergency", "Internal Medicine", "Surgery", "Pediatrics", "Obstetrics & Gynecology", "Neurology", "Cardiology", "Orthopedics", "Ophthalmology", "Dermatology"],
                "sort_options": ["Distance", "Wait Time", "Available Beds"]
            }
        }
        
        # 药房查找功能
        self.pharmacy = {
            "zh": {
                "header": "💊 查找附近药房",
                "description": "查找附近可提供您所需药物的药房",
                "enter_medication": "输入需要购买的药物名称(可选):",
                "select_medication": "选择需要购买的药物:",
                "search_radius": "搜索半径(公里):",
                "sort_by": "排序方式:",
                "searching": "正在查找附近药房...",
                "found_pharmacies": "找到 {} 家附近药房",
                "pharmacy_map": "附近药房地图",
                "pharmacy_list": "药房列表",
                "address": "地址:",
                "distance_km": "距离: {} 公里",
                "navigation_started": "已开始导航至 {}",
                "added_to_cart": "已将 {} 添加到 {} 的购物车",
                "select_medication_first": "请先选择药物",
                "no_pharmacies": "在半径 {}公里内未找到提供{}的药房",
                "suggestion": "请尝试增加搜索半径或更改药物名称",
                "sort_options": ["距离", "评分", "价格"]
            },
            "en": {
                "header": "💊 Find Nearby Pharmacies",
                "description": "Find nearby pharmacies that provide the medication you need",
                "enter_medication": "Enter medication name (optional):",
                "select_medication": "Select medication to purchase:",
                "search_radius": "Search Radius (km):",
                "sort_by": "Sort by:",
                "searching": "Searching for nearby pharmacies...",
                "found_pharmacies": "Found {} nearby pharmacies",
                "pharmacy_map": "Nearby Pharmacies Map",
                "pharmacy_list": "Pharmacy List",
                "address": "Address:",
                "distance_km": "Distance: {} km",
                "navigation_started": "Navigation started to {}",
                "added_to_cart": "Added {} to {} shopping cart",
                "select_medication_first": "Please select a medication first",
                "no_pharmacies": "No pharmacies found within {} km radius that provide {}",
                "suggestion": "Please try increasing the search radius or changing the medication name",
                "sort_options": ["Distance", "Rating", "Price"]
            }
        }
        
        # 数据分析仪表盘
        self.dashboard = {
            "zh": {
                "header": "📊 医疗数据分析仪表盘",
                "total_patients": "患者总数",
                "avg_bill": "平均账单金额",
                "avg_stay": "平均住院天数",
                "disease_count": "疾病种类数",
                "disease_tab": "疾病分布",
                "billing_tab": "账单分析",
                "condition_distribution": "Top 10 医疗条件分布",
                "patient_count": "患者数量",
                "medical_condition": "医疗条件",
                "avg_bill_by_condition": "各医疗条件的平均账单金额",
                "avg_bill_amount": "平均账单金额 ($)"
            },
            "en": {
                "header": "📊 Medical Data Analysis Dashboard",
                "total_patients": "Total Patients",
                "avg_bill": "Average Bill Amount",
                "avg_stay": "Average Stay Duration",
                "disease_count": "Disease Count",
                "disease_tab": "Disease Distribution",
                "billing_tab": "Billing Analysis",
                "condition_distribution": "Top 10 Medical Conditions Distribution",
                "patient_count": "Patient Count",
                "medical_condition": "Medical Condition",
                "avg_bill_by_condition": "Average Bill Amount by Medical Condition",
                "avg_bill_amount": "Average Bill Amount ($)"
            }
        }

# 创建翻译字典的全局实例
translations = TranslationDict()