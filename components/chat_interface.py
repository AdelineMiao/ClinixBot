# chat_interface.py
import streamlit as st
import time
import random
import os
import pandas as pd
from translations import translations
from models.fine_tuned_model import FineTunedMedicalModel

class ChatInterface:
    def __init__(self, rag_model, lang="zh"):
        self.rag_model = rag_model
        self.lang = lang
        self.t = translations
        
        # 初始化并加载微调模型
        self._initialize_fine_tuned_model()
    
    def _initialize_fine_tuned_model(self):
        """初始化并加载微调模型"""
        # 检查是否已经初始化了微调模型
        if 'fine_tuned_model' not in st.session_state:
            # 检查是否有已经训练好的模型ID
            model_id = st.session_state.get('fine_tuned_model_id', None)
            
            # 创建微调模型实例
            st.session_state.fine_tuned_model = FineTunedMedicalModel(model_name=model_id)
            
            # 尝试加载数据集并准备训练数据
            dataset_path = "data/hospital_records_2021_2024_with_bills.csv"
            if os.path.exists(dataset_path):
                try:
                    st.session_state.fine_tuned_model.training_data = st.session_state.fine_tuned_model.prepare_training_data(dataset_path)
                    # 设置标志表示已加载数据集
                    st.session_state.dataset_loaded = True
                except Exception as e:
                    st.error(f"加载数据集失败: {str(e)}" if self.lang == "zh" else f"Failed to load dataset: {str(e)}")
                    st.session_state.dataset_loaded = False
    
    def _get_avatar(self, is_user):
        """获取头像图标"""
        return "👤" if is_user else "🏥"
    
    def _add_message(self, message, is_user=True, metadata=None):
        """添加消息到聊天历史，可选择性地添加元数据"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 创建消息对象，包含可选的元数据
        msg_obj = {
            "message": message,
            "is_user": is_user,
            "timestamp": time.time(),
            "lang": self.lang  # 保存消息语言
        }
        
        # 如果提供了元数据，添加到消息对象
        if metadata:
            msg_obj["metadata"] = metadata
        
        st.session_state.chat_history.append(msg_obj)
    
    def _get_initial_greeting(self):
        """获取基于当前语言的初始欢迎消息"""
        greetings = [
            self.t.chat[self.lang]["initial_greeting1"],
            self.t.chat[self.lang]["initial_greeting2"],
            self.t.chat[self.lang]["initial_greeting3"]
        ]
        return random.choice(greetings)
    
    def _handle_user_input(self):
        """处理用户输入"""
        user_input = st.session_state.user_input
        
        if user_input:
            # 添加用户消息
            self._add_message(user_input, is_user=True)
            
            # 清除输入框
            st.session_state.user_input = ""
            
            # 检查是否为特殊命令
            if user_input.startswith("/train") and 'dataset_loaded' in st.session_state and st.session_state.dataset_loaded:
                # 提取训练轮次参数
                parts = user_input.split()
                epochs = 3  # 默认轮次
                if len(parts) > 1 and parts[1].isdigit():
                    epochs = int(parts[1])
                
                # 开始训练流程
                train_message = f"正在开始训练模型(轮次={epochs})..." if self.lang == "zh" else f"Starting model training (epochs={epochs})..."
                self._add_message(train_message, is_user=False)
                
                # 执行训练
                try:
                    with st.spinner(train_message):
                        response = st.session_state.fine_tuned_model.train_model(
                            hyperparameters={"n_epochs": epochs}
                        )
                    
                    if isinstance(response, str) and response.startswith("Error"):
                        self._add_message(response, is_user=False)
                    else:
                        success_msg = f"微调任务创建成功！作业ID: {response.id}\n完成后，您的模型ID将是: ft-{response.id}" if self.lang == "zh" else f"Fine-tuning job created successfully! Job ID: {response.id}\nOnce completed, your model ID will be: ft-{response.id}"
                        self._add_message(success_msg, is_user=False)
                        
                        # 保存作业ID
                        st.session_state.fine_tune_job_id = response.id
                except Exception as e:
                    error_msg = f"训练过程出错: {str(e)}" if self.lang == "zh" else f"Error during training: {str(e)}"
                    self._add_message(error_msg, is_user=False)
                
                return
            
            elif user_input.startswith("/status") and 'fine_tune_job_id' in st.session_state:
                # 检查训练状态
                status_message = f"正在检查训练状态..." if self.lang == "zh" else f"Checking training status..."
                self._add_message(status_message, is_user=False)
                
                try:
                    with st.spinner(status_message):
                        status = st.session_state.fine_tuned_model.check_training_status()
                    
                    if isinstance(status, str):
                        self._add_message(status, is_user=False)
                    else:
                        status_info = f"状态: {status.status}\n进度: {status.training_progress or '未知'}" if self.lang == "zh" else f"Status: {status.status}\nProgress: {status.training_progress or 'unknown'}"
                        self._add_message(status_info, is_user=False)
                except Exception as e:
                    error_msg = f"检查状态时出错: {str(e)}" if self.lang == "zh" else f"Error checking status: {str(e)}"
                    self._add_message(error_msg, is_user=False)
                
                return
            
            elif user_input.startswith("/load"):
                # 加载微调模型
                parts = user_input.split()
                if len(parts) > 1:
                    model_id = parts[1]
                    try:
                        st.session_state.fine_tuned_model.load_fine_tuned_model(model_id)
                        success_msg = f"已加载微调模型: {model_id}" if self.lang == "zh" else f"Fine-tuned model loaded: {model_id}"
                        self._add_message(success_msg, is_user=False)
                        
                        # 保存模型ID
                        st.session_state.fine_tuned_model_id = model_id
                        # 切换到使用微调模型
                        st.session_state.use_fine_tuned = True
                    except Exception as e:
                        error_msg = f"加载模型时出错: {str(e)}" if self.lang == "zh" else f"Error loading model: {str(e)}"
                        self._add_message(error_msg, is_user=False)
                else:
                    help_msg = "/load 命令需要模型ID参数。例如: /load ft-your-model-id" if self.lang == "zh" else "/load command requires model ID parameter. Example: /load ft-your-model-id"
                    self._add_message(help_msg, is_user=False)
                
                return
            
            elif user_input.startswith("/help"):
                # 显示帮助信息
                help_info = """
                可用命令:
                /train [轮次] - 使用已加载的数据集训练模型 (例如: /train 3)
                /status - 检查当前训练任务的状态
                /load [模型ID] - 加载一个已训练的模型 (例如: /load ft-abc123)
                /help - 显示此帮助信息
                """ if self.lang == "zh" else """
                Available Commands:
                /train [epochs] - Train model using loaded dataset (e.g.: /train 3)
                /status - Check status of current training job
                /load [model_id] - Load a trained model (e.g.: /load ft-abc123)
                /help - Show this help information
                """
                self._add_message(help_info, is_user=False)
                return
            
            # 检查是否为对追问的回答
            is_followup_response = 'awaiting_followup' in st.session_state and st.session_state.awaiting_followup
            
            if is_followup_response:
                # 合并原始症状和追问回答
                original_input = st.session_state.original_input
                combined_input = f"{original_input}\n\n追问回答: {user_input}" if self.lang == "zh" else f"{original_input}\n\nFollow-up answer: {user_input}"
                
                # 重置追问状态
                st.session_state.awaiting_followup = False
                
                # 检查是否使用微调模型
                if st.session_state.get('use_fine_tuned', False) and 'fine_tuned_model_id' in st.session_state:
                    # 使用微调模型进行诊断
                    with st.spinner(self.t.chat[self.lang]["analyzing"]):
                        diagnosis = st.session_state.fine_tuned_model.get_diagnosis(combined_input, lang=self.lang)
                        diagnosis_result = {"diagnosis": diagnosis, "model_type": "fine_tuned"}
                else:
                    # 使用RAG模型进行诊断
                    with st.spinner(self.t.chat[self.lang]["analyzing"]):
                        diagnosis_result = self.rag_model.get_diagnosis(
                            combined_input, 
                            lang=self.lang,
                            use_fine_tuned_override=st.session_state.get('use_fine_tuned', False)
                        )
                
                # 保存当前诊断
                st.session_state.current_diagnosis = diagnosis_result["diagnosis"]
                
                # 添加机器人响应，包含模型类型元数据
                self._add_message(diagnosis_result["diagnosis"], is_user=False, metadata={
                    "model_type": diagnosis_result.get("model_type", "unknown")
                })
                
                # 如果诊断包含结果，获取药物推荐
                diagnosis_key = "初步诊断" if self.lang == "zh" else "Preliminary diagnosis"
                if diagnosis_key in diagnosis_result["diagnosis"]:
                    with st.spinner(self.t.chat[self.lang]["generating_recommendations"]):
                        medications = self.rag_model.get_medication_recommendations(diagnosis_result["diagnosis"], lang=self.lang)
                        st.session_state.recommended_medications = medications
            else:
                # 正常流程：获取诊断结果
                # 检查是否使用微调模型
                if st.session_state.get('use_fine_tuned', False) and 'fine_tuned_model_id' in st.session_state:
                    # 使用微调模型进行诊断
                    with st.spinner(self.t.chat[self.lang]["analyzing"]):
                        diagnosis = st.session_state.fine_tuned_model.get_diagnosis(user_input, lang=self.lang)
                        diagnosis_result = {"diagnosis": diagnosis, "model_type": "fine_tuned"}
                else:
                    # 使用RAG模型进行诊断
                    with st.spinner(self.t.chat[self.lang]["analyzing"]):
                        diagnosis_result = self.rag_model.get_diagnosis(
                            user_input, 
                            lang=self.lang,
                            use_fine_tuned_override=st.session_state.get('use_fine_tuned', False)
                        )
                
                # 保存当前诊断
                st.session_state.current_diagnosis = diagnosis_result["diagnosis"]
                
                # 添加机器人响应，包含模型类型元数据
                self._add_message(diagnosis_result["diagnosis"], is_user=False, metadata={
                    "model_type": diagnosis_result.get("model_type", "unknown")
                })
                
                # 保存原始输入用于追问
                st.session_state.original_input = user_input
                
                # 生成追问
                with st.spinner(self.t.chat[self.lang]["generating_followup"]):
                    followup_question = self.rag_model.get_follow_up_question(user_input, lang=self.lang)
                    
                    # 在聊天中添加追问
                    if followup_question:
                        followup_prefix = "追问：" if self.lang == "zh" else "Follow-up question: "
                        self._add_message(f"{followup_prefix}{followup_question}", is_user=False)
                        
                        # 设置等待追问回答状态
                        st.session_state.awaiting_followup = True
                
                # 如果诊断包含结果，获取药物推荐
                diagnosis_key = "初步诊断" if self.lang == "zh" else "Preliminary diagnosis"
                if diagnosis_key in diagnosis_result["diagnosis"] and not st.session_state.awaiting_followup:
                    with st.spinner(self.t.chat[self.lang]["generating_recommendations"]):
                        medications = self.rag_model.get_medication_recommendations(diagnosis_result["diagnosis"], lang=self.lang)
                        st.session_state.recommended_medications = medications
    
    def render(self):
        """渲染聊天界面"""
        t = self.t.chat[self.lang]
        
        # 模型设置区
        with st.sidebar.expander("模型设置" if self.lang == "zh" else "Model Settings", expanded=True):
            # 数据集状态
            dataset_status = "✅ 已加载" if st.session_state.get('dataset_loaded', False) else "❌ 未加载"
            dataset_label = f"数据集状态: {dataset_status}" if self.lang == "zh" else f"Dataset Status: {dataset_status}"
            st.write(dataset_label)
            
            # 微调模型状态
            model_id = st.session_state.get('fine_tuned_model_id', None)
            if model_id:
                model_status = f"✅ 已加载: {model_id}"
            else:
                model_status = "❌ 未加载"
            model_label = f"微调模型: {model_status}" if self.lang == "zh" else f"Fine-tuned Model: {model_status}"
            st.write(model_label)
            
            # 使用微调模型的切换 - 添加唯一的key参数
            use_fine_tuned = st.checkbox(
                "使用微调模型" if self.lang == "zh" else "Use Fine-tuned Model",
                value=st.session_state.get('use_fine_tuned', False),
                disabled=not model_id,
                key="use_fine_tuned_checkbox"  # 添加唯一key
            )
            
            if use_fine_tuned != st.session_state.get('use_fine_tuned', False):
                st.session_state.use_fine_tuned = use_fine_tuned
                
                # 显示状态变化消息
                if use_fine_tuned:
                    st.success("已切换到微调模型" if self.lang == "zh" else "Switched to fine-tuned model")
                else:
                    st.success("已切换到RAG模型" if self.lang == "zh" else "Switched to RAG model")
        
        # 如果需要，初始化聊天历史
        if 'chat_history' not in st.session_state or not st.session_state.chat_history:
            initial_greeting = self._get_initial_greeting()
            
            # 添加命令帮助提示
            command_help = "\n\n输入 /help 查看可用命令。" if self.lang == "zh" else "\n\nType /help to see available commands."
            initial_greeting += command_help
            
            self._add_message(initial_greeting, is_user=False)
            
            # 初始化追问状态
            if 'awaiting_followup' not in st.session_state:
                st.session_state.awaiting_followup = False
        
        # 聊天容器
        chat_container = st.container()
        
        with chat_container:
            # 显示头部
            st.header(t.get("header", "AI Medical Assistant"))
            
            # 显示聊天历史
            for message in st.session_state.chat_history:
                with st.chat_message(name="user" if message["is_user"] else "assistant", avatar=self._get_avatar(message["is_user"])):
                    st.write(message["message"])
                    
                    # 如果有元数据并且不是用户消息，显示模型类型标签
                    if not message["is_user"] and "metadata" in message and "model_type" in message["metadata"]:
                        model_type = message["metadata"]["model_type"]
                        if model_type == "fine_tuned":
                            model_label = "🧠 " + (t.get("fine_tuned_model", "微调模型") if self.lang == "zh" else "Fine-tuned Model")
                        elif model_type == "rag":
                            model_label = "🔍 " + (t.get("rag_model", "检索增强模型") if self.lang == "zh" else "RAG Model")
                        else:
                            model_label = None
                            
                        if model_label:
                            st.caption(model_label)
        
        # 用户输入提示
        if 'awaiting_followup' in st.session_state and st.session_state.awaiting_followup:
            input_placeholder = t.get("followup_input_placeholder", "请回答医生的追问...")
        else:
            input_placeholder = t.get("input_placeholder", "请描述您的症状或输入命令 (/help 查看帮助):")
        
        # 用户输入
        st.text_input(input_placeholder, key="user_input", on_change=self._handle_user_input)
        
        # 如果有药物推荐，显示它们并提供药店/医院按钮
        if 'recommended_medications' in st.session_state and st.session_state.recommended_medications:
            with st.expander(t.get("view_recommendations", "查看药物推荐"), expanded=True):
                st.markdown(st.session_state.recommended_medications)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t.get("find_nearby_pharmacy", "查找附近药房"), key="find_pharmacy_button"):  # 添加唯一key
                        st.session_state.active_view = "pharmacy"
                        st.rerun()
                with col2:
                    if st.button(t.get("find_nearby_hospital", "查找附近医院"), key="find_hospital_button"):  # 添加唯一key
                        st.session_state.active_view = "hospital"
                        st.rerun()
