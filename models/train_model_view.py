# train_model_view.py
import streamlit as st
import pandas as pd
from models.fine_tuned_model import FineTunedMedicalModel
from translations import translations

class TrainModelView:
    def __init__(self, lang="zh"):
        self.lang = lang
        self.t = translations
        self.fine_tuned_model = FineTunedMedicalModel()
    
    def render(self):
        """渲染模型训练界面"""
        # 根据语言选择标题
        header_text = "模型训练" if self.lang == "zh" else "Model Training"
        st.header(header_text)
        
        # 培训介绍文本
        intro_text = """
        通过上传医疗记录数据来训练一个定制的诊断模型。训练完成后，该模型将能够从患者症状直接预测诊断结果，
        无需每次查询时进行相似案例检索。这可以提高诊断速度并使响应更加一致。
        
        请上传包含以下列的CSV文件:
        - Medical Condition (疾病/诊断)
        - Doctor's Notes (医生记录)
        - Treatments (治疗方法)
        """ if self.lang == "zh" else """
        Train a customized diagnostic model by uploading medical record data. Once trained, the model will be able to predict 
        diagnoses directly from patient symptoms, without needing to retrieve similar cases for each query. This can improve 
        diagnostic speed and make responses more consistent.
        
        Please upload a CSV file containing the following columns:
        - Medical Condition
        - Doctor's Notes
        - Treatments
        """
        
        st.markdown(intro_text)
        
        # 上传训练数据
        upload_label = "上传用于训练的CSV数据" if self.lang == "zh" else "Upload CSV data for training"
        uploaded_file = st.file_uploader(upload_label, type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # 检查必须的列是否存在
                required_columns = ["Medical Condition", "Doctor's Notes", "Treatments"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    missing_cols_str = ", ".join(missing_columns)
                    error_msg = f"数据中缺少必需的列: {missing_cols_str}" if self.lang == "zh" else f"Missing required columns in data: {missing_cols_str}"
                    st.error(error_msg)
                else:
                    # 显示加载的记录数
                    records_msg = f"已加载 {len(df)} 条记录用于训练" if self.lang == "zh" else f"Loaded {len(df)} records for training"
                    st.write(records_msg)
                    
                    # 预览数据
                    preview_label = "数据预览" if self.lang == "zh" else "Data Preview"
                    st.subheader(preview_label)
                    st.dataframe(df.head())
                    
                    # 训练配置
                    config_label = "训练配置" if self.lang == "zh" else "Training Configuration"
                    st.subheader(config_label)
                    
                    epochs_label = "训练轮次" if self.lang == "zh" else "Number of epochs"
                    epochs = st.slider(epochs_label, min_value=1, max_value=5, value=3)
                    
                    # 训练按钮
                    train_button_label = "开始训练" if self.lang == "zh" else "Start Training"
                    if st.button(train_button_label):
                        # 准备训练数据的消息
                        preparing_msg = "正在准备训练数据..." if self.lang == "zh" else "Preparing training data..."
                        with st.spinner(preparing_msg):
                            training_data = self.fine_tuned_model.prepare_training_data(uploaded_file)
                            prepared_msg = f"已准备 {len(training_data)} 个训练样本" if self.lang == "zh" else f"Prepared {len(training_data)} training examples"
                            st.write(prepared_msg)
                        
                        # 创建微调任务的消息
                        creating_msg = "正在创建微调任务..." if self.lang == "zh" else "Creating fine-tuning job..."
                        with st.spinner(creating_msg):
                            response = self.fine_tuned_model.create_fine_tuning_job(
                                training_data, 
                                hyperparameters={"n_epochs": epochs}
                            )
                            
                            if isinstance(response, str) and response.startswith("Error"):
                                st.error(response)
                            else:
                                # 成功消息
                                success_msg = "微调任务创建成功！" if self.lang == "zh" else "Fine-tuning job created successfully!"
                                st.success(success_msg)
                                st.json(response)
                                
                                # 保存任务ID以便跟踪
                                st.session_state.job_id = response.id
                                
                                # 提示文本
                                if self.lang == "zh":
                                    prompt_text = """
                                    微调任务已提交到OpenAI。训练可能需要几小时至几天的时间，具体取决于数据大小和OpenAI的处理队列。
                                    
                                    完成后，您将可以在模型设置中选择此微调模型。请将以下模型ID保存在安全的地方：
                                    """
                                else:
                                    prompt_text = """
                                    The fine-tuning job has been submitted to OpenAI. Training may take several hours to days, 
                                    depending on the data size and OpenAI's processing queue.
                                    
                                    Once completed, you'll be able to select this fine-tuned model in the model settings. 
                                    Please save the following model ID in a secure location:
                                    """
                                
                                st.markdown(prompt_text)
                                st.code(f"ft-{response.id}", language="text")
            
            except Exception as e:
                error_msg = f"处理上传的文件时出错: {str(e)}" if self.lang == "zh" else f"Error processing uploaded file: {str(e)}"
                st.error(error_msg)
        
        # 如果有进行中的任务ID，显示状态部分
        if 'job_id' in st.session_state:
            status_label = "任务状态" if self.lang == "zh" else "Job Status"
            st.subheader(status_label)
            
            job_id = st.session_state.job_id
            st.write(f"Job ID: {job_id}")
            
            # 检查状态按钮
            check_label = "检查任务状态" if self.lang == "zh" else "Check Job Status"
            if st.button(check_label):
                checking_msg = "正在检查状态..." if self.lang == "zh" else "Checking status..."
                with st.spinner(checking_msg):
                    try:
                        # 注意：需要在FineTunedMedicalModel类中实现check_job_status方法
                        # status = self.fine_tuned_model.check_job_status(job_id)
                        # st.json(status)
                        
                        # 临时代码，假设还没有实现check_job_status方法
                        st.info("状态检查功能正在开发中。请访问OpenAI控制面板检查进度。" if self.lang == "zh" else 
                                "Status checking feature is in development. Please check progress on the OpenAI dashboard.")
                    except Exception as e:
                        st.error(str(e))