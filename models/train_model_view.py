# train_model_view.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from models.fine_tuned_model import FineTunedMedicalModel
from translations import translations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class TrainModelView:
    def __init__(self, lang="zh"):
        self.lang = lang
        self.t = translations
        self.fine_tuned_model = FineTunedMedicalModel()
    
    def render(self):
        """渲染模型训练界面"""
        # 根据语言选择标题
        header_text = "模型训练与评估" if self.lang == "zh" else "Model Training & Evaluation"
        st.header(header_text)
        
        # Create tabs for training and evaluation
        train_tab_text = "训练模型" if self.lang == "zh" else "Train Model"
        eval_tab_text = "评估模型" if self.lang == "zh" else "Evaluate Model"
        
        train_tab, eval_tab = st.tabs([train_tab_text, eval_tab_text])
        
        # Training tab
        with train_tab:
            self._render_training_section()
        
        # Evaluation tab
        with eval_tab:
            self._render_evaluation_section()
    
    def _render_training_section(self):
        """Render the model training section"""
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
        uploaded_file = st.file_uploader(upload_label, type=["csv"], key="train_data_upload")
        
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
    def _render_evaluation_section(self):
        """Render the model evaluation section"""
        # Evaluation introduction text
        intro_text = """
        评估您的医疗诊断模型性能。上传测试数据集，选择要评估的指标，并查看详细分析。
        
        该评估将提供一般指标（准确率、精确率等）和医疗特定指标（按病情、语言和临床重要性的细分评估）。
        """ if self.lang == "zh" else """
        Evaluate your medical diagnostic model performance. Upload a test dataset, select metrics to evaluate, and view detailed analysis.
        
        The evaluation will provide both general metrics (accuracy, precision, etc.) and medical-specific metrics (breakdowns by condition, language, and clinical importance).
        """
        
        st.markdown(intro_text)
        
        # Upload test data
        upload_label = "上传测试数据CSV文件" if self.lang == "zh" else "Upload Test Data CSV"
        test_data = st.file_uploader(upload_label, type=["csv"], key="test_data_upload")
        
        # Model selection
        model_label = "要评估的模型" if self.lang == "zh" else "Model to Evaluate"
        model_name = st.text_input(
            model_label,
            value=st.session_state.fine_tuned_model_name if 'fine_tuned_model_name' in st.session_state and st.session_state.fine_tuned_model_name else ""
        )
        
        # Create columns for metrics selection
        st.subheader("评估指标" if self.lang == "zh" else "Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        # General metrics
        with col1:
            st.markdown("**一般指标**" if self.lang == "zh" else "**General Metrics**")
            use_accuracy = st.checkbox("准确率 (Accuracy)" if self.lang == "zh" else "Accuracy", value=True)
            use_precision = st.checkbox("精确率 (Precision)" if self.lang == "zh" else "Precision", value=True)
            use_recall = st.checkbox("召回率 (Recall)" if self.lang == "zh" else "Recall", value=True)
            use_f1 = st.checkbox("F1 分数" if self.lang == "zh" else "F1 Score", value=True)
            use_response_time = st.checkbox("响应时间" if self.lang == "zh" else "Response Time", value=True)
        
        # Domain-specific metrics
        with col2:
            st.markdown("**医疗特定指标**" if self.lang == "zh" else "**Medical-Specific Metrics**")
            use_by_condition = st.checkbox("按病情分析" if self.lang == "zh" else "Analysis by Condition", value=True)
            use_by_language = st.checkbox("按语言分析" if self.lang == "zh" else "Analysis by Language", value=True)
            use_weighted_score = st.checkbox("医疗加权评分" if self.lang == "zh" else "Medical Weighted Score", value=True)
            use_confusion_matrix = st.checkbox("混淆矩阵" if self.lang == "zh" else "Confusion Matrix", value=True)
        
        # Cost analysis section
        st.subheader("成本分析" if self.lang == "zh" else "Cost Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            include_cost = st.checkbox("包含成本分析" if self.lang == "zh" else "Include Cost Analysis", value=True)
        
        with col4:
            if include_cost:
                monthly_queries = st.number_input(
                    "估计每月查询量" if self.lang == "zh" else "Estimated Monthly Queries",
                    min_value=100,
                    max_value=100000,
                    value=1000,
                    step=100
                )
        
        # Evaluate button
        eval_button_text = "评估模型" if self.lang == "zh" else "Evaluate Model"
        if st.button(eval_button_text):
            if test_data is not None and model_name:
                # Save uploaded file temporarily
                with open("temp_test_data.csv", "wb") as f:
                    f.write(test_data.getbuffer())
                
                # Collect selected metrics
                metrics = []
                if use_accuracy: metrics.append("accuracy")
                if use_precision: metrics.append("precision")
                if use_recall: metrics.append("recall")
                if use_f1: metrics.append("f1")
                if use_response_time: metrics.append("response_time")
                if use_by_condition: metrics.append("by_condition")
                if use_by_language: metrics.append("by_language")
                if use_confusion_matrix: metrics.append("confusion_matrix")
                
                # Initialize model with specified name
                evaluation_model = FineTunedMedicalModel(model_name=model_name)
                
                # Run evaluation
                with st.spinner("正在评估模型..." if self.lang == "zh" else "Evaluating model..."):
                    try:
                        # Run evaluation
                        results = self.evaluate_model("temp_test_data.csv", evaluation_model, metrics)
                        
                        # Add weighted score if selected
                        if use_weighted_score and "predictions" in results and "ground_truth" in results:
                            weighted_results = self.calculate_medical_weighted_score(
                                results["predictions"],
                                results["ground_truth"]
                            )
                            results["weighted_score"] = weighted_results
                        
                        # Add cost analysis if selected
                        if include_cost:
                            cost_analysis = self.calculate_inference_costs(
                                results,
                                {"monthly_queries": monthly_queries}
                            )
                            results["cost_analysis"] = cost_analysis
                        
                        # Display results
                        self.display_evaluation_results(results)
                        
                        # Clean up temp file
                        os.remove("temp_test_data.csv")
                        
                    except Exception as e:
                        st.error(f"评估过程中出错: {str(e)}" if self.lang == "zh" else f"Error during evaluation: {str(e)}")
            else:
                st.error("请上传测试数据并指定模型名称" if self.lang == "zh" else "Please upload test data and specify a model name")
    def evaluate_model(self, test_data_path, model, metrics=None):
        """
        Evaluate the fine-tuned model on test data
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # Results containers
        results = {
            "general": {},
            "by_condition": {},
            "by_language": {"zh": {}, "en": {}}
        }
        
        # Track predictions and ground truth
        predictions = []
        ground_truth = []
        response_times = []
        languages = []
        conditions = []
        
        # Add logging for debugging
        st.info(f"Loaded {len(test_df)} test records. Starting evaluation...")
        
        # Process each test case
        for idx, row in test_df.iterrows():
            # Make sure Medical Condition column exists
            if "Medical Condition" not in row:
                st.error("Test data missing 'Medical Condition' column")
                return {"error": "Test data missing required columns"}
                
            # Make sure Doctor's Notes column exists
            if "Doctor's Notes" not in row:
                st.error("Test data missing 'Doctor's Notes' column")
                return {"error": "Test data missing required columns"}
                
            # Test in both languages (or just one if testing performance)
            test_langs = ["en"]  # Can change to ["zh", "en"] to test both
            
            for lang in test_langs:
                try:
                    # Start timer
                    start_time = time.time()
                    
                    # Get model prediction - add debugging
                    symptoms = row["Doctor's Notes"]
                    st.write(f"Testing sample {idx+1}/{len(test_df)} ({lang})")
                    
                    # Verify model has get_diagnosis method
                    if not hasattr(model, 'get_diagnosis'):
                        st.error(f"Model {model.__class__.__name__} does not have get_diagnosis method")
                        return {"error": "Invalid model: missing get_diagnosis method"}
                    
                    # Get diagnosis
                    response = model.get_diagnosis(symptoms, lang=lang)
                    
                    # End timer
                    end_time = time.time()
                    
                    # Show raw response for debugging
                    if idx < 3:  # Show first 3 responses for debugging
                        st.code(response, language="text")
                    
                    # Parse prediction
                    if lang == "zh":
                        predicted_condition = self._extract_diagnosis(response, "初步诊断:")
                    else:
                        predicted_condition = self._extract_diagnosis(response, "Preliminary diagnosis:")
                    
                    # Record results
                    predictions.append(predicted_condition)
                    ground_truth.append(row["Medical Condition"])
                    response_times.append(end_time - start_time)
                    languages.append(lang)
                    conditions.append(row["Medical Condition"])
                    
                except Exception as e:
                    st.error(f"Error processing sample {idx+1} ({lang}): {str(e)}")
                    # Instead of stopping, continue with next sample
                    continue
        
        # Check if we have any successful predictions
        if not predictions:
            st.error("No successful predictions were made. Cannot calculate metrics.")
            return {"error": "No successful predictions"}
    
        
        # Calculate general metrics
        if any(m in metrics for m in ["accuracy", "precision", "recall", "f1"]):
            if "accuracy" in metrics:
                results["general"]["accuracy"] = accuracy_score(ground_truth, predictions)
            
            if any(m in metrics for m in ["precision", "recall", "f1"]):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    ground_truth, 
                    predictions, 
                    average='weighted'
                )
                
                if "precision" in metrics:
                    results["general"]["precision"] = precision
                
                if "recall" in metrics:
                    results["general"]["recall"] = recall
                
                if "f1" in metrics:
                    results["general"]["f1"] = f1
        
        # Calculate response time metrics
        if "response_time" in metrics:
            results["general"]["avg_response_time"] = sum(response_times) / len(response_times)
            results["general"]["min_response_time"] = min(response_times)
            results["general"]["max_response_time"] = max(response_times)
        
        # Calculate per-condition metrics
        if "by_condition" in metrics:
            unique_conditions = set(ground_truth)
            
            for condition in unique_conditions:
                # Filter data for this condition
                condition_indices = [i for i, gt in enumerate(ground_truth) if gt == condition]
                condition_preds = [predictions[i] for i in condition_indices]
                condition_truth = [ground_truth[i] for i in condition_indices]
                
                # Calculate condition-specific accuracy
                condition_accuracy = sum(p == t for p, t in zip(condition_preds, condition_truth)) / len(condition_indices)
                results["by_condition"][condition] = {
                    "accuracy": condition_accuracy,
                    "count": len(condition_indices)
                }
        
        # Calculate per-language metrics
        if "by_language" in metrics:
            for lang in ["zh", "en"]:
                # Filter data for this language
                lang_indices = [i for i, l in enumerate(languages) if l == lang]
                
                if not lang_indices:
                    continue
                    
                lang_preds = [predictions[i] for i in lang_indices]
                lang_truth = [ground_truth[i] for i in lang_indices]
                
                # Calculate language-specific accuracy
                lang_accuracy = sum(p == t for p, t in zip(lang_preds, lang_truth)) / len(lang_indices)
                
                results["by_language"][lang]["accuracy"] = lang_accuracy
                
                # Calculate precision, recall, f1 for each language
                try:
                    lang_precision, lang_recall, lang_f1, _ = precision_recall_fscore_support(
                        lang_truth, 
                        lang_preds, 
                        average='weighted'
                    )
                    
                    results["by_language"][lang]["precision"] = lang_precision
                    results["by_language"][lang]["recall"] = lang_recall
                    results["by_language"][lang]["f1"] = lang_f1
                except:
                    # Handle cases where metrics can't be calculated
                    pass
        
        # Generate confusion matrix
        if "confusion_matrix" in metrics:
            try:
                # Get unique classes (sorted to ensure consistent ordering)
                classes = sorted(list(set(ground_truth)))
                
                # Calculate confusion matrix
                cm = confusion_matrix(ground_truth, predictions, labels=classes)
                
                # Store in results
                results["confusion_matrix"] = {
                    "matrix": cm.tolist(),
                    "classes": classes
                }
            except Exception as e:
                st.warning(f"计算混淆矩阵时出错: {str(e)}" if self.lang == "zh" else f"Error calculating confusion matrix: {str(e)}")
        
        return results
    
    def _extract_diagnosis(self, model_output, prefix):
        """Extract diagnosis from model output based on prefix"""
        if not model_output:
            return "Unknown"
            
        # Look for the prefix in each line
        for line in model_output.split('\n'):
            line = line.strip()
            if line.startswith(prefix):
                diagnosis = line[len(prefix):].strip()
                return diagnosis
        
        # If no diagnosis found with prefix, return the first non-empty line
        # or a default value if everything fails
        for line in model_output.split('\n'):
            line = line.strip()
            if line:
                return line
                
        return "Unknown"
    
    def calculate_medical_weighted_score(self, predictions, ground_truth, condition_weights=None):
        """
        Calculate weighted score based on medical significance
        
        Args:
            predictions: List of predicted conditions
            ground_truth: List of actual conditions
            condition_weights: Optional dict of condition weights
            
        Returns:
            Dictionary with weighted score results
        """
        # Default weights if not provided
        if condition_weights is None:
            condition_weights = {
                # Critical conditions (high weight due to importance)
                "Pneumonia": 3.0,
                "Myocardial Infarction": 3.0,
                "Stroke": 3.0,
                "Sepsis": 3.0,
                
                # Moderate conditions
                "Hypertension": 2.0,
                "Diabetes": 2.0,
                "Bronchitis": 2.0,
                "Asthma": 2.0,
                
                # Mild conditions
                "Common Cold": 1.0,
                "Seasonal Allergies": 1.0,
                "Sinusitis": 1.0
            }
        
        # Default weight for unlisted conditions
        default_weight = 1.5
        
        # Calculate weighted score
        total_weight = 0
        weighted_correct = 0
        condition_results = {}
        
        for pred, truth in zip(predictions, ground_truth):
            # Get weight for this condition
            weight = condition_weights.get(truth, default_weight)
            total_weight += weight
            
            # Add weighted score if correct
            is_correct = pred == truth
            if is_correct:
                weighted_correct += weight
            
            # Track per-condition results
            if truth not in condition_results:
                condition_results[truth] = {
                    "weight": weight,
                    "total": 0,
                    "correct": 0
                }
            
            condition_results[truth]["total"] += 1
            if is_correct:
                condition_results[truth]["correct"] += 1
        
        # Calculate final weighted score
        weighted_score = weighted_correct / total_weight if total_weight > 0 else 0
        
        # Calculate per-condition weighted accuracy
        for condition, results in condition_results.items():
            if results["total"] > 0:
                results["accuracy"] = results["correct"] / results["total"]
        
        return {
            "weighted_score": weighted_score,
            "total_weight": total_weight,
            "weighted_correct": weighted_correct,
            "by_condition": condition_results
        }
    
    def calculate_inference_costs(self, evaluation_results, usage_projection=None):
        """
        Calculate inference costs based on evaluation results
        
        Args:
            evaluation_results: Results from model evaluation
            usage_projection: Dictionary with usage projections
            
        Returns:
            Dictionary with cost analysis
        """
        # Default to 1000 queries per month if not specified
        if usage_projection is None:
            usage_projection = {"monthly_queries": 1000}
        
        # Estimate tokens from response times (rough approximation)
        if "general" in evaluation_results and "avg_response_time" in evaluation_results["general"]:
            avg_response_time = evaluation_results["general"]["avg_response_time"]
            
            # Estimate token counts based on response time
            # This is a very rough approximation and should be replaced with actual token counts if available
            estimated_input_tokens = 100  # Placeholder - ideally get from API response
            estimated_output_tokens = max(50, int(avg_response_time * 10))  # Rough estimate
        else:
            # Default estimates if response time not available
            estimated_input_tokens = 100
            estimated_output_tokens = 150
        
        # Current pricing (as of May 2025 - update as needed)
        # These are example rates - replace with accurate values
        base_input_cost_per_1k = 0.003  # $0.003 per 1K input tokens
        base_output_cost_per_1k = 0.006  # $0.006 per 1K output tokens
        
        # Fine-tuned model has higher rates
        ft_input_cost_per_1k = 0.012   # $0.012 per 1K input tokens
        ft_output_cost_per_1k = 0.016  # $0.016 per 1K output tokens
        
        # Calculate both base model and fine-tuned model costs
        base_input_cost = (estimated_input_tokens / 1000) * base_input_cost_per_1k
        base_output_cost = (estimated_output_tokens / 1000) * base_output_cost_per_1k
        base_query_cost = base_input_cost + base_output_cost
        
        ft_input_cost = (estimated_input_tokens / 1000) * ft_input_cost_per_1k
        ft_output_cost = (estimated_output_tokens / 1000) * ft_output_cost_per_1k
        ft_query_cost = ft_input_cost + ft_output_cost
        
        # Project monthly costs
        monthly_queries = usage_projection["monthly_queries"]
        base_monthly_cost = base_query_cost * monthly_queries
        ft_monthly_cost = ft_query_cost * monthly_queries
        
        # Calculate ROI if we have accuracy results
        roi_analysis = None
        if "general" in evaluation_results and "accuracy" in evaluation_results["general"]:
            base_accuracy = 0.80  # Assumed baseline accuracy - ideally compare with actual base model
            ft_accuracy = evaluation_results["general"]["accuracy"]
            
            # Calculate accuracy improvement
            accuracy_improvement = ft_accuracy - base_accuracy
            
            # Simple ROI calculation (very simplified)
            cost_difference = ft_monthly_cost - base_monthly_cost
            roi_analysis = {
                "base_accuracy": base_accuracy,
                "ft_accuracy": ft_accuracy,
                "accuracy_improvement": accuracy_improvement,
                "additional_monthly_cost": cost_difference,
                "cost_per_accuracy_point": cost_difference / (accuracy_improvement * 100) if accuracy_improvement > 0 else "N/A"
            }
        
        return {
            "per_query": {
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "base_model_cost": base_query_cost,
                "fine_tuned_model_cost": ft_query_cost
            },
            "monthly": {
                "query_volume": monthly_queries,
                "base_model_cost": base_monthly_cost,
                "fine_tuned_model_cost": ft_monthly_cost,
                "cost_difference": ft_monthly_cost - base_monthly_cost
            },
            "roi_analysis": roi_analysis
        }
    
    def display_evaluation_results(self, results):
        """Display evaluation results in a formatted way"""
        # Create main metrics display
        st.subheader("评估结果摘要" if self.lang == "zh" else "Evaluation Results Summary")
        
        if "general" in results and results["general"]:
            # Create metrics display for general metrics
            metric_cols = st.columns(4)
            
            metrics_displayed = 0
            if "accuracy" in results["general"]:
                with metric_cols[metrics_displayed % 4]:
                    st.metric(
                        "准确率" if self.lang == "zh" else "Accuracy",
                        f"{results['general']['accuracy']:.2%}"
                    )
                metrics_displayed += 1
            
            if "precision" in results["general"]:
                with metric_cols[metrics_displayed % 4]:
                    st.metric(
                        "精确率" if self.lang == "zh" else "Precision",
                        f"{results['general']['precision']:.2%}"
                    )
                metrics_displayed += 1
            
            if "recall" in results["general"]:
                with metric_cols[metrics_displayed % 4]:
                    st.metric(
                        "召回率" if self.lang == "zh" else "Recall",
                        f"{results['general']['recall']:.2%}"
                    )
                metrics_displayed += 1
            
            if "f1" in results["general"]:
                with metric_cols[metrics_displayed % 4]:
                    st.metric(
                        "F1 分数" if self.lang == "zh" else "F1 Score",
                        f"{results['general']['f1']:.2%}"
                    )
                metrics_displayed += 1
            
            if "avg_response_time" in results["general"]:
                with metric_cols[metrics_displayed % 4]:
                    st.metric(
                        "平均响应时间" if self.lang == "zh" else "Avg. Response Time",
                        f"{results['general']['avg_response_time']:.2f}s"
                    )
                metrics_displayed += 1
        
        # Display weighted score if available
        if "weighted_score" in results:
            st.subheader("医疗加权评分" if self.lang == "zh" else "Medical Weighted Score")
            
            weighted_score = results["weighted_score"]["weighted_score"]
            st.metric(
                "医疗加权准确率" if self.lang == "zh" else "Medical Weighted Accuracy",
                f"{weighted_score:.2%}"
            )
            
            st.markdown(
                "该指标根据医疗条件的严重性为每个诊断分配不同权重。" if self.lang == "zh" else
                "This metric assigns different weights to each diagnosis based on the severity of medical conditions."
            )
        
        # Display breakdown by condition
        if "by_condition" in results and results["by_condition"]:
            st.subheader("按条件分析" if self.lang == "zh" else "Analysis by Condition")
            
            # Prepare data for chart
            condition_data = []
            for condition, data in results["by_condition"].items():
                condition_data.append({
                    "condition": condition,
                    "accuracy": data["accuracy"],
                    "count": data["count"]
                })
            
            # Sort by accuracy (descending)
            condition_data = sorted(condition_data, key=lambda x: x["accuracy"], reverse=True)
            
            # Create DataFrame for chart
            condition_df = pd.DataFrame(condition_data)
            
            # Plot horizontal bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, max(4, len(condition_data) * 0.5)))
            
            bars = ax.barh(condition_df["condition"], condition_df["accuracy"])
            
            # Add labels and percentages
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width + 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f"{width:.1%} (n={condition_df['count'].iloc[i]})",
                        va='center')
            
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("准确率" if self.lang == "zh" else "Accuracy")
            ax.set_title("各条件准确率" if self.lang == "zh" else "Accuracy by Condition")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Display breakdown by language
        if "by_language" in results and all(lang in results["by_language"] for lang in ["zh", "en"]):
            st.subheader("按语言分析" if self.lang == "zh" else "Analysis by Language")
            
            # Create columns for language comparison
            lang_cols = st.columns(2)
            
            # Chinese metrics
            with lang_cols[0]:
                st.markdown("**中文**")
                zh_data = results["by_language"]["zh"]
                
                if "accuracy" in zh_data:
                    st.metric("准确率", f"{zh_data['accuracy']:.2%}")
                
                if "precision" in zh_data:
                    st.metric("精确率", f"{zh_data['precision']:.2%}")
                
                if "recall" in zh_data:
                    st.metric("召回率", f"{zh_data['recall']:.2%}")
                
                if "f1" in zh_data:
                    st.metric("F1 分数", f"{zh_data['f1']:.2%}")
            
            # English metrics
            with lang_cols[1]:
                st.markdown("**English**")
                en_data = results["by_language"]["en"]
                
                if "accuracy" in en_data:
                    st.metric("Accuracy", f"{en_data['accuracy']:.2%}")
                
                if "precision" in en_data:
                    st.metric("Precision", f"{en_data['precision']:.2%}")
                
                if "recall" in en_data:
                    st.metric("Recall", f"{en_data['recall']:.2%}")
                
                if "f1" in en_data:
                    st.metric("F1 Score", f"{en_data['f1']:.2%}")
        
        # Display confusion matrix if available
        if "confusion_matrix" in results:
            st.subheader("混淆矩阵" if self.lang == "zh" else "Confusion Matrix")
            
            matrix = np.array(results["confusion_matrix"]["matrix"])
            classes = results["confusion_matrix"]["classes"]
            
            # Create confusion matrix plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot confusion matrix as heatmap
            im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Add axis labels
            ax.set(xticks=np.arange(matrix.shape[1]),
                   yticks=np.arange(matrix.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   ylabel='真实标签' if self.lang == "zh" else 'True label',
                   xlabel='预测标签' if self.lang == "zh" else 'Predicted label')
            
            # Rotate x labels and set alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            thresh = matrix.max() / 2.
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, format(matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if matrix[i, j] > thresh else "black")
            
            fig.tight_layout()
            st.pyplot(fig)
        
        # Display cost analysis if available
        if "cost_analysis" in results:
            st.subheader("成本分析" if self.lang == "zh" else "Cost Analysis")
            
            cost_data = results["cost_analysis"]
            
            # Per query costs
            st.markdown("**每次查询成本**" if self.lang == "zh" else "**Per Query Costs**")
            query_cols = st.columns(2)
            
            with query_cols[0]:
                st.metric(
                    "基础模型成本" if self.lang == "zh" else "Base Model Cost",
                    f"${cost_data['per_query']['base_model_cost']:.5f}"
                )
                st.caption(f"输入令牌: {cost_data['per_query']['estimated_input_tokens']}" if self.lang == "zh" else 
                          f"Input tokens: {cost_data['per_query']['estimated_input_tokens']}")
                st.caption(f"输出令牌: {cost_data['per_query']['estimated_output_tokens']}" if self.lang == "zh" else 
                          f"Output tokens: {cost_data['per_query']['estimated_output_tokens']}")
            
            with query_cols[1]:
                st.metric(
                    "微调模型成本" if self.lang == "zh" else "Fine-Tuned Model Cost",
                    f"${cost_data['per_query']['fine_tuned_model_cost']:.5f}"
                )
                
                # Calculate and display cost difference
                cost_diff = cost_data['per_query']['fine_tuned_model_cost'] - cost_data['per_query']['base_model_cost']
                st.caption(f"额外成本: ${cost_diff:.5f}" if self.lang == "zh" else f"Additional cost: ${cost_diff:.5f}")
            
            # Monthly projections
            st.markdown("**月度成本预估**" if self.lang == "zh" else "**Monthly Cost Projections**")
            monthly_cols = st.columns(3)
            
            with monthly_cols[0]:
                st.metric(
                    "每月查询量" if self.lang == "zh" else "Monthly Queries",
                    f"{cost_data['monthly']['query_volume']:,}"
                )
            
            with monthly_cols[1]:
                st.metric(
                    "基础模型月度成本" if self.lang == "zh" else "Base Model Monthly Cost",
                    f"${cost_data['monthly']['base_model_cost']:.2f}"
                )
            
            with monthly_cols[2]:
                st.metric(
                    "微调模型月度成本" if self.lang == "zh" else "Fine-Tuned Model Monthly Cost",
                    f"${cost_data['monthly']['fine_tuned_model_cost']:.2f}"
                )
                st.caption(f"额外月度成本: ${cost_data['monthly']['cost_difference']:.2f}" if self.lang == "zh" else 
                          f"Additional monthly cost: ${cost_data['monthly']['cost_difference']:.2f}")
            
            # ROI analysis if available
            if cost_data["roi_analysis"]:
                st.markdown("**投资回报分析**" if self.lang == "zh" else "**ROI Analysis**")
                roi_data = cost_data["roi_analysis"]
                
                roi_cols = st.columns(3)
                
                with roi_cols[0]:
                    st.metric(
                        "准确率提升" if self.lang == "zh" else "Accuracy Improvement",
                        f"{roi_data['accuracy_improvement']:.2%}"
                    )
                    st.caption(f"基础: {roi_data['base_accuracy']:.2%}, 微调: {roi_data['ft_accuracy']:.2%}" if self.lang == "zh" else 
                              f"Base: {roi_data['base_accuracy']:.2%}, Fine-tuned: {roi_data['ft_accuracy']:.2%}")
                
                with roi_cols[1]:
                    st.metric(
                        "额外月度成本" if self.lang == "zh" else "Additional Monthly Cost",
                        f"${roi_data['additional_monthly_cost']:.2f}"
                    )
                
                with roi_cols[2]:
                    if roi_data['cost_per_accuracy_point'] != "N/A":
                        st.metric(
                            "每百分点准确率成本" if self.lang == "zh" else "Cost per Point of Accuracy",
                            f"${roi_data['cost_per_accuracy_point']:.2f}"
                        )
                        st.caption("每提高1%准确率的月度成本" if self.lang == "zh" else "Monthly cost per 1% accuracy improvement")
                    else:
                        st.warning("无法计算ROI（准确率未提高）" if self.lang == "zh" else "Cannot calculate ROI (no accuracy improvement)")
        
        # Raw data download
        st.subheader("原始评估数据" if self.lang == "zh" else "Raw Evaluation Data")
        
        # Convert results to JSON
        results_json = json.dumps(results, indent=2)
        st.download_button(
            "下载评估数据 (JSON)" if self.lang == "zh" else "Download Evaluation Data (JSON)",
            results_json,
            file_name="model_evaluation_results.json",
            mime="application/json"
        )
    def _render_evaluation_section(self):
        """Render the model evaluation section"""
        # Evaluation introduction text
        intro_text = """
        评估您的医疗诊断模型性能。上传测试数据集，选择要评估的指标，并查看详细分析。
        
        该评估将提供一般指标（准确率、精确率等）和医疗特定指标（按病情、语言和临床重要性的细分评估）。
        """ if self.lang == "zh" else """
        Evaluate your medical diagnostic model performance. Upload a test dataset, select metrics to evaluate, and view detailed analysis.
        
        The evaluation will provide both general metrics (accuracy, precision, etc.) and medical-specific metrics (breakdowns by condition, language, and clinical importance).
        """
        
        st.markdown(intro_text)
        
        # Upload test data
        upload_label = "上传测试数据CSV文件" if self.lang == "zh" else "Upload Test Data CSV"
        test_data = st.file_uploader(upload_label, type=["csv"], key="test_data_upload")
        
        # Model selection
        model_label = "要评估的模型" if self.lang == "zh" else "Model to Evaluate"
        model_name = st.text_input(
            model_label,
            value=st.session_state.fine_tuned_model_name if 'fine_tuned_model_name' in st.session_state and st.session_state.fine_tuned_model_name else ""
        )
        
        # Create columns for metrics selection
        st.subheader("评估指标" if self.lang == "zh" else "Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        # General metrics
        with col1:
            st.markdown("**一般指标**" if self.lang == "zh" else "**General Metrics**")
            use_accuracy = st.checkbox("准确率 (Accuracy)" if self.lang == "zh" else "Accuracy", value=True)
            use_precision = st.checkbox("精确率 (Precision)" if self.lang == "zh" else "Precision", value=True)
            use_recall = st.checkbox("召回率 (Recall)" if self.lang == "zh" else "Recall", value=True)
            use_f1 = st.checkbox("F1 分数" if self.lang == "zh" else "F1 Score", value=True)
            use_response_time = st.checkbox("响应时间" if self.lang == "zh" else "Response Time", value=True)
        
        # Domain-specific metrics
        with col2:
            st.markdown("**医疗特定指标**" if self.lang == "zh" else "**Medical-Specific Metrics**")
            use_by_condition = st.checkbox("按病情分析" if self.lang == "zh" else "Analysis by Condition", value=True)
            use_by_language = st.checkbox("按语言分析" if self.lang == "zh" else "Analysis by Language", value=True)
            use_weighted_score = st.checkbox("医疗加权评分" if self.lang == "zh" else "Medical Weighted Score", value=True)
            use_confusion_matrix = st.checkbox("混淆矩阵" if self.lang == "zh" else "Confusion Matrix", value=True)
        
        # Cost analysis section
        st.subheader("成本分析" if self.lang == "zh" else "Cost Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            include_cost = st.checkbox("包含成本分析" if self.lang == "zh" else "Include Cost Analysis", value=True)
        
        with col4:
            if include_cost:
                monthly_queries = st.number_input(
                    "估计每月查询量" if self.lang == "zh" else "Estimated Monthly Queries",
                    min_value=100,
                    max_value=100000,
                    value=1000,
                    step=100
                )
        
        # Evaluate button
        eval_button_text = "评估模型" if self.lang == "zh" else "Evaluate Model"
        if st.button(eval_button_text):
            if test_data is not None and model_name:
                # Save uploaded file temporarily
                with open("temp_test_data.csv", "wb") as f:
                    f.write(test_data.getbuffer())
                
                # Collect selected metrics
                metrics = []
                if use_accuracy: metrics.append("accuracy")
                if use_precision: metrics.append("precision")
                if use_recall: metrics.append("recall")
                if use_f1: metrics.append("f1")
                if use_response_time: metrics.append("response_time")
                if use_by_condition: metrics.append("by_condition")
                if use_by_language: metrics.append("by_language")
                if use_confusion_matrix: metrics.append("confusion_matrix")
                
                # Initialize model with better error handling
                try:
                    # Show user we're trying to load the model
                    st.info(f"Initializing model: {model_name}")
                    
                    # Initialize model
                    evaluation_model = FineTunedMedicalModel(model_name=model_name)
                    
                    # Verify model is properly initialized
                    if not hasattr(evaluation_model, 'get_diagnosis'):
                        st.error("Model initialization failed: missing get_diagnosis method")
                    else:
                        # Run evaluation with progress
                        with st.spinner("Evaluating model..."):
                            # Test with a single sample first to verify it works
                            test_df = pd.read_csv("temp_test_data.csv")
                            if len(test_df) > 0:
                                try:
                                    # Test a single prediction
                                    test_symptoms = test_df.iloc[0]["Doctor's Notes"]
                                    test_response = evaluation_model.get_diagnosis(test_symptoms, "en")
                                    st.success("Model initialized successfully! Starting evaluation...")
                                    
                                    # Run full evaluation
                                    results = self.evaluate_model("temp_test_data.csv", evaluation_model, metrics)
                                    
                                    # Display results if successful
                                    if "error" in results:
                                        st.error(f"Evaluation failed: {results['error']}")
                                    else:
                                        # Add weighted score if selected
                                        if use_weighted_score and "predictions" in results and "ground_truth" in results:
                                            weighted_results = self.calculate_medical_weighted_score(
                                                results["predictions"],
                                                results["ground_truth"]
                                            )
                                            results["weighted_score"] = weighted_results
                                        
                                        # Add cost analysis if selected
                                        if include_cost:
                                            cost_analysis = self.calculate_inference_costs(
                                                results,
                                                {"monthly_queries": monthly_queries}
                                            )
                                            results["cost_analysis"] = cost_analysis
                                        
                                        # Display results
                                        self.display_evaluation_results(results)
                                except Exception as e:
                                    st.error(f"Model test failed: {str(e)}")
                            else:
                                st.error("Test data is empty")
                                
                except Exception as e:
                    st.error(f"Model initialization failed: {str(e)}")
                    
                # Clean up
                try:
                    os.remove("temp_test_data.csv")
                except:
                    pass
            else:
                st.error("请上传测试数据并指定模型名称" if self.lang == "zh" else "Please upload test data and specify a model name")
