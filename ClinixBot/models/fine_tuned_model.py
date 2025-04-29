# fine_tuned_model.py
import os
import openai
import pandas as pd

class FineTunedMedicalModel:
    def __init__(self, model_name=None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
        # 默认使用GPT-4，如果有微调模型则使用微调模型
        self.model_name = model_name if model_name else "gpt-4-turbo"
    
    def prepare_training_data(self, csv_path):
        """准备微调训练数据"""
        df = pd.read_csv(csv_path)
        
        # 创建训练样本
        training_data = []
        
        for _, row in df.iterrows():
            # 创建合成的症状描述
            symptoms = f"患者描述症状: {row['Medical Condition']}相关症状。{row['Doctor\'s Notes']}"
            
            # 创建理想的输出响应
            response = f"""
            初步诊断: {row['Medical Condition']}
            建议治疗: {row['Treatments']}
            """
            
            # 添加到训练数据
            training_data.append({
                "messages": [
                    {"role": "system", "content": "你是ClinixBot，一个专业的医疗诊断助手，根据患者的症状提供初步诊断和治疗建议。"},
                    {"role": "user", "content": symptoms},
                    {"role": "assistant", "content": response}
                ]
            })
        
        return training_data
    
    def create_fine_tuning_job(self, training_data, validation_data=None):
        """创建微调任务"""
        try:
            # 将训练数据保存为JSONL文件
            import json
            
            with open("training_data.jsonl", "w") as f:
                for entry in training_data:
                    f.write(json.dumps(entry) + "\n")
            
            # 上传训练文件
            training_file = openai.File.create(
                file=open("training_data.jsonl", "rb"),
                purpose="fine-tune"
            )
            
            # 创建微调任务
            response = openai.FineTuningJob.create(
                training_file=training_file.id,
                model="gpt-3.5-turbo",
                hyperparameters={
                    "n_epochs": 3
                }
            )
            
            return response
            
        except Exception as e:
            return f"创建微调任务时出错: {str(e)}"
    
    def get_diagnosis(self, symptoms_description):
        """使用微调模型获取诊断"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是ClinixBot，一个专业的医疗诊断助手，根据患者的症状提供初步诊断和治疗建议。"},
                    {"role": "user", "content": f"患者描述症状: {symptoms_description}"}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message["content"]
        except Exception as e:
            return f"获取诊断时出错: {str(e)}"