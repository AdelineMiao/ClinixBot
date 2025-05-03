# fine_tuned_model.py
import os
import json
import openai
import pandas as pd
from openai import OpenAI

class FineTunedMedicalModel:
    def __init__(self, model_name=None):
        """Initialize fine-tuned model with optional model name"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        
        # Default to GPT-4, use fine-tuned model if provided
        self.model_name = model_name if model_name else "gpt-4-turbo"
    
    def prepare_training_data(self, csv_path):
        """Prepare fine-tuning training data from CSV"""
        # Handle both file paths and uploaded file objects
        if isinstance(csv_path, str):
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_csv(csv_path)
        
        # Create training samples
        training_data = []
        
        for _, row in df.iterrows():
            # Create both Chinese and English training examples
            for lang in ["zh", "en"]:
                if lang == "zh":
                    # Chinese example
                    symptoms = f"患者描述症状: {row['Medical Condition']}相关症状。{row['Doctor\'s Notes']}"
                    response = f"""
                    初步诊断: {row['Medical Condition']}
                    建议治疗: {row['Treatments']}
                    """
                else:
                    # English example
                    symptoms = f"Patient describes symptoms: {row['Medical Condition']} related symptoms. {row['Doctor\'s Notes']}"
                    response = f"""
                    Preliminary diagnosis: {row['Medical Condition']}
                    Recommended treatment: {row['Treatments']}
                    """
                
                # Add to training data with language-specific system prompt
                system_prompt = "你是ClinixBot，一个专业的医疗诊断助手，根据患者的症状提供初步诊断和治疗建议。" if lang == "zh" else "You are ClinixBot, a professional medical diagnostic assistant providing preliminary diagnosis and treatment suggestions based on patient symptoms."
                
                training_data.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": symptoms},
                        {"role": "assistant", "content": response}
                    ]
                })
        
        return training_data
    
    def create_fine_tuning_job(self, training_data, hyperparameters=None):
        """Create a fine-tuning job with OpenAI"""
        try:
            # Save training data to JSONL file
            with open("training_data.jsonl", "w") as f:
                for entry in training_data:
                    f.write(json.dumps(entry) + "\n")
            
            # Upload training file
            with open("training_data.jsonl", "rb") as f:
                training_file = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            # Default hyperparameters
            if hyperparameters is None:
                hyperparameters = {"n_epochs": 3}
            
            # Create fine-tuning job
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file.id,
                model="gpt-3.5-turbo",
                hyperparameters=hyperparameters
            )
            
            return response
            
        except Exception as e:
            return f"Error creating fine-tuning job: {str(e)}"
    
    def get_diagnosis(self, symptoms_description, lang="zh"):
        """Get diagnosis using fine-tuned model"""
        try:
            # Select system prompt based on language
            system_prompt = "你是ClinixBot，一个专业的医疗诊断助手，根据患者的症状提供初步诊断和治疗建议。" if lang == "zh" else "You are ClinixBot, a professional medical diagnostic assistant providing preliminary diagnosis and treatment suggestions based on patient symptoms."
            
            # Format input based on language
            user_input = f"患者描述症状: {symptoms_description}" if lang == "zh" else f"Patient describes symptoms: {symptoms_description}"
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"获取诊断时出错: {str(e)}" if lang == "zh" else f"Error getting diagnosis: {str(e)}"
            return error_msg
    
    def list_fine_tuned_models(self):
        """List available fine-tuned models"""
        try:
            models = self.client.models.list()
            # Filter to only fine-tuned models
            fine_tuned_models = [model for model in models.data if "ft-" in model.id]
            return fine_tuned_models
        except Exception as e:
            return f"Error listing models: {str(e)}"
