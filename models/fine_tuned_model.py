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
                    doctor_notes = row["Doctor's Notes"]
                    symptoms = f"患者描述的症状: {doctor_notes}"
                    response = f"""
                    初步诊断: {row['Medical Condition']}
                    建议治疗: {row['Treatments']}
                    """
                else:
                    # English example
                    doctor_notes = row["Doctor's Notes"]
                    symptoms = f"Patient describes the following symptoms: {doctor_notes}"
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
    def _extract_diagnosis(self, model_output, prefix):
        if not model_output:
            return "Unknown"
        
        # Normalize model output (lowercase, remove extra spaces)
        normalized_output = " ".join(model_output.lower().strip().split())
        
        # Search for prefix in case-insensitive way
        prefix_lower = prefix.lower()
        
        # Try exact prefix match first
        for line in model_output.split('\n'):
            line = line.strip()
            if line.lower().startswith(prefix_lower):
                diagnosis = line[len(prefix):].strip()
                return diagnosis
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
    
    def get_diagnosis(self, symptoms, lang="en"):
        """
        Get diagnosis from model based on symptoms
        
        Args:
            symptoms (str): Patient symptoms/doctor's notes
            lang (str): Language code ('en' or 'zh')
            
        Returns:
            str: Model's diagnosis response
        """
        try:
            # Format prompt based on language
            if lang == "zh":
                prompt = f"患者症状: {symptoms}\n\n请提供初步诊断:"
            else:
                prompt = f"Patient symptoms: {symptoms}\n\nPlease provide preliminary diagnosis:"
                
            # Use chat completions API instead of completions API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical diagnostic assistant that provides concise, accurate diagnoses based on symptoms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            # Extract the response text from chat format
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting diagnosis: {str(e)}")
            return f"Error: {str(e)}"

    def list_fine_tuned_models(self):
        """List available fine-tuned models"""
        try:
            models = self.client.models.list()
            # Filter to only fine-tuned models
            fine_tuned_models = [model for model in models.data if "ft-" in model.id]
            return fine_tuned_models
        except Exception as e:
            return f"Error listing models: {str(e)}"
    def _normalize_condition(self, condition):
        """Normalize medical condition text for better matching"""
        condition = condition.lower().strip()
        # Remove common prefixes/suffixes
        prefixes = ["a case of ", "patient has ", "diagnosed with "]
        for prefix in prefixes:
            if condition.startswith(prefix):
                condition = condition[len(prefix):]
        # Remove punctuation
        condition = condition.rstrip('.,;:')
        return condition.strip()
