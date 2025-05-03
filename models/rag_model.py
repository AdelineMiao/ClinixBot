# rag_model.py
import os
import openai
from openai import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI as ChatOpenAINew
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
from collections import Counter
from models.fine_tuned_model import FineTunedMedicalModel

warnings.filterwarnings('ignore', category=FutureWarning)

class MedicalRAGModel:
    def __init__(self, use_fine_tuned=False, fine_tuned_model_name=None):
        # 初始化OpenAI API密钥
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # 创建OpenAI客户端实例（用于直接API调用）
        self.client = OpenAI()
        
        # 使用更新的导入路径创建LLM和嵌入
        try:
            # 尝试使用新版API
            self.llm = ChatOpenAINew(model_name="gpt-4-turbo", temperature=0.2)
            self.embeddings = OpenAIEmbeddings()
        except ImportError:
            # 如果新版API不可用，使用社区版本
            self.llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)
            self.embeddings = OpenAIEmbeddings()
        
        # 加载医疗数据并创建DataFrame
        self._load_medical_data()
        
        # 创建向量存储
        self._create_vector_store()
        
        # 初始化检索QA链
        self._initialize_qa_chains()
        
        # Initialize fine-tuned model if requested
        self.use_fine_tuned = use_fine_tuned
        if use_fine_tuned:
            self.fine_tuned_model = FineTunedMedicalModel(model_name=fine_tuned_model_name)
    
    def _load_medical_data(self):
        """加载医疗记录数据"""
        try:
            self.medical_df = pd.read_csv("./data/hospital_records_2021_2024_with_bills.csv")
            # 确保DataFrame有所需的列
            required_columns = ["Patient ID", "Medical Condition", "Doctor's Notes", "Treatments"]
            missing_columns = [col for col in required_columns if col not in self.medical_df.columns]
            if missing_columns:
                print(f"警告: 数据集缺少以下列: {missing_columns}")
                
            # 正确修复缺失值（避免链式赋值的警告）
            for col in self.medical_df.columns:
                if self.medical_df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(self.medical_df[col]):
                        # 使用更安全的方法替换缺失值
                        median_value = self.medical_df[col].median()
                        self.medical_df[col] = self.medical_df[col].fillna(median_value)
                    else:
                        # 文本列使用空字符串填充
                        self.medical_df[col] = self.medical_df[col].fillna("")
                        
        except Exception as e:
            print(f"加载医疗数据时出错: {str(e)}")
            self.medical_df = pd.DataFrame()
    
    def _create_vector_store(self):
        """创建和加载向量存储"""
        # 检查是否有现有的向量存储
        if os.path.exists("./vector_store"):
            self.vector_store = FAISS.load_local("./vector_store", self.embeddings, allow_dangerous_deserialization=True)
        else:
            # 加载CSV数据
            loader = CSVLoader("./data/hospital_records_2021_2024_with_bills.csv")
            documents = loader.load()
            
            # 切分文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # 创建向量存储
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            # 保存向量存储以供将来使用
            self.vector_store.save_local("./vector_store")
    
    def _initialize_qa_chains(self):
        """初始化中英文检索QA链"""
        # 创建中文医疗诊断提示模板 - 更新以包含相似案例统计信息而非具体患者信息
        template_zh = """
        你是一位经验丰富的医疗AI助手ClinixBot。基于患者的症状描述和我们的医疗知识库，请提供准确的初步诊断。
        
        医疗知识库上下文:
        {context}
        
        历史相似病例统计:
        {similar_cases_stats}
        
        患者症状描述: {query}
        
        请按照以下格式回答:
        1. 初步诊断：[可能的疾病及其概率]
        2. 症状分析：[分析患者描述的症状与疾病的关联]
        3. 相似病例参考：[提到在数据库中有多少相似病例，以及这些病例的主要诊断结果分布，不要提及任何具体患者姓名或ID]
        4. 建议检查：[如有必要，建议进行的医学检查]
        5. 用药建议：[如适用，建议的药物治疗]
        6. 就医建议：[是否需要就医，以及建议的科室]
        
        重要提示：
        1. 绝对不要在回复中包含任何患者的姓名、ID或其他可能识别个人的信息
        2. 如果无法确定诊断或症状严重，务必建议患者及时就医
        3. 你不是医生，你的建议不能替代专业医疗咨询
        """
        
        # 创建英文医疗诊断提示模板 - 更新以包含相似案例统计信息而非具体患者信息
        template_en = """
        You are ClinixBot, an experienced medical AI assistant. Based on the patient's symptom description and our medical knowledge base, please provide an accurate preliminary diagnosis.
        
        Medical knowledge context:
        {context}
        
        Historical similar cases statistics:
        {similar_cases_stats}
        
        Patient symptom description: {query}
        
        Please respond in the following format:
        1. Preliminary diagnosis: [Possible diseases and their probabilities]
        2. Symptom analysis: [Analysis of the patient's described symptoms and their association with the disease]
        3. Similar case reference: [Mention how many similar cases exist in the database and the distribution of their primary diagnoses, without mentioning any specific patient names or IDs]
        4. Recommended examinations: [If necessary, recommended medical examinations]
        5. Medication recommendations: [If applicable, recommended drug treatment]
        6. Medical advice: [Whether medical attention is needed, and recommended departments]
        
        Important notes:
        1. NEVER include any patient names, IDs, or other personally identifiable information in your response
        2. If you cannot determine a diagnosis or the symptoms are severe, be sure to advise the patient to seek medical attention promptly
        3. You are not a doctor, and your advice cannot replace professional medical consultation
        """
        
        # 创建中英文提示模板
        self.QA_PROMPT_ZH = PromptTemplate(
            template=template_zh, 
            input_variables=["context", "query", "similar_cases_stats"]
        )
        
        self.QA_PROMPT_EN = PromptTemplate(
            template=template_en, 
            input_variables=["context", "query", "similar_cases_stats"]
        )
        
        # 创建检索器
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    def _find_similar_cases(self, symptoms_description, top_n=5):
        """查找与当前症状描述最相似的历史病例"""
        if self.medical_df.empty:
            return "无法获取历史病例信息。"
        
        try:
            # 使用OpenAI获取症状描述的嵌入向量
            symptom_embedding_response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=symptoms_description
            )
            symptom_embedding = symptom_embedding_response.data[0].embedding
            
            # 为医疗数据库中的所有症状描述获取嵌入向量
            # 将医生笔记和医疗情况结合作为特征
            features = []
            for _, row in self.medical_df.iterrows():
                combined_text = f"{row['Medical Condition']} {row['Doctor\'s Notes']}"
                features.append(combined_text)
            
            # 批量获取嵌入向量
            feature_embedding_response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=features
            )
            feature_embeddings = [item.embedding for item in feature_embedding_response.data]
            
            # 计算余弦相似度
            similarities = cosine_similarity(
                [symptom_embedding], 
                feature_embeddings
            )[0]
            
            # 获取前N个最相似的案例的索引
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            # 创建相似案例描述
            similar_cases = []
            for idx in top_indices:
                row = self.medical_df.iloc[idx]
                condition = row['Medical Condition']
                treatment = row['Treatments']
                similarity_score = similarities[idx]
                
                case_info = {
                    "condition": condition,
                    "treatment": treatment,
                    "similarity": similarity_score
                }
                similar_cases.append(case_info)
            
            return similar_cases
            
        except Exception as e:
            print(f"查找相似案例时出错: {str(e)}")
            return []
    
    def _format_similar_cases_stats(self, similar_cases, lang="zh"):
        """格式化相似案例统计信息（移除所有患者ID和姓名）"""
        if not similar_cases or isinstance(similar_cases, str):
            return "无相似病例数据" if lang == "zh" else "No similar case data available"
        
        # 收集条件出现频率
        conditions = [case['condition'] for case in similar_cases]
        condition_counts = Counter(conditions)
        
        # 收集治疗方法出现频率
        treatments = [case['treatment'] for case in similar_cases]
        treatment_counts = Counter(treatments)
        
        if lang == "zh":
            # 格式化诊断结果统计
            conditions_stats = []
            for condition, count in condition_counts.items():
                percentage = (count / len(similar_cases)) * 100
                conditions_stats.append(f"- {condition}: {count}例 ({percentage:.1f}%)")
            
            # 格式化治疗方法统计
            treatments_stats = []
            for treatment, count in treatment_counts.items():
                percentage = (count / len(similar_cases)) * 100
                treatments_stats.append(f"- {treatment}: {count}例 ({percentage:.1f}%)")
            
            stats_text = (
                f"找到{len(similar_cases)}个相似病例。\n\n"
                f"诊断结果分布:\n"
                f"{chr(10).join(conditions_stats)}\n\n"
                f"治疗方法分布:\n"
                f"{chr(10).join(treatments_stats)}"
            )
        else:
            # 格式化诊断结果统计
            conditions_stats = []
            for condition, count in condition_counts.items():
                percentage = (count / len(similar_cases)) * 100
                conditions_stats.append(f"- {condition}: {count} cases ({percentage:.1f}%)")
            
            # 格式化治疗方法统计
            treatments_stats = []
            for treatment, count in treatment_counts.items():
                percentage = (count / len(similar_cases)) * 100
                treatments_stats.append(f"- {treatment}: {count} cases ({percentage:.1f}%)")
            
            stats_text = (
                f"Found {len(similar_cases)} similar cases.\n\n"
                f"Diagnosis distribution:\n"
                f"{chr(10).join(conditions_stats)}\n\n"
                f"Treatment distribution:\n"
                f"{chr(10).join(treatments_stats)}"
            )
        
        return stats_text
    
    def get_diagnosis(self, symptoms_description, lang="zh", use_fine_tuned_override=None):
        """基于症状描述获取诊断结果，支持使用微调模型"""
        # 确定是否使用微调模型
        use_fine_tuned = use_fine_tuned_override if use_fine_tuned_override is not None else self.use_fine_tuned
        
        try:
            if use_fine_tuned and hasattr(self, 'fine_tuned_model'):
                # 使用微调模型进行诊断
                diagnosis_text = self.fine_tuned_model.get_diagnosis(symptoms_description, lang=lang)
                
                return {
                    "diagnosis": diagnosis_text,
                    "sources": ["使用微调模型进行此次诊断" if lang == "zh" else "Fine-tuned model was used for this diagnosis"],
                    "similar_cases_stats": "",
                    "model_type": "fine_tuned"
                }
            else:
                # 使用现有的RAG方法
                # 查找相似病例
                similar_cases = self._find_similar_cases(symptoms_description)
                
                # 生成统计信息而非具体病例
                similar_cases_stats = self._format_similar_cases_stats(similar_cases, lang)
                
                # 检索相关文档
                docs = self.retriever.get_relevant_documents(symptoms_description)
                
                # 合并所有文档内容以创建上下文
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # 根据语言选择提示模板
                prompt = self.QA_PROMPT_ZH if lang == "zh" else self.QA_PROMPT_EN
                
                # 创建并运行LLM链
                chain = LLMChain(llm=self.llm, prompt=prompt)
                
                # 使用链直接调用，传递所有必需的参数
                response = chain.invoke({
                    "query": symptoms_description,
                    "context": context,
                    "similar_cases_stats": similar_cases_stats
                })
                
                # 添加模型类型到响应
                result = {
                    "diagnosis": response["text"],
                    "sources": [doc.page_content for doc in docs],
                    "similar_cases_stats": similar_cases_stats,
                    "model_type": "rag"
                }
                
                return result
        except Exception as e:
            # 根据语言选择错误消息
            error_msg = f"诊断过程中出现错误: {str(e)}" if lang == "zh" else f"Error during diagnosis: {str(e)}"
            print(f"错误详情: {str(e)}")  # 在服务器端记录详细错误
            return {
                "diagnosis": error_msg,
                "sources": [],
                "similar_cases_stats": "",
                "model_type": "error"
            }
    
    def get_follow_up_question(self, prompt, lang="zh"):
        """生成一个进一步需要询问的问题，以帮助缩小诊断范围"""
        try:
            system_prompt = "你是临床医生助手，请根据用户描述的症状提出一个进一步需要询问的问题。" if lang == "zh" else "You are a clinical assistant. Given the patient's symptoms, ask one important follow-up question to narrow down the diagnosis."
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"获取追问失败: {str(e)}" if lang == "zh" else f"Failed to generate follow-up question: {str(e)}"
    
    def get_medication_recommendations(self, diagnosis, lang="zh"):
        """基于诊断结果推荐药物"""
        try:
            # 根据语言选择提示模板
            if lang == "zh":
                prompt = f"""
                基于以下诊断结果，推荐合适的非处方药物治疗方案：
                
                {diagnosis}
                
                请列出:
                1. 推荐药物名称
                2. 用法用量
                3. 预期效果
                4. 可能的副作用
                5. 注意事项
                
                重要提示: 不要在回复中包含任何患者的个人信息（如姓名或ID）。
                """
                system_message = "你是一位经验丰富的药剂师，专注于为患者提供准确的用药建议。"
            else:
                prompt = f"""
                Based on the following diagnosis results, recommend appropriate over-the-counter medication treatment options:
                
                {diagnosis}
                
                Please list:
                1. Recommended medication
                2. Usage and dosage
                3. Expected effects
                4. Possible side effects
                5. Precautions
                
                Important note: Do not include any patient personal information (such as names or IDs) in your response.
                """
                system_message = "You are an experienced pharmacist focused on providing accurate medication advice to patients."
        
            # API调用
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
        
            return response.choices[0].message.content
        except Exception as e:
            # 根据语言选择错误消息
            return f"获取药物推荐时出现错误: {str(e)}" if lang == "zh" else f"Error getting medication recommendations: {str(e)}"
