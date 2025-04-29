# rag_model.py
import os
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
import pandas as pd

class MedicalRAGModel:
    def __init__(self):
        # 初始化OpenAI API密钥
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)
        self.embeddings = OpenAIEmbeddings()
        
        # 创建向量存储
        self._create_vector_store()
        
        # 初始化检索QA链
        self._initialize_qa_chains()
    
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
        # 创建中文医疗诊断提示模板
        template_zh = """
        你是一位经验丰富的医疗AI助手ClinixBot。基于患者的症状描述和我们的医疗知识库，请提供准确的初步诊断。
        
        医疗知识库上下文:
        {context}
        
        患者症状描述: {question}
        
        请按照以下格式回答:
        1. 初步诊断：[可能的疾病及其概率]
        2. 症状分析：[分析患者描述的症状与疾病的关联]
        3. 建议检查：[如有必要，建议进行的医学检查]
        4. 用药建议：[如适用，建议的药物治疗]
        5. 就医建议：[是否需要就医，以及建议的科室]
        
        重要提示：如果无法确定诊断或症状严重，务必建议患者及时就医。你不是医生，你的建议不能替代专业医疗咨询。
        """
        
        # 创建英文医疗诊断提示模板
        template_en = """
        You are ClinixBot, an experienced medical AI assistant. Based on the patient's symptom description and our medical knowledge base, please provide an accurate preliminary diagnosis.
        
        Medical knowledge context:
        {context}
        
        Patient symptom description: {question}
        
        Please respond in the following format:
        1. Preliminary diagnosis: [Possible diseases and their probabilities]
        2. Symptom analysis: [Analysis of the patient's described symptoms and their association with the disease]
        3. Recommended examinations: [If necessary, recommended medical examinations]
        4. Medication recommendations: [If applicable, recommended drug treatment]
        5. Medical advice: [Whether medical attention is needed, and recommended departments]
        
        Important note: If you cannot determine a diagnosis or the symptoms are severe, be sure to advise the patient to seek medical attention promptly. You are not a doctor, and your advice cannot replace professional medical consultation.
        """
        
        QA_PROMPT_ZH = PromptTemplate(
            template=template_zh, 
            input_variables=["context", "question"]
        )
        
        QA_PROMPT_EN = PromptTemplate(
            template=template_en, 
            input_variables=["context", "question"]
        )
        
        # 初始化中文QA链
        self.qa_chain_zh = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT_ZH}
        )
        
        # 初始化英文QA链
        self.qa_chain_en = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT_EN}
        )
    
    def get_diagnosis(self, symptoms_description, lang="zh"):
        """基于症状描述获取诊断结果"""
        try:
            # 根据语言选择使用的QA链
            qa_chain = self.qa_chain_zh if lang == "zh" else self.qa_chain_en
            
            result = qa_chain({"query": symptoms_description})
            return {
                "diagnosis": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            # 根据语言选择错误消息
            error_msg = f"诊断过程中出现错误: {str(e)}" if lang == "zh" else f"Error during diagnosis: {str(e)}"
            return {
                "diagnosis": error_msg,
                "sources": []
            }
    
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
                """
                system_message = "You are an experienced pharmacist focused on providing accurate medication advice to patients."
        
            # API调用
            response = openai.chat.completions.create(
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