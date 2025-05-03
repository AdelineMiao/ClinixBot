import os
from openai import OpenAI
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
        self.client = OpenAI()
        self.llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)
        self.embeddings = OpenAIEmbeddings()
        self._create_vector_store()
        self._initialize_qa_chains()

    def _create_vector_store(self):
        if os.path.exists("./vector_store"):
            self.vector_store = FAISS.load_local("./vector_store", self.embeddings, allow_dangerous_deserialization=True)
        else:
            loader = CSVLoader("./data/hospital_records_2021_2024_with_bills.csv")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            self.vector_store.save_local("./vector_store")

    def _initialize_qa_chains(self):
        template_zh = """
        你是一位经验丰富的医疗AI助手ClinixBot。基于患者的症状描述和我们的医疗知识库，请提供准确的初步诊断。

        医疗知识库上下文:
        {context}

        患者症状描述: {question}

        请按照以下格式回答:
        1. 初步诊断：[可能的疾病及其概率]
        2. 症状分析：[分析患者描述的症状与疾病的关联]
        3. 建议检查：[如有必要，建议进行的医学检查]
        4. 就医建议：[是否需要就医，以及建议的科室]

        重要提示：如果无法确定诊断或症状严重，务必建议患者及时就医。你不是医生，你的建议不能替代专业医疗咨询。
        """

        template_en = """
        You are ClinixBot, an experienced medical AI assistant. Based on the patient's symptom description and our medical knowledge base, please provide an accurate preliminary diagnosis.

        Medical knowledge context:
        {context}

        Patient symptom description: {question}

        Please respond in the following format:
        1. Preliminary diagnosis: [Possible diseases and their probabilities]
        2. Symptom analysis: [Analysis of the patient's described symptoms and their association with the disease]
        3. Recommended examinations: [If necessary, recommended medical examinations]
        4. Medical advice: [Whether medical attention is needed, and recommended departments]

        Important note: If you cannot determine a diagnosis or the symptoms are severe, be sure to advise the patient to seek medical attention promptly. You are not a doctor, and your advice cannot replace professional medical consultation.
        """

        QA_PROMPT_ZH = PromptTemplate(template=template_zh, input_variables=["context", "question"])
        QA_PROMPT_EN = PromptTemplate(template=template_en, input_variables=["context", "question"])

        self.qa_chain_zh = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT_ZH}
        )

        self.qa_chain_en = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT_EN}
        )

    def get_diagnosis(self, symptoms_description, lang="zh"):
        try:
            qa_chain = self.qa_chain_zh if lang == "zh" else self.qa_chain_en
            result = qa_chain({"query": symptoms_description})
            return {
                "diagnosis": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            error_msg = f"诊断过程中出现错误: {str(e)}" if lang == "zh" else f"Error during diagnosis: {str(e)}"
            return {"diagnosis": error_msg, "sources": []}

    def get_follow_up_question(self, prompt, lang="zh"):
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
