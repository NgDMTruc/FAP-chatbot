from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub 
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import chainlit as cl
import getpass
import os

class QABot:
    def __init__(self):
        self.embeddings = self.load_embed()
        self.db = FAISS.load_local("/kaggle/input/me5new/me5_new", self.embeddings, allow_dangerous_deserialization=True)
        self.llm = self.load_llm_2()
        self.qa_prompt = self.set_custom_prompt()
        self.memory = self.return_memory()
        self.qa = self.retrive_qa_chain()

    def load_llm_2(self):
        os.environ['HUGGING_FACE_HUB_API_KEY'] = getpass.getpass('Hugging face api key:')
        repo_id = 'vilm/vinallama-2.7b-chat'  # has 3B parameters: https://huggingface.co/lmsys/fastchat-t5-3b-v1.0
        llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGING_FACE_HUB_API_KEY'],
                            repo_id=repo_id,
                            model_kwargs={'temperature':1e-10, 'max_new_tokens':64})
        
        return llm

    def load_embed(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        return embeddings

    def load_llm(self):
        model_name = "vilm/vinallama-2.7b-chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
        question_answerer = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            return_tensors='pt',
            max_new_tokens=64,
        )
        llm = HuggingFacePipeline(
            pipeline=question_answerer,
            model_kwargs={"temperature": 0.7, "max_length": 512},
        )
        return llm

    def set_custom_prompt(self):
        prompt_template = """Sử dụng thông tin dưới đây để trả lời câu hỏi của người dùng. Nếu như không biết đáp án, hãy chỉ nói 'Tôi không biết', đừng cố tạo ra câu trả lời.

                            Thông tin hỗ trợ: {context}
                            Câu hỏi: {question}

                            Hãy trả lời chính xác câu hỏi một cách thân thiện.
                            Câu trả lời:
                            """
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        return prompt

    def return_memory(self):
        memory = ConversationBufferMemory(
            memory_key="history",
            input_key="question"
        )
        return memory

    def retrive_qa_chain(self):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.qa_prompt,
                "memory": self.memory
            }
        )
        return qa_chain

    def final(self, query):
        answer = self.qa({"query": query})
        return answer

    def return_history(self):
        return self.memory.load_memory_variables({})
    
