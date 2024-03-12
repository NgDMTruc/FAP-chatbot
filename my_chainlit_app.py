from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

from tools.utils import load_configs


class QABot:
    """A class representing a question answering bot.

    Attributes:
        embeddings (HuggingFaceEmbeddings): The embeddings used by the bot.
        db (FAISS): The FAISS vector store used by the bot.
        llm (HuggingFacePipeline): The Hugging Face language model (LLM) used by the bot.
        qa_prompt (PromptTemplate): The custom prompt template for question answering.
        memory (ConversationBufferMemory): The memory buffer used by the bot.
        qa (RetrievalQA): The question answering chain used by the bot.
    """
    
    def __init__(self):
        """Initialize the QABot instance."""
        config = load_configs()
        self.embeddings = self.load_embed(config['embedding_model'])
        self.db = FAISS.load_local(config['dataset']['faiss_db'], self.embeddings, allow_dangerous_deserialization=True)
        self.llm = self.load_llm(config['llm'])
        self.qa_prompt = self.set_custom_prompt()
        self.memory = self.return_memory()
        self.qa = self.retrive_qa_chain(self.llm, self.qa_prompt, self.db)

    def load_embed(self, embedding_config):
        """Load HuggingFace embeddings based on the given configuration.

        Args:
            embedding_config (dict): Configuration for the embeddings, including the model path, device, and normalization.

        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings initialized with the specified configuration.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_config['path'],     
            model_kwargs={'device': embedding_config.get('device', 'cpu')}, 
            encode_kwargs={'normalize_embeddings': embedding_config.get('normalize_embeddings', False)}
        )
        return embeddings

    def load_llm(self, llm_config):
        """Load a HuggingFace language model (LLM) based on the given configuration.

        Args:
            llm_config (dict): Configuration for the language model, including the model path, temperature, max length, and mode.

        Returns:
            HuggingFacePipeline: An instance of HuggingFacePipeline initialized with the specified configuration.
        """
        model_name = llm_config['path']
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=llm_config.get('max_length', 512))
        question_answerer = pipeline(
            llm_config['mode'], # "text-generation"
            model=model_name,
            tokenizer=tokenizer,
            return_tensors='pt',
            max_new_tokens = 64,
        )
        llm = HuggingFacePipeline(
            pipeline=question_answerer,
            model_kwargs={"temperature": llm_config.get('temperature', 0.7), "max_length": llm_config.get('max_length', 512)},
        )
        return llm

    def set_custom_prompt(self):
        """Set a custom prompt template for question answering.

        Returns:
            PromptTemplate: An instance of PromptTemplate initialized with the specified template.
        """
        prompt_template = """Sử dụng thông tin dưới đây để trả lời câu hỏi của người dùng. Nếu như không biết đáp án, hãy chỉ nói 'Tôi không biết', đừng cố tạo ra câu trả lời.

                            Thông tin hỗ trợ: {context}
                            Câu hỏi: {question}

                            Hãy trả lời chính xác câu hỏi một cách thân thiện.
                            Câu trả lời:
                            """
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        return prompt

    def return_memory(self):
        """Return the memory buffer used by the bot.

        Returns:
            ConversationBufferMemory: An instance of ConversationBufferMemory.
        """
        memory = ConversationBufferMemory(
            memory_key="history",
            input_key="question"
        )
        return memory

    def retrive_qa_chain(self, llm, prompt, db):
        """Retrieve a question answering chain based on the provided components.

        Args:
            llm (HuggingFacePipeline): Language model for question answering.
            prompt (PromptTemplate): Custom prompt template for question answering.
            db (FAISS): FAISS vector store for document retrieval.

        Returns:
            RetrievalQA: A question answering chain initialized with the specified components.
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=db.as_retriever(search_kwargs={'k':3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        return qa_chain

    def final(self, query):
        """Perform question answering based on the provided query.

        Args:
            query (str): Query for question answering.

        Returns:
            str: Answer to the query.
        """
        answer = self.qa.invoke({"query": query})
        return answer

    def return_history(self):
        """Return the memory buffer content.

        Returns:
            dict: The content of the memory buffer.
        """
        return self.memory.load_memory_variables({})


if __name__ == '__main__':
    bot = QABot()
    query = 'Bảo lưu như thế nào?'
    qa_result = bot.final(query)
    print(qa_result)



# @cl.on_chat_start
# async def start():
#   chain = qa_bot()
#   msg = cl.Message(content="Khởi động bot...")
#   await msg.send()
#   msg.content = "Xin chào, chào mừng bạn đến với FAP Chatbot. Câu hỏi của bạn là gì?"
#   await msg.update()
#   cl.user_session.set("chain", chain)
  
# @cl.on_message
# async def main(message):
#   chain = cl.user_session.get("chain")
#   cb = cl.AsyncLangchainCallbackHandler(
#       stream_final_answer = True, answer_prefix_tokens = ["CUỐI CÙNG", "CÂU TRẢ LỜI"]
#   )
#   cb.answer_reached = True
#   res = await chain.acall(message.content, callbacks=[cb])
#   answer = res["result"]
#   sources = res["source_documents"]

#   if sources:
#     answer += f"\nNguồn:" +str(sources)
#   else:
#     answer += f"\nKhông tìm thấy nguồn"
#   await cl.Message(content=answer).send()
