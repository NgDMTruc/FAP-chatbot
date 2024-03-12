from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from tools.utils import load_configs

def load_embed(embedding_config):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_config['path'],     
        model_kwargs={'device': embedding_config.get('device', 'cpu')}, 
        encode_kwargs={'normalize_embeddings': embedding_config.get('normalize_embeddings', False)}
    )
    return embeddings

def load_llm(llm_config):
    model_name = llm_config['path']

    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=llm_config.get('max_length', 512))

    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        llm_config['mode'], # "question-answering"
        model=model_name,
        tokenizer=tokenizer,
        return_tensors='pt',
        max_new_tokens = 64,
    )

    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm = HuggingFacePipeline(
        pipeline=question_answerer,
        model_kwargs={"temperature": llm_config.get('temperature', 0.7), "max_length": llm_config.get('max_length', 512)},
    )
    return llm

def set_custom_prompt(prompt_template):
  prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
  return prompt

def retrive_qa_chain(llm, prompt, db):
  qa_chain = RetrievalQA.from_chain_type(
      llm = llm,
      chain_type = "stuff", 
      retriever = db.as_retriever(search_kwargs={'k':3}),
      return_source_documents = True,
      chain_type_kwargs = {'prompt':prompt}
  )

  return qa_chain

def qa_bot(query):
    config = load_configs()
    embedding_config = config['embedding_model']
    llm_config = config['llm']
    faiss_db_path = config['dataset']['faiss_db']

    embeddings = load_embed(embedding_config)
    db = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm(llm_config)
    prompt_template="""Sử dụng thông tin dưới đây để trả lời câu hỏi của người dùng. Nếu như không biết đáp án, hãy chỉ nói 'Tôi không biết', đừng cố tạo ra câu trả lời.

Thông tin hỗ trợ: {context}
Câu hỏi: {question}

Hãy trả lời chính xác câu hỏi một cách thân thiện.
Câu trả lời:
"""

    qa_prompt = set_custom_prompt(prompt_template)
    qa = retrive_qa_chain(llm, qa_prompt, db)
    answer = qa.invoke({"query": query})
    return answer

if __name__ == '__main__':
    query = 'Bao luu nhu the nao?'
    qa_result = qa_bot(query)
    print(qa_result)
