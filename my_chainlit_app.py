# !pip install -q langchain
# !pip install -q torch
# !pip install -q transformers
# !pip install -q sentence-transformers
# !pip install -q datasets
# !pip install -q faiss-cpu
# !pip install -q chainlit

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
import chainlit as cl

def load_embed():
  embeddings = HuggingFaceEmbeddings(
      model_name="intfloat/multilingual-e5-large",     # Provide the pre-trained model's path
      model_kwargs={'device':'cpu'}, # Pass the model configuration options
      encode_kwargs={'normalize_embeddings': False} # Pass the encoding options
  )
  return embeddings

def load_llm():
  model_name = "vilm/vinallama-2.7b-chat"

  # Load the tokenizer associated with the specified model
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

  # Define a question-answering pipeline using the model and tokenizer
  question_answerer = pipeline(
      "question-answering",
      model=model_name,
      tokenizer=tokenizer,
      return_tensors='pt'
  )

  # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
  # with additional model-specific arguments (temperature and max_length)
  llm = HuggingFacePipeline(
      pipeline=question_answerer,
      model_kwargs={"temperature": 0.7, "max_length": 512},
  )
  return llm

prompt_template="""Sử dụng thông tin dưới đây để trả lời câu hỏi của người dùng. Nếu như không biết đáp án, hãy chỉ nói 'Tôi không biết', đừng cố tạo ra câu trả lời.

Thông tin hỗ trợ: {}
Câu hỏi: {}

Hãy trả lời chính xác câu hỏi một cách thân thiện.
Câu trả lời:
"""
def set_custom_prompt():
  prompt = PromptTemplate(template=prompt_template, input_variables=['context', ' question'])
  return prompt

def retrieve_qa_chain(llm, prompt, db):
  qa_chain = RetrievalQA.from_chain_type(
      llm = llm,
      chain_type = "stuff", 
      retriever = db.as_retriever(search_kwargs={'k':2}),
      return_source_documents = True,
      chain_type_kwargs = {'prompt':prompt}
  )

  return qa_chain

def qa_bot():
  embeddings = load_embed()
  db = FAISS.load_local("/content/drive/MyDrive/Data/faiss_index", embeddings, allow_dangerous_deserialization=True)
  llm = load_llm()
  qa_prompt = set_custom_prompt()
  qa = retrieve_qa_chain(llm, qa_prompt, db)

  return qa

def final(query):
  qa_result = qa_bot()
  response = qa_result({'query':query})

  return response

@cl.on_chat_start
async def start():
  chain = qa_bot()
  msg = cl.Message(content="Khởi động bot...")
  await msg.send()
  msg.content = "Xin chào, chào mừng bạn đến với FAP Chatbot. Câu hỏi của bạn là gì?"
  await msg.update()
  cl.user_session.set("chain", chain)
  
@cl.on_message
async def main(message):
  chain = cl.user_session.get("chain")
  cb = cl.AsyncLangchainCallbackHandler(
      stream_final_answer = True, answer_prefix_tokens = ["CUỐI CÙNG", "CÂU TRẢ LỜI"]
  )
  cb.answer_reached = True
  res = await chain.acall(message.content, callbacks=[cb])
  answer = res["result"]
  sources = res["source_documents"]

  if sources:
    answer += f"\nNguồn:" +str(sources)
  else:
    answer += f"\nKhông tìm thấy nguồn"
  await cl.Message(content=answer).send()
