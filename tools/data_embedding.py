from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

from tools.utils import load_configs

class Embedder:
    """A class for embedding text documents using Hugging Face models and saving to a FAISS vector store.

    Args:
        configs (dict): A dictionary containing configuration parameters for embedding and dataset loading.
    """

    def __init__(self, configs):
        """Initialize the Embedder class with configuration parameters.

        Args:
            configs (dict): A dictionary containing configuration parameters for embedding and dataset loading.
        """
        self.modelPath = configs['embedding_model']['path']
        self.model_kwargs = {'device': configs['embedding_model']['device']}
        self.encode_kwargs = {'normalize_embeddings': configs['embedding_model']['normalize_embeddings']}
        self.output_path = configs['embedding_model']['output_path']
        self.csv_path = configs['dataset']['csv_dataset']

    def embeder(self):
        """Instantiate an instance of HuggingFaceEmbeddings with the specified parameters.

        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings.
        """
        return HuggingFaceEmbeddings(model_name=self.modelPath,     
                                    model_kwargs=self.model_kwargs,
                                    encode_kwargs=self.encode_kwargs
                                   )
    
    def data_process(self, csv_path):
        """Load and preprocess data from a CSV file.

        Args:
            csv_path (str): The path to the CSV file containing the dataset.

        Returns:
            list: A list of preprocessed text documents.
        """
        loader = CSVLoader(file_path=csv_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)
        return docs

    def embed(self):
        """Embed the dataset and save the embeddings to a FAISS vector store."""
        docs = self.data_process(self.csv_path)
        embeddings = self.embeder()

        db = FAISS.from_documents(docs, embeddings)
        
        db.save_local(self.output_path)

        print(f'Successfully Embedded and Saved to {self.output_path}')


if __name__ == '__main__':
    configs = load_configs()
    embedder = Embedder(configs)
    embedder.embed()
