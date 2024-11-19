from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
import textwrap

def load_pdf_data(file_path):
    loader = PyMuPDFLoader(file_path=file_path)

    docs = loader.load()

    return docs

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_spliter.split_documents(documents=documents)

    return chunks

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding
        }
    )

def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

def load_qa_chain(retriver, llm, prompt):
    return create_retrieval_chain.from_chain_type(
        llm=llm,
        retriever=retriver, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )

def get_response(query, chain):
    # Getting response from chain
    response = chain({'query': query})
    
    # Wrapping the text for better output in Jupyter Notebook
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)