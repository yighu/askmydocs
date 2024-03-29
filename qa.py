import os
import sys
import constants

from langchain_openai import OpenAIEmbeddings
#from langchain_openai import ChatOpenAI
#from langchain.document_loaders import TextLoader
#from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"]=constants.APIKEY
underlying_embeddings = OpenAIEmbeddings()

store = LocalFileStore(constants.CACHE)
FASIS_INDEX=constants.FASIS_INDEX
DOCS_FOLDER=constants.DOCS_FOLDER
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)

def loadAndCache():
    loader = PyPDFDirectoryLoader(DOCS_FOLDER)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500, add_start_index=True)
    documents = text_splitter.split_documents(loader.load())
    db = FAISS.from_documents(documents, cached_embedder)
    db.save_local(FASIS_INDEX)
   
def showStore():
    print(list(store.yield_keys())[:5])

def getRetriver():
    db = FAISS.load_local(FASIS_INDEX, underlying_embeddings)
    return db.as_retriever()
db = FAISS.load_local(FASIS_INDEX, underlying_embeddings, allow_dangerous_deserialization=True)
def findDocs(question):
    #db = FAISS.load_local(FASIS_INDEX, underlying_embeddings,allow_dangerous_deserialization=True)
    docs= db.similarity_search_with_score(question)
    return docs
    '''
    retriever = getRetriver()
    docs = retriever.invoke(question) 
    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)
        #print(doc[1])
        print("-------------------")
    '''

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

def createResponse(query):
    docs=findDocs(query)
    context_text="\n\n---\n\n".join([doc.page_content for doc, _score in docs]) #could add filter based on score
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    return response_text    
    '''
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    '''


#loadAndCache()
print(createResponse("如何教育孩子?"))

