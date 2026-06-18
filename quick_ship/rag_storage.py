#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def setup_rag():
    # Knowledge documents
    documents = [
        Document(page_content="Refund Policy: Users can request a full refund within 30 days of purchase if the package is lost or damaged. Refunds take 5-7 business days to process.", metadata={"source": "refund_policy"}),
        Document(page_content="Delivery Policy: Standard delivery takes 3-5 business days. Next-day delivery is available for premium users. Drivers must log delivery updates instantly.", metadata={"source": "delivery_policy"}),
        Document(page_content="FAQ: What if my package is missing? Check with your neighbors first, then contact support. Can I change my address? Address modifications are allowed only before dispatch.", metadata={"source": "FAQ"}),
        Document(page_content="Internal Operations SOP: Drivers must accept orders assigned via the system. Upon handover, status must be updated to 'Delivered' inside the API immediately.", metadata={"source": "internal_sop"})
    ]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store into local vector database
    vector_db = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory="./chroma_db"
    )
    print("RAG System Vectorized and Saved locally.")

if __name__ == "__main__":
    setup_rag()