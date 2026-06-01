import pandas as pd 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_index_data(filepath):
    #read excel file
    df=pd.read_excel(filepath)

    #combine topic and content into one string per row
    chunks =[]
    for _, row in df.iterrows():
        chunk=f"Topic: {row['Topic']}\nContent: {row['Content']}"
        chunks.append(chunk)
    
    #convert chunks to embeddings
    embeddings=embedding_model.encode(chunks)

    #convert float 32 to faiss
    embeddings=np.array(embeddings).astype('float32')

    #create faiss index
    dimension=embeddings.shape[1]
    index=faiss.IndeFlatL2(dimension)
    index.add(embeddings)

    return index, chunks

def retrieve_relevant_chunks (query, index,chunks,top_k=3):
    #convert query to embedding
    query_embedding = embedding_model.encode([query])
    query_embedding=np.array(query_embedding).astype('float32')

    #search for closest chunks
    distance, indices = index.search(query_embedding, top_k)

    #return actual text chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks