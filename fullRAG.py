from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import textwrap
from multiprocessing import Pool, cpu_count

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


model = None

def chunk_text(text, max_chars=600):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    paragraphs_check = []
    current = ""
    cpt = 0
    for p in paragraphs:
        p = f'Doc Number {cpt}. content: ['+ p
        p = p + ']'
        cpt += 1
        paragraphs_check.append(p)
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + " " + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks, paragraphs_check


def encode_text(chunks):
    global model
    """ if model == None:
        model = SentenceTransformer('all-miniLM-L6-v2') """
    
    # embed all chunks
    print('\nEncoding chunks...')
    chunk_embeddings = model.encode(chunks, batch_size= 16, show_progress_bar=False, convert_to_numpy= True, normalize_embeddings= True)
    print(f'Embeddings shape :{chunk_embeddings.shape}')
    return(chunk_embeddings)


def search(query_embedding, chunks,  chunk_embedding, k = 3):
  # calculate cosine similarity   
    sim = cosine_similarity([query_embedding], chunk_embedding)
    # find similar chunks indexes
    top_idx = sim[0].argsort()[::-1][:k]

    # Find corresponding chunks to return
    top_chunks = []
    chunks_only = []
    for idx in top_idx:
        chunks_only.append(chunks[idx])
        top_chunks.append({'similarity' : sim[0][idx], 'text': chunks[idx]})
    return chunks_only, top_chunks


def promptLLM(system_prompt, user_prompt, query, model_name):
    
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': model_name,
        'prompt': f" {system_prompt}.\nAre you ready ? Here is the context:\n{user_prompt} \n\nQuestion: {query}\n\nRemember, answer based ONLY on the context above.",
        'stream': False    
    } 

    resp = requests.post(url = url, json= data, stream= False)
    resp.raise_for_status()
    out = resp.json()
    return(out['response'])


def init_worker():
    global model
    if model is None:
        print('loading model in worker...')
        model = SentenceTransformer('all-miniLM-L6-v2', device= 'cuda')
        print('model loaded in worker...')

def multiprocess_encoding(chunks):
    # number of cores to use
    n_proc = max(1, cpu_count()-5)
    print(f'Cores To use: {n_proc}')

    # Devide chunks into batches each to be precessed in a separate core
    print('affecting chunks ...')
    chunk_splits = np.array_split(chunks, n_proc)
    
    # Launch multiprocessing
    print('Multiprocessing ...')

    #  as arg
    with Pool(processes= n_proc, initializer= init_worker) as pool:
        print(f'encode_text: {encode_text}')
        results =pool.map(encode_text, chunk_splits)

    print('Merging back results ...')
    return np.vstack(results)

def multiprocess_encoding_builtin(model, chunks, batch_size = 32):
    print('Starting multi process...')
    pool = model.start_multi_process_pool()
    print('Mult-process encoding...')
    embedded_chunks =  model.encode(chunks, pool= pool, batch_size= batch_size, normalize_embeddings = True)
    print('Closing Multiprocesses!')
    model.stop_multi_process_pool(pool)
    print('Chunks enconding done!')
    return embedded_chunks


def customRAG(context, query, models, best_chunks= None):

    # retrieving top (k) similar chunks to query
    # Either can be entered as argument ... (1)
    # Or ...
    # We chunk and encode the context, encode the query. 
    # calculate cosiane similarity between query and each chunks (384 dimension vectors)
    # Retrieve (k) closest verctors to query vectors in terms of similarity
    # Retrieve (k) corresponding chunks... and Voila ... (2)

    if best_chunks:
        retrieved_chunks = best_chunks
    else:
        # embedding query
        print('Enconding Query')
        query_embedding = model.encode(query, batch_size= 16, show_progress_bar= True, convert_to_numpy= True, normalize_embeddings= True )
        print(f'Embeddings shape :{query_embedding.shape}')

        # chunking text
        chunks, _ = chunk_text(context)

        # embedding chunks in multiprocess
        print('Enconding Chunks...')
        chunks_embeddings = multiprocess_encoding_builtin(model, chunks) 
        print(f'Quick check: chunks: {len(chunks_embeddings)}, one chunk embedding: {len(chunks_embeddings[0])} ')
        retrieved_chunks, _ = search(query_embedding, chunks, chunks_embeddings, k= 5)


    # So far we have our best Chunks, one way or another... 
    # Part 2 is using them as context along with the Query to promt ou LLM
    # Prompt Preparation
    system_prompt=( "You are an assistant that answers questions based ONLY on the provided context. "
        "Give clearest full answer based on the context only, and quote from context to back up your answer"
        "BUT, If the context does not contain the answer, say it is not mentionned. "
        "Do not invent information."
        "At the end, make a separation from previous text, and answer from your own knowledge even out of provided context. \n\n")
        
    context_text = "\n\n".join(retrieved_chunks)
    user_prompt= f"Context:\n{context_text}"

    # sending POST Ollama server on localhost sepecifying model and prompt
    answers = []
    for model_name in models:
        answer = promptLLM(system_prompt, user_prompt, query, model_name)
        answers.append({model_name:answer})
    return answers, retrieved_chunks


if __name__ == '__main__':

    # preps
    model = SentenceTransformer('all-miniLM-L6-v2', device= 'cuda')    
    models = ['llama3', 'qwen2.5:7b', 'mistral:7b']
    best_chunks = None
    query = 'What Did Donald Trump say regarding immigrants ?'

    # Execution
    with open('data/fake.txt', 'r', encoding = 'utf-8', errors = 'ignore') as file:
        text = file.read()
    print(next(model.parameters()).device)
    answers, best_chunks = customRAG(text, query, models, best_chunks)

    # Results Display
    print('~'*80)
    print('\n\nQuestion: \n\n')
    print(query)
    print('='*80)

    print('Answers: \n\n')
    if len(models) != len(answers):
        print('Incoherence !')
    for answer in answers:
        for key,value in answer.items():
            print('='*80)
            print(f'model: {key}\n\n')
            print(value)
            print('='*80)