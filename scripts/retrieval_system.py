import os
import re
import unicodedata
import pandas as pd
import numpy as np
from tqdm.auto import tqdm 
import faiss 
from sentence_transformers import SentenceTransformer, CrossEncoder 
from sklearn.metrics import average_precision_score, ndcg_score 


DATA_DIR = 'data'
CORPUS_FILE = os.path.join(DATA_DIR, 'corpus.txt')  
QUERIES_FILE = os.path.join(DATA_DIR, 'queries.txt') 
QRELS_FILE = os.path.join(DATA_DIR, 'qrels.txt')

SBERT_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Retrieval parameters
TOP_K_RETRIEVAL = 100 # Number of documents to retrieve in the first stage (SBERT + FAISS)
TOP_K_RERANKING = 10  # Number of documents to re-rank and return as final results

# --- Text Preprocessing Functions ---
def clean_text(text):
    """
    Performs basic text cleaning:
    - Converts to lowercase.
    - Removes extra whitespace.
    - Removes numbers (optional, depending on if numbers are important for retrieval).
    - Removes punctuation (optional).
    - Normalizes Unicode characters.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower() # Convert to lowercase
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8') # Normalize unicode
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation (keep letters, numbers, whitespace)
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace and strip
    return text

# --- Data Loading and Parsing Functions (UPDATED for .txt files) ---
def parse_corpus_txt(file_path):
    """
    Parses the corpus from a .txt file.
    Assumes each document starts with <DOCNO> and ends with </DOC>,
    and BODY contains the text.
    """
    corpus = {}
    current_docno = None
    current_body = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing Corpus"):
                line = line.strip()
                if line.startswith('<DOCNO>'):
                    current_docno = line.replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
                    current_body = []
                elif line.startswith('<BODY>'):
                    current_body.append(line.replace('<BODY>', '').strip())
                elif line.endswith('</BODY>'):
                    current_body.append(line.replace('</BODY>', '').strip())
                    if current_docno:
                        corpus[current_docno] = clean_text(" ".join(current_body))
                        current_docno = None
                        current_body = []
                elif current_docno and current_body: # If within a document body
                    current_body.append(line)
    except FileNotFoundError:
        print(f"Error: Corpus file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error parsing corpus file: {e}")
        return {}
    print(f"Loaded {len(corpus)} documents from corpus.")
    return corpus

def parse_queries_txt(file_path):
    """
    Parses the queries from a .txt file.
    Assumes each query starts with <num> and <title> contains the query text.
    """
    queries = {}
    current_num = None
    current_title = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing Queries"):
                line = line.strip()
                if line.startswith('<num>'):
                    current_num = line.replace('<num>', '').replace('</num>', '').strip()
                    # Extract only the number from "num: 1"
                    query_id = re.search(r'\d+', current_num).group(0) if re.search(r'\d+', current_num) else current_num
                    current_num = query_id # Store just the ID
                    current_title = []
                elif line.startswith('<title>'):
                    current_title.append(line.replace('<title>', '').strip())
                elif line.startswith('<desc>') or line.startswith('<narr>') or line.startswith('</top>'):
                    # End of title section, process the query
                    if current_num and current_title:
                        queries[current_num] = clean_text(" ".join(current_title))
                        current_num = None
                        current_title = []
                elif current_num and current_title: # If within a title section
                    current_title.append(line)
    except FileNotFoundError:
        print(f"Error: Queries file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error parsing queries file: {e}")
        return {}
    print(f"Loaded {len(queries)} queries.")
    return queries

def parse_qrels(file_path):
    """
    Parses the qrels file and returns a dictionary of {query_id: {doc_id: relevance_score}}.
    Assumes format: query_id Q0 doc_id relevance_score
    """
    qrels = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing Qrels"):
                parts = line.strip().split()
                if len(parts) == 4:
                    query_id, _, doc_id, score = parts
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = int(score)
    except FileNotFoundError:
        print(f"Error: Qrels file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error parsing qrels file: {e}")
        return {}
    print(f"Loaded {len(qrels)} qrels entries.")
    return qrels

# --- Main Retrieval System Logic ---
def run_retrieval_system():
    """
    Orchestrates the entire retrieval process:
    1. Loads data.
    2. Loads SBERT and Cross-Encoder models.
    3. Embeds corpus documents and builds a FAISS index.
    4. Processes each query:
        a. Embeds the query.
        b. Performs initial retrieval using FAISS.
        c. Re-ranks top results using the Cross-Encoder.
    5. Evaluates the results against qrels.
    """
    print("--- Starting Retrieval System ---")

    # 1. Load Data
    print("\nLoading data...")
    # Use the updated parsing functions for .txt files
    corpus = parse_corpus_txt(CORPUS_FILE)
    queries = parse_queries_txt(QUERIES_FILE)
    qrels = parse_qrels(QRELS_FILE)

    if not corpus or not queries or not qrels:
        print("Exiting: Failed to load all necessary data.")
        return

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]

    # 2. Load Models
    print(f"\nLoading SBERT model: {SBERT_MODEL_NAME}...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"Loading Cross-Encoder model: {CROSS_ENCODER_MODEL_NAME}...")
    cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    # 3. Embed Corpus and Build FAISS Index
    print("\nGenerating corpus embeddings...")
    # Encode documents in batches for efficiency
    corpus_embeddings = sbert_model.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True)

    embedding_dim = corpus_embeddings.shape[1]
    print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

    print("Building FAISS index...")
    # Using IndexFlatL2 for L2 distance (Euclidean distance)
    # For cosine similarity, you might normalize embeddings and use IndexFlatIP (inner product)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(corpus_embeddings)
    print(f"FAISS index built with {index.ntotal} documents.")

    # 4. Process Queries
    print("\nProcessing queries and performing retrieval...")
    retrieved_results = {} # Store results for evaluation {query_id: {doc_id: rank}}

    for query_id, query_text in tqdm(queries.items(), desc="Processing Queries"):
        # Generate query embedding
        query_embedding = sbert_model.encode(query_text, convert_to_numpy=True)
        query_embedding = np.expand_dims(query_embedding, axis=0) # Add batch dimension

        # Initial retrieval with FAISS (Stage 1: Candidate Generation)
        # D: distances, I: indices of retrieved documents
        distances, faiss_indices = index.search(query_embedding, TOP_K_RETRIEVAL)
        
        # Get actual document IDs from FAISS indices
        initial_retrieved_doc_ids = [corpus_ids[idx] for idx in faiss_indices[0]]
        # initial_retrieved_scores = [1 - dist for dist in distances[0]] # Convert L2 distance to similarity (higher is better)

        # Prepare pairs for Cross-Encoder re-ranking
        cross_encoder_pairs = []
        for doc_id in initial_retrieved_doc_ids:
            cross_encoder_pairs.append([query_text, corpus[doc_id]])

        if not cross_encoder_pairs:
            # Handle cases where no documents were retrieved by FAISS
            retrieved_results[query_id] = {}
            continue

        # Re-ranking with Cross-Encoder (Stage 2: Re-ranking)
        cross_encoder_scores = cross_encoder_model.predict(cross_encoder_pairs)

        # Combine initial retrieval and re-ranking scores
        # Create a list of (doc_id, cross_encoder_score) tuples
        doc_scores = []
        for i, doc_id in enumerate(initial_retrieved_doc_ids):
            doc_scores.append((doc_id, cross_encoder_scores[i]))

        # Sort by Cross-Encoder score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Store the top K re-ranked results
        current_query_results = {}
        for rank, (doc_id, score) in enumerate(doc_scores[:TOP_K_RERANKING]):
            current_query_results[doc_id] = rank + 1 # Store rank (1-based)
        retrieved_results[query_id] = current_query_results

    # 5. Evaluate Results
    print("\nEvaluating retrieval performance...")
    
    all_map_scores = []
    all_ndcg_scores = []

    for query_id, relevant_docs in qrels.items():
        if query_id not in retrieved_results:
            # If a query from qrels was not processed or had no results, skip or handle as needed
            continue

        retrieved_doc_ranks = retrieved_results[query_id] # {doc_id: rank}
        
        all_candidate_docs_for_query = set(relevant_docs.keys()).union(set(retrieved_doc_ranks.keys()))
        
        if not all_candidate_docs_for_query:
            continue

        true_relevance_for_query = []
        predicted_scores_for_query = [] 

        sorted_candidates = sorted(list(all_candidate_docs_for_query), 
                                   key=lambda doc_id: retrieved_doc_ranks.get(doc_id, TOP_K_RERANKING + 1)) 

        for doc_id in sorted_candidates:
            true_relevance_for_query.append(relevant_docs.get(doc_id, 0)) 
            if doc_id in retrieved_doc_ranks:
                predicted_scores_for_query.append(1.0 / retrieved_doc_ranks[doc_id])
            else:
                predicted_scores_for_query.append(0.0) 

        if true_relevance_for_query and predicted_scores_for_query:
            if 1 in true_relevance_for_query: 
                ap = average_precision_score(true_relevance_for_query, predicted_scores_for_query)
                all_map_scores.append(ap)
            
            ndcg = ndcg_score(np.asarray([true_relevance_for_query]), np.asarray([predicted_scores_for_query]))
            all_ndcg_scores.append(ndcg)

    mean_map = np.mean(all_map_scores) if all_map_scores else 0
    mean_ndcg = np.mean(all_ndcg_scores) if all_ndcg_scores else 0

    print(f"\n--- Evaluation Results (Top {TOP_K_RERANKING} Re-ranked) ---")
    print(f"Mean Average Precision (MAP): {mean_map:.4f}")
    print(f"Mean NDCG: {mean_ndcg:.4f}")
    print("--- Retrieval System Finished ---")

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and place your corpus, queries, and qrels files inside.")
    else:
        run_retrieval_system()

