import numpy as np
from logging_utils import load_repository
from src.retrieval_faiss_transistor import retrieve_k_nearest_transistor

repository = load_repository("data/transistor_repository_clean.jsonl")
query_spec = {'target_peaking_db': 8.0, 'noise_limit_mvrms': 1.5}
res = retrieve_k_nearest_transistor(query_spec, repository, k=1)
print(res)
