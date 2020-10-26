import torch
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_random_embeddings(sentences: List[str], length: int = 768) -> torch.Tensor:
    """
    Returns random embeddings of length `length` for each input sentence. Each random
    value is a float between 0 and 1.
    :return: `Tensor` with all the embeddings.
    """
    return torch.rand(len(sentences), length)


tfidf_vectorizer = None


def compute_tfidf_features(sentences: List[str]):
    """
    Returns a matrix representing tfidf features
    :return: `scipy.sparse.csr.csr_matrix` instance
    """
    global tfidf_vectorizer
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    return tfidf_vectorizer.fit_transform(sentences)
