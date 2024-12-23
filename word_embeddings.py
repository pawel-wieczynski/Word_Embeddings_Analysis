import requests
import re
from gensim.models.keyedvectors import load_word2vec_format
import numpy as np
# import nltk
from nltk.corpus import stopwords
# nltk.download("stopwords")

class TextReader:
    """
    Downloading data from Project Gutenberg.
    """

    def __init__(self, book_id: str) -> None:
        self.book_id = book_id
        self.url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        self.text = None
    
    def download_text(self) -> str:
        response = requests.get(self.url)
        self.text = response.text

class TextPreprocessor:
    """
    Preprocessing steps:
        - Convert to lowercase (embeddings and stop words should be lowercase as well).
        - Remove special character and numbers / keep only Latin letters.
        - Tokenize text.
        - Remove stop words.
    """

    def __init__(self, language: str) -> None:
        self.stop_words = stopwords.words(language)
    
    def preprocess_text(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        return tokens

class WordEmbeddings:
    def __init__(self, embedding_path: str) -> None:
        self.embedding_path = embedding_path
        self.embeddings = None
    
    def load_embeddings(self) -> None:
        self.embeddings = load_word2vec_format(self.embedding_path, binary = True)

    def embed_text(self, tokens: list[str]) -> list[np.ndarray]:
        return [self.embeddings[word] for word in tokens if word in self.embeddings]

    def calculate_coverage(self, tokens: list[str]) -> float:
        n_tokens = len(tokens)
        n_covered = sum(token in self.embeddings for token in tokens)
        return n_covered / n_tokens if n_tokens > 0 else 0.0
