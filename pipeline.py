import word_embeddings
import correlations
import numpy as np

class TextAnalysisPipeline:
    def __init__(self, book_id: str, language: str, embedding_path: str):
        self.book_id = book_id
        self.language = language
        self.embedding_path = embedding_path

        self.reader = word_embeddings.TextReader(book_id)
        self.preprocessor = word_embeddings.TextPreprocessor(language)
        self.embedder = word_embeddings.WordEmbeddings(embedding_path)

        self.raw_text = None
        self.tokens = None
        self.vectors = None

        self.lags = []
        # self.autocorrelation_pearson = None
        self.autocorrelation_cosine = []
        self.power_law = None

    def run_pipeline(self):
        # Step 1: download text from Project Gutenberg
        self.reader.download_text()
        self.raw_text = self.reader.text
        # Step 2: preprocessing and tokenization
        self.tokens = self.preprocessor.preprocess_text(self.raw_text)
        # Step 3: words to vectors
        self.embedder.load_embeddings()
        self.vectors = np.asarray(self.embedder.embed_text(self.tokens))
        # Step 4. calculate autocorrelation
        self.calculate_autocorrelation()
        # Step 5. fit power law
    
    def calculate_autocorrelation(self):
        max_lag = 0.5 * (len(self.vectors) - 1)
        current_lag = 1
        while current_lag < max_lag:
            self.lags.append(current_lag)
            current_acf = correlations.calculate_cosine_correlation(self.vectors, L = current_lag)
            self.autocorrelation_cosine.append(current_acf)
            current_lag = int(np.ceil(current_lag * 1.1))
