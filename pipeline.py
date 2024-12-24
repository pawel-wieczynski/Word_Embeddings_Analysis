import word_embeddings
import correlations
import utils
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt

class TextAnalysisPipeline:
    def __init__(self, book_id: str, source: str, language: str, embedding_path: str):
        """
        Parameters:
            source (str): "PG" for Project Gutenberg website, "SGPC" for local files standardized by Gerlach, Font-Clos (2018).
        """
        self.book_id = book_id
        self.source = source
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
        if self.source == "PG":
            self.reader.download_text()
            self.raw_text = self.reader.text
        elif self.source == "SGPC":
            self.reader.read_tokens()
            self.tokens = self.reader.tokens
        else:
            ValueError("Unknown source. Please use either PG or SGPC.")
        # Step 2: preprocessing and tokenization
        if self.source == "PG":
            self.tokens = self.preprocessor.preprocess_text(self.raw_text)
        else:
            ValueError("Unknown source. Please use either PG or SGPC.")
        # Step 3: words to vectors
        self.embedder.load_embeddings()
        self.vectors = np.asarray(self.embedder.embed_text(self.tokens))
        # Step 4. calculate autocorrelation
        self.calculate_autocorrelation()
        # Step 5. fit power law
        self.fit_power_law()
    
    def calculate_autocorrelation(self):
        max_lag = 0.5 * (len(self.vectors) - 1)
        current_lag = 1
        while current_lag < max_lag:
            self.lags.append(current_lag)
            current_acf = correlations.calculate_cosine_correlation(self.vectors, L = current_lag)
            self.autocorrelation_cosine.append(current_acf)
            current_lag = int(np.ceil(current_lag * 1.1))
    
    def fit_power_law(self):
        popt, pcov = curve_fit(utils.power_law, self.lags, self.autocorrelation_cosine, p0 = [1, -1, 1])
        self.power_law = popt

    def make_plots(self, scales = 'normal', n = 200):
        if scales == 'log':
            lags_this = np.log2(self.lags)
            acf_this = np.log2(np.abs(self.autocorrelation_cosine))
        else:
            lags_this = self.lags
            acf_this = self.autocorrelation_cosine

        lags_fit = np.linspace(min(lags_this), max(lags_this), n)
        acf_fit = utils.power_law(lags_fit, self.power_law[0], self.power_law[1], self.power_law[2])

        plt.figure()
        sns.scatterplot(x = lags_this, y = acf_this)
        sns.lineplot(x = lags_fit, y = acf_fit, color = 'red')
        plt.savefig("plot.png")
