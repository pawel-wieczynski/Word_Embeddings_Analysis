from scripts import word_embeddings
from scripts import correlations
from scripts import utils
from scripts import coocurrence_matrix
import numpy as np
import pingouin as pg
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt

class TextAnalysisPipeline:
    def __init__(self, book_id: str, source: str, language: str, method: str, window_size: int = 3, sparse: bool = True, embedder: word_embeddings.WordEmbeddings = None):
        """
        Parameters:
            book_id (str)
            source (str): "PG" for Project Gutenberg website, "SGPC" for local files standardized by Gerlach, Font-Clos (2018).
            language (str)
            method (str): Either "cooccurence" or "embeddings".
            window_size (int): Window size for method "cooccurence".
            embedder: Object of class WordEmbeddings.
        """
        self.book_id = book_id
        self.source = source
        self.language = language
        self.method = method
        self.window_size = window_size
        self.sparse = sparse
        self.embedder = embedder

        self.reader = word_embeddings.TextReader(book_id)
        self.preprocessor = word_embeddings.TextPreprocessor(language)

        self.raw_text = None
        self.tokens = None
        self.vectors = None
        self.vocabulary = None

        self.lags = []
        self.autocorrelation_pearson = []
        self.autocorrelation_cosine = []
        self.power_law_pearson = None
        self.power_law_cosine = None
        self.normality = None
        self.coverage = None

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
        if self.method == "embeddings":
            self.vectors = np.asarray(self.embedder.embed_text(self.tokens))
            # Step 3.5: coverage of tokens by embedder
            self.coverage = self.embedder.calculate_coverage(self.tokens)
        elif self.method == 'cooccurence':
            self.build_coocurrence_matrix()
        else:
            ValueError("Unknown method. Please use either 'embeddings' or 'cooccurence'.")
        # Step 4. calculate autocorrelation
        self.calculate_autocorrelation()
        # Step 5. fit power law
        self.fit_power_law()
        # Step 6. check if word embeddings are normally distributed
        # self.check_normality()
    
    def build_coocurrence_matrix(self):
        cooccurence_matrix = coocurrence_matrix.CoocurrenceMatrix(self.tokens, self.window_size)
        matrix, self.vocabulary = cooccurence_matrix.build_matrix(sparse = self.sparse)
        if self.sparse:
            self.vectors = np.asarray([matrix[[i], :] for i in range(matrix.shape[0])])
        else:
            self.vectors = np.asarray([matrix[i, :] for i in range(matrix.shape[0])])
    
    def calculate_autocorrelation(self):
        max_lag = 0.5 * (len(self.vectors) - 1)
        current_lag = 1
        while current_lag < max_lag:
            self.lags.append(current_lag)
            current_acf_cosine = correlations.calculate_cosine_correlation(self.vectors, L = current_lag, sparse = self.sparse)
            # current_acf_pearson = correlations.calculate_pearson_correlation(self.vectors, L = current_lag)
            self.autocorrelation_cosine.append(current_acf_cosine)
            # self.autocorrelation_pearson.append(current_acf_pearson)
            current_lag = int(np.ceil(current_lag * 1.1))
    
    def fit_power_law(self):
        def _fit_power_law(lags, acf_function):
            try:
                popt, _ = curve_fit(utils.power_law, lags, acf_function, p0 = [1, -1, 1], maxfev = 5000)
                return popt
            except Exception as e:
                print(f"Error fitting power law for autocorrelation: {e}")
        self.power_law_cosine = _fit_power_law(self.lags, self.autocorrelation_cosine)
        # self.power_law_pearson = _fit_power_law(self.lags, self.autocorrelation_pearson)
    
    def check_normality(self):
        self.normality = pg.multivariate_normality(self.vectors)[1] # p-value of Henze-Zirkler test

    def make_plots(self, scales = 'normal', n = 200):
        # if scales == 'log':
        #     lags_this = np.log2(self.lags)
        #     acf_this = np.log2(np.abs(self.autocorrelation_cosine))
        # else:
        #     lags_this = self.lags
        #     acf_this = self.autocorrelation_cosine
        lags_this = self.lags
        def _make_plots(lags_this, coefficients, book_id, acf, acf_name):
            lags_fit = np.linspace(min(lags_this), max(lags_this), n)
            acf_fit = utils.power_law(lags_fit, coefficients[0], coefficients[1], coefficients[2])

            plt.figure()
            sns.scatterplot(x = lags_this, y = acf)
            sns.lineplot(x = lags_fit, y = acf_fit, color = 'red')
            plt.savefig(f"plots/plot{book_id}_{acf_name}.png")

        _make_plots(lags_this, self.power_law_cosine, self.book_id, self.autocorrelation_cosine, "cosine")
        # _make_plots(lags_this, self.power_law_pearson, self.book_id, self.autocorrelation_pearson, "pearson")
