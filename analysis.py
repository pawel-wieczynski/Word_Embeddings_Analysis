import word_embeddings
import pipeline
import pandas as pd

# Issues / TBD:
#   - add error handling for failed optimization when fitting power-law coefficients ("RuntimeError: Optimal parameters not found...")
#   - add parallel processing to speed up analysis of 50000 books
#   - PCA or other dimensionality reduction technique

# Research questions:
#   - analyze power-law coefficient distribution among subjects, languages, authors etc.

# Load word embeddings
embedder = word_embeddings.WordEmbeddings("embeddings/word2vec-google-news-300.gensim")
embedder.load_embeddings()

# Import metadata
metadata = pd.read_csv("SPGC-metadata-2018-07-18.csv")

# Filter books in English
metadata = metadata[metadata["language"] == "['en']"]

metadata = metadata.iloc[0:8, :] # FOR TESTING PURPOSES
# Initialize lists to store results
books_ids = metadata["id"].tolist()
text_coverage = []
p_value_normality = []
power_law_coefficient_cosine = []
power_law_coefficient_pearson = []

# Run pipeline for each book
for id in books_ids:
    print(f"Analyzing book {id}...")

    book_pipeline = pipeline.TextAnalysisPipeline(id, "SGPC", "english", embedder)
    book_pipeline.run_pipeline()
    text_coverage.append(book_pipeline.coverage)
    p_value_normality.append(book_pipeline.normality)
    power_law_coefficient_cosine.append(book_pipeline.power_law_cosine)
    power_law_coefficient_pearson.append(book_pipeline.power_law_pearson)

# Export results
metadata["Coverage"] = text_coverage
metadata["p-value HZ"] = p_value_normality
metadata["Power law (cosine)"] = power_law_coefficient_cosine
metadata["Power law (Pearson)"] = power_law_coefficient_pearson

metadata.to_csv("power_law_results.csv")
