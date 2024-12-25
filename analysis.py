import pipeline
import pandas as pd

# Issues / TBD:
#   - add error handling for failed optimization when fitting power-law coefficients ("RuntimeError: Optimal parameters not found...")
#   - add parallel processing to speed up analysis of 50000 books
#   - check of embeddings are Gaussian, if so use also Pearson based autocorrelation
#   - PCA or other dimensionality reduction technique

# Research questions:
#   - analyze power-law coefficient distribution among subjects, languages, authors etc.

# Import metadata
metadata = pd.read_csv("SPGC-metadata-2018-07-18.csv")

# Filter books in English
metadata = metadata[metadata["language"] == "['en']"]

# Initialize lists to store results
books_ids = metadata["id"].tolist()[0:10]
coeffs = []

# Run pipeline for each book
for id in books_ids:
    print(f"Analyzing book {id}...")
    book_pipeline = pipeline.TextAnalysisPipeline(id, "SGPC", "english", "embeddings/word2vec-google-news-300.gensim")
    book_pipeline.run_pipeline()
    print(f"Tokens coverage: {book_pipeline.embedder.calculate_coverage(book_pipeline.tokens)}")
    coeffs.append(book_pipeline.power_law[1])
    book_pipeline.make_plots()

print(coeffs)
# Export results
power_law_results = pd.DataFrame({
    "book_id": books_ids,
    "coeff": coeffs
})

power_law_results.to_csv("power_law_results.csv")
