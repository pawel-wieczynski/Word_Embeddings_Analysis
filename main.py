from scripts import pipeline
import pandas as pd
import time

# Research questions:
#   - analyze power-law coefficient distribution among subjects, languages, authors etc.

# Import metadata
metadata = pd.read_csv("SPGC-metadata-2018-07-18.csv")
metadata = metadata[~metadata["file_size"].isna()]
metadata = metadata.iloc[0:8, :] # FOR TESTING PURPOSES
books_ids = metadata["id"].tolist()

# Run pipeline for each book
tic = time.perf_counter()
with open("results.txt", "w") as results:
    for id in books_ids:
        print(f"Analyzing book {id}...")
        book_pipeline = pipeline.TextAnalysisPipeline(
            book_id = id
            , source = "SGPC"
            , language = "english"
            , method = "cooccurence"
            , window_size = 3
            , sparse = False
            , embedder = None
        )
        book_pipeline.run_pipeline()
        results.write(f"{id},{book_pipeline.power_law_cosine[0]},{book_pipeline.power_law_cosine[1]},{book_pipeline.power_law_cosine[2]}\n")
    
toc = time.perf_counter()
print(f"Time elapsed: {toc - tic:.4f}")