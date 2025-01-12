from scripts import pipeline
import pandas as pd
import time

# Research questions:
#   - analyze power-law coefficient distribution among subjects, languages, authors etc.

# Import metadata
metadata = pd.read_csv("SPGC-metadata-2018-07-18.csv")
metadata = metadata[(metadata["file_size"] > 0.0) & (metadata["file_size"] < 5000.0)]
metadata = metadata.iloc[0:8, :] # FOR TESTING PURPOSES
books_ids = metadata["id"].tolist()

# Run pipeline for each book
tic = time.perf_counter()
i = 1
n = len(books_ids)
with open("results.txt", "w") as results:
    for id in books_ids:
        print(f"Analyzing book {id}...({i}/{n})")
        book_pipeline = pipeline.TextAnalysisPipeline(
            book_id = id
            , source = "SGPC"
            , language = "english"
            , method = "cooccurence"
            , window_size = 3
            , sparse = False
            , embedder = None
        )
        try:
            book_pipeline.run_pipeline()
            results.write(f"{id},{book_pipeline.power_law_cosine[0]},{book_pipeline.power_law_cosine[1]},{book_pipeline.power_law_cosine[2]}\n")
        except:
            results.write(f"{id},99.9,99.9,99.9\n")
        i += 1
    
toc = time.perf_counter()
print(f"Time elapsed: {toc - tic:.4f}")