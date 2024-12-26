import os
import pandas as pd

folder_path = "SPGC-tokens-2018-07-18"
file_info = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        file_size_kb = os.path.getsize(file_path) / 1024  # Convert size to kilobytes
        file_info.append({"File Name": file_name.split("_")[0], "File Size (KB)": round(file_size_kb, 2)})

df = pd.DataFrame(file_info)
output_file = "file_sizes.csv"
df.to_csv(output_file, index = False)
