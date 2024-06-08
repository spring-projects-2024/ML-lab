import pandas as pd
from utils import extract_labels

csv_file = "data/TCGA.csv"
df = pd.read_csv(csv_file)
df.head()
df = extract_labels(df)
df.to_csv("data/TCGA_labels.csv", index=False)
