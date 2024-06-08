import pandas as pd
from utils import extract_labels
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="TCGA")
args = argparser.parse_args()


csv_file = f"data/{args.dataset}.csv"
df = pd.read_csv(csv_file)
df.head()
df = extract_labels(df)

df.to_csv(f"data/{args.dataset}_labels.csv", index=False)
