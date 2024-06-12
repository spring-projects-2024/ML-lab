import pandas as pd
def preprocess_df(name, extension='csv'):
    csv_file = f'data/{name}.{extension}'
    df = pd.read_csv(csv_file)
    # remove is_true column
    df = df.drop(columns=['is_true', "Variant_Classification", "mutation"])
    # clip all values to positive
    df = df.clip(lower=0)
    # replace column names containing ".." with split[0] on ".."
    df.columns = [s.split("..")[0] for s in df.columns]

    # save to disk
    df.to_csv(f'data/{name}_processed_gpt.{extension}', index=False)