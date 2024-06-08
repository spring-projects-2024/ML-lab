import pandas as pd


def extract_and_remove_first_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the Variant_Classification column on 'TCGA' and extract the first part.
    Add as a new column and remove from Variant_Classification.
    """
    labels = df["Variant_Classification"].str.split("TCGA", expand=True)
    mask = labels[0].str[-1] == "_"
    labels[0][mask] = labels[0][mask].str[:-1]
    df["first_label"] = labels[0]
    df["Variant_Classification"] = "TCGA" + labels[1]
    return df


def extract_and_remove_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    create a mask. It should have True where the Variant_Classification
    column contains a substring 'True' and False if it contains 'False'.
    Extract the substring and add as a new column.
    """
    trues = df["Variant_Classification"].str.contains("True")
    falses = df["Variant_Classification"].str.contains("False")
    assert (trues | falses).all(), "Some rows do not contain True or False."
    df["is_true"] = trues
    return df


def extract_and_remove_mutation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mutations are in the Variant_Classification column.
    Extract, remove from Variant_Classification and add as a column.
    assert that the length of the column is the same as the length of the dataframe.
    """
    mutation_types = [
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Frame_Shift_Ins",
        "Frame_Shift_Del",
        "In_Frame_Del",
        "Splice_Site",
        "In_Frame_Ins",
        "Splice_Region",
        "Fusion_",
        "Translation_Start_Site",
    ]
    mutation_regex = "|".join(mutation_types)
    mutations = df["Variant_Classification"].str.extract(
        rf"({mutation_regex})_", expand=False
    )
    nan_count = mutations.isna().sum()
    assert nan_count == 0, f"Found {nan_count} NaNs in mutations. Expected 0."
    df["mutation"] = mutations
    # df["Variant_Classification"] = df["Variant_Classification"].str.replace(
    #     rf"(.*)({mutation_regex})_(.*)", "\g<1>\g<3>", regex=True
    # )
    return df


def extract_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels are in the Variant_Classification column, separated by '_'.
    First label and Mutations are handled separately because the separation
    is not consistent. Damn it.
    """
    df = extract_and_remove_boolean(df)
    df = extract_and_remove_mutation(df)
    # labels = df["Variant_Classification"].str.split("_", expand=True)
    # for i in range(labels.shape[1]):
    #     df[f"label_{i}"] = labels[i]
    # df.drop(columns=["Variant_Classification"], inplace=True)
    return df
