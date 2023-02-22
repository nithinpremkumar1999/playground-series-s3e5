import pandas as pd


def target_to_encoding(df: pd.DataFrame):
    return df.map({3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5})


def encoding_to_target(df: pd.DataFrame):
    return df.map({0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8})