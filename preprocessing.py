import pandas as pd


def impute_na(df, cols_to_impute, group_col, how="mean"):
    df_imp = df.copy()

    if how == "mean":
        for target in cols_to_impute:
            df_imp[target] = df_imp.groupby(group_col)[target].transform(
                lambda x: x.fillna(x.mean())
            )
    elif how == "median":
        for target in cols_to_impute:
            df_imp[target] = df_imp.groupby(group_col)[target].transform(
                lambda x: x.fillna(x.median())
            )
    else:
        for target in cols_to_impute:
            df_imp[target] = df_imp.groupby(group_col)[target].transform(
                lambda x: x.fillna(0)
            )

    return df_imp
