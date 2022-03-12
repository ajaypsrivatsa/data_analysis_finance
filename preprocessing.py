import pandas as pd


def impute_na(df, feature_cols, group_col, how="mean"):
    df_imp = df.copy()

    if how == "mean":
        for target in feature_cols:
            df_imp[target] = df_imp.groupby(group_col)[target].transform(
                lambda x: x.fillna(x.mean())
            )
    elif how == "median":
        for target in feature_cols:
            df_imp[target] = df_imp.groupby(group_col)[target].transform(
                lambda x: x.fillna(x.median())
            )
    else:
        for target in feature_cols:
            df_imp[target] = df_imp.groupby(group_col)[target].transform(
                lambda x: x.fillna(0)
            )

    df_imp.dropna(inplace=True)
    return df_imp


def split_X_y(df, feature_cols, target_col):
    X = df[feature_cols]
    y = df[target_col]
    return X, y
